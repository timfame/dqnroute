import warnings
import networkx as nx
import numpy as np
import scipy.linalg as lg
import scipy.sparse as sp
import node2vec

from typing import Union
from ..utils import agent_idx

class Embedding(object):
    """
    Abstract class for graph node embeddings.
    """
    def __init__(self, dim, **kwargs):
        self.dim = dim

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], **kwargs):
        raise NotImplementedError()

    def transform(self, nodes):
        raise NotImplementedError()


class HOPEEmbedding(Embedding):
    def __init__(self, dim, proximity='katz', beta=0.01, **kwargs):
        if dim % 2 != 0:
            dim -= dim % 2
            print('HOPE supports only even embedding dimensions; falling back to {}'.format(dim))

        if proximity not in ('katz', 'common-neighbors', 'adamic-adar'):
            raise Exception('Unsupported proximity measure: ' + proximity)

        super().__init__(dim, **kwargs)
        self.proximity = proximity
        self.beta = beta
        self._W = None

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight='weight', real_graph_size=0):
        if type(graph) == nx.DiGraph:
            graph = nx.relabel_nodes(graph, agent_idx)
            A = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes), weight=weight)
            n = graph.number_of_nodes()
        else:
            if real_graph_size > 0:
                new_graph = np.array([0.0] * (real_graph_size * real_graph_size), dtype=np.float32)
                for i in range(real_graph_size):
                    for j in range(real_graph_size):
                        idx = i * real_graph_size + j
                        new_graph[idx] = graph[i][j]
                graph = new_graph.reshape((real_graph_size, real_graph_size))
            A = np.mat(graph)
            n = A.shape[0]

        if self.proximity == 'katz':
            M_g = np.eye(n) - self.beta * A
            M_l = self.beta * A
        elif self.proximity == 'common-neighbors':
            M_g = np.eye(n)
            M_l = A * A
        elif self.proximity == 'adamic-adar':
            M_g = np.eye(n)
            D = np.mat(np.diag([1 / (np.sum(A[:, i]) + np.sum(A[i, :])) for i in range(n)]))
            M_l = A * D * A

        S = np.dot(np.linalg.inv(M_g), M_l)

        # (Changed by Igor):
        # Added v0 parameter, the "starting vector for iteration".
        # Otherwise, the operation behaves nondeterministically, and as a result
        # different nodes may learn different embeddings. I am not speaking about
        # minor floating point errors, the problem was worse.

        #u, s, vt = sp.linalg.svds(S, k=self.dim // 2)
        u, s, vt = sp.linalg.svds(S, k=self.dim // 2, v0=np.ones(A.shape[0]))
        
        X1 = np.dot(u, np.diag(np.sqrt(s)))
        X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
        self._W = np.concatenate((X1, X2), axis=1)

    def transform(self, idx):
        return self._W[idx]


class Node2Vec(Embedding):
    def __init__(self, dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1, **kwargs):
        super().__init__(dim, **kwargs)
        self.emb = None

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight='weight', **kwargs):
        n2v = node2vec.Node2Vec(graph, dimensions=self.dim, weight_key=weight, num_walks=300)
        self.emb = n2v.fit(window=3, min_count=1, batch_words=2)

    def transform(self, node):
        return self.emb.wv[node]


class LaplacianEigenmap(Embedding):
    def __init__(self, dim, renormalize_weights=True, weight_transform='heat',
                 temp=1.0, **kwargs):
        super().__init__(dim, **kwargs)
        self.renormalize_weights = renormalize_weights
        self.weight_transform = weight_transform
        self.temp = temp
        self._X = None

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight='weight', real_graph_size=0):
        if type(graph) == np.ndarray:
            if real_graph_size > 0:
                new_graph = np.array([0.0] * (real_graph_size * real_graph_size), dtype=np.float32)
                for i in range(real_graph_size):
                    for j in range(real_graph_size):
                        idx = i * real_graph_size + j
                        new_graph[idx] = graph[i][j]
                graph = new_graph.reshape((real_graph_size, real_graph_size))

            graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)
            weight = 'weight'

        graph = nx.relabel_nodes(graph.to_undirected(), agent_idx)
        
        if weight is not None:
            if self.renormalize_weights:
                sum_w = sum([ps[weight] for _, _, ps in graph.edges(data=True)])
                avg_w = sum_w / len(graph.edges())
                for u, v, ps in graph.edges(data=True):
                    graph[u][v][weight] /= avg_w

            if self.weight_transform == 'inv':
                for u, v, ps in graph.edges(data=True):
                    graph[u][v][weight] = 1 / ps[weight]

            elif self.weight_transform == 'heat':
                for u, v, ps in graph.edges(data=True):
                    w = ps[weight]
                    graph[u][v][weight] = np.exp(-w*w)

        A = nx.to_scipy_sparse_matrix(graph, nodelist=sorted(graph.nodes),
                                      weight=weight, format='csr', dtype=np.float32)
        
        n, m = A.shape
        diags = A.sum(axis=1)
        D = sp.spdiags(diags.flatten(), [0], m, n, format='csr')
        L = D - A

        # (Changed by Igor):
        # Added v0 parameter, the "starting vector for iteration".
        # Otherwise, the operation behaves nondeterministically, and as a result
        # different nodes may learn different embeddings. I am not speaking about
        # minor floating point errors, the problem was worse.
        
        #values, vectors = sp.linalg.eigsh(L, k=self.dim + 1, M=D, which='SM')
        # print('L')
        # print(L)
        # print('D')
        # print(D)
        # print(A.shape)
        # print(np.ones(A.shape[0]))
        from scipy.sparse.linalg._eigen.arpack import ArpackError
        try:
            values, vectors = sp.linalg.eigsh(L, k=self.dim + 1, M=D, which='SM', v0=np.ones(A.shape[0]))
        except ArpackError as ae:
            print(A)
            print(A.shape, np.ones(A.shape[0]))
            print(L)
            print(D)
            raise ae
     
        # End (Changed by Igor)
        
        self._X = vectors[:, 1:]
        
        if weight is not None and self.renormalize_weights:
            self._X *= avg_w
        #print(self._X.flatten()[:3])

    def transform(self, idx):
        return self._X[idx]


_emb_classes = {
    'hope': HOPEEmbedding,
    'lap': LaplacianEigenmap,
    'node2vec': Node2Vec
}

def get_embedding(alg: str, **kwargs):
    try:
        return _emb_classes[alg](**kwargs)
    except KeyError:
        raise Exception('Unsupported embedding algorithm: ' + alg)

