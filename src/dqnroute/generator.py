"""
Data generator for performing a supervised learning procedure
"""
from collections import defaultdict

import numpy as np
import networkx as nx
import pandas as pd
import pprint
import copy
import os

from .utils import *
from .constants import *
from .agents import *
from .simulation.factory.router import RouterFactory
from .simulation.runner.conveyor import ConveyorsRunner
from .simulation.runner.network import NetworkRunner


def add_input_cols(tag, dim, graph_size_delta=0):
    if tag == 'amatrix':
        return get_amatrix_cols(dim+graph_size_delta)
    else:
        return mk_num_list(tag + '_', dim)


def unsqueeze(arr, min_d=2):
    if len(arr.shape) == 0:
        arr = np.array([arr])
    if len(arr.shape) < min_d:
        return arr.reshape(arr.shape[0], -1)
    return arr


def _cols(tag, n):
    if n == 1:
        return [tag]
    return mk_num_list(tag + '_', n)


def update_network(router, G):
    if router is None:
        return
    router.network = G
    router.networkStateChanged()


def get_random_router_node(g):
    return list(g.nodes)[random.randint(0, len(g.nodes) - 1)]


def _gen_episodes(
        router_type: str,
        one_out: bool,
        factory: RouterFactory,
        num_episodes: int,
        bar=None,
        sinks=None,
        random_seed=None,
        use_full_topology=True,
        graph_size_delta=0,
) -> pd.DataFrame:
    # print(f'\nUse full topolology: {use_full_topology}')

    G = factory.topology_graph
    nodes = sorted(G.nodes)
    n = len(nodes)

    # edges_indexes = []
    # for from_idx, from_node in enumerate(nodes):
    #     for to_idx, to_node in enumerate(nodes):
    #         if amatrix[from_idx][to_idx] > 0:
    #             # edge exists
    #             edges_indexes.append((from_node, to_node))
    #             assert nx.path_weight(G, [from_node, to_node], weight=factory.edge_weight) == amatrix[from_idx][to_idx]
    # edges_indexes = np.array(edges_indexes)

    # best_transitions = defaultdict(dict)
    # lengths = defaultdict(dict)

    # Shortest path preprocessing
    # for start_node in nodes:
    #     for finish_node in nodes:
    #         if start_node != finish_node and nx.has_path(G, start_node, finish_node):
    #             path = nx.dijkstra_path(G, start_node, finish_node, weight=factory.edge_weight)
    #             length = nx.path_weight(G, path, weight=factory.edge_weight)
    #             best_transitions[start_node][finish_node] = path[1] if len(path) > 1 else start_node
    #             lengths[start_node][finish_node] = length

    if sinks is None:
        sinks = nodes

    additional_inputs = None
    routers = {}
    master_router = factory.makeMasterHandler() if factory.centralized() else None
    node_dim = 1 if one_out else n

    for rid in nodes:
        router = factory._makeHandler(rid)
        if master_router is None:
            update_network(router, G)
        routers[rid] = router
        if additional_inputs is None and master_router is None:
            additional_inputs = router.additional_inputs
    update_network(master_router, G)
    additional_inputs = master_router.additional_inputs if master_router is not None else additional_inputs

    cols = ['addr', 'dst']

    ppo_or_global = 'ppo' in router_type or 'global' in router_type
    amatrix_need = False

    if ppo_or_global:
        for inp in additional_inputs:
            if inp['tag'] == 'amatrix':
                amatrix_need = True
            cols += add_input_cols(inp['tag'], inp.get('dim', n))
        cols += ['next_addr', 'addr_v_func']
        if 'global' in router_type:
            cols += ['nb1', 'nb2', 'nb3']
    else:
        if node_dim == 1:
            cols.append('neighbour')
        else:
            cols += get_neighbors_cols(node_dim + graph_size_delta)

        for inp in additional_inputs:
            cols += add_input_cols(inp['tag'], inp.get('dim', n), graph_size_delta=graph_size_delta)

        if node_dim == 1:
            cols.append('predict')
        else:
            cols += get_target_cols(n + graph_size_delta)

    df = pd.DataFrame(columns=cols)

    pkg_id = 1
    episode = 0
    while episode < num_episodes:
        current_graph = copy.deepcopy(G)
        # if not use_full_topology:
        #     np.random.shuffle(edges_indexes)
        #     remove_edge_count = np.random.randint(0, len(edges_indexes) // 4)
        #
        #     for edge in edges_indexes[:remove_edge_count]:
        #         u = (edge[0][0], int(edge[0][1]))
        #         v = (edge[1][0], int(edge[1][1]))
        #         current_graph.remove_edge(u, v)

        amatrix = nx.convert_matrix.to_numpy_array(
            current_graph, nodelist=nodes, weight=factory.edge_weight, dtype=np.float32)
        gstate = np.ravel(amatrix)

        dst = random.choice(sinks)
        cur = random.choice(only_reachable(current_graph, dst, nodes))
        if cur == dst:
            continue
        router = routers[cur]
        out_nbrs = current_graph.successors(router.id)
        nbrs = only_reachable(current_graph, dst, out_nbrs)

        if len(nbrs) == 0:
            continue

        episode += 1

        def getNNState(pkg, nbrs):
            return router._getNNState(pkg, nbrs, graph_size_delta=graph_size_delta) if master_router is None else master_router._getNNState(router.id, pkg, nbrs)

        # ppo addition
        if ppo_or_global:
            path = nx.dijkstra_path(G, cur, dst, weight=factory.edge_weight)
            # print(path, cur, dst)
            next_addr = path[1]
            full_path_length = -nx.path_weight(G, path, weight=factory.edge_weight)

            row = [cur[1], dst[1]]

            if amatrix_need:
                row = row + gstate.tolist()

            row = row + [next_addr[1], full_path_length]

            if 'global' in router_type:
                rowNbrs = [nbrs[0][1]]
                rowNbrs += [nbrs[1][1]] if len(nbrs) > 1 else [-1]
                rowNbrs += [nbrs[2][1]] if len(nbrs) > 2 else [-1]
                row = row + rowNbrs

            df.loc[len(df)] = row
        else:
            pkg = Package(pkg_id, DEF_PKG_SIZE, dst, 0, None)
            state = list(getNNState(pkg, nbrs))

            def plen_func(v):
                plen = nx.dijkstra_path_length(G, v, dst, weight=factory.edge_weight)
                elen = G.get_edge_data(cur, v)[factory.edge_weight]
                return -(plen + elen)

            if one_out:
                predict = np.fromiter(map(plen_func, nbrs), dtype=np.float32)
                state.append(predict)
                cat_state = np.concatenate([unsqueeze(y) for y in state], axis=1)
                for row in cat_state:
                    df.loc[len(df)] = row
            else:
                predict = np.fromiter(map(lambda i: plen_func(('router', i)) if ('router', i) in nbrs else -INFTY,
                                          range(n + graph_size_delta)),
                                      dtype=np.float32)
                state.append(predict)
                state_ = [unsqueeze(y, 1) for y in state]
                # pprint.pprint(state_)
                cat_state = np.concatenate(state_)
                df.loc[len(df)] = cat_state

        if bar is not None:
            bar.update(1)

    return df


def gen_episodes(
        router_type: str,
        num_episodes: int,
        context: str,
        one_out=True,
        sinks=None,
        bar=None,
        random_seed=None,
        router_params={},
        save_path=None,
        ignore_saved=False,
        run_params={},
        use_full_topology=True,
        delete_delta=0,
        add_delta=0,
        graph_size_delta=0,
        is_dqn=False,
) -> pd.DataFrame:
    if save_path is not None:
        if not ignore_saved and os.path.isfile(save_path):
            df = pd.read_csv(save_path, index_col=False)
            if bar is not None:
                bar.update(num_episodes)
            return df

    RunnerClass = NetworkRunner if context == 'network' else ConveyorsRunner

    router_params['random_init'] = True
    params_override = {
        'settings': {'router': {router_type: {'random_init': True}}}
    }

    original_run_params = run_params['network']
    if delete_delta > 0 or add_delta > 0:
        G = make_network_graph(run_params['network'])
        graph_size_delta = graph_size_delta + delete_delta - add_delta
        for _ in range(delete_delta):
            new_g = G.copy()
            node_to_delete = get_random_router_node(new_g)
            # print('try delete node:', node_to_delete)
            new_g.remove_node(node_to_delete)
            is_connected = nx.is_strongly_connected if new_g.is_directed() else nx.is_connected
            while not is_connected(new_g):
                new_g = G.copy()
                node_to_delete = get_random_router_node(new_g)
                new_g.remove_node(node_to_delete)
            # print('finished deleting node is:', node_to_delete)

            cur_nodes = sorted(new_g.nodes)
            for idx in range(len(cur_nodes)):
                node = sorted(list(new_g.nodes))[idx]
                if node[1] > node_to_delete[1]:
                    existed_edges = list(new_g.edges(node, data=True))
                    new_g.remove_node(node)
                    new_node = (node[0], node[1] - 1)
                    new_g.add_node(new_node)
                    for e in existed_edges:
                        new_g.add_edge(new_node, e[1], **e[2])
            G = new_g
        # print(delete_delta, 'nodes was deleted')
        # print(G, G.nodes, list(G.edges()))

        for _ in range(add_delta):
            new_g = G.copy()
            new_node = ('router', len(new_g.nodes))
            edges_cnt = random.randint(1, 3)
            new_edges = []
            for _ in range(edges_cnt):
                to = get_random_router_node(new_g)
                while (new_node, to) in new_edges:
                    to = get_random_router_node(new_g)
                new_edges.append((new_node, to))
            new_g.add_node(new_node)
            for e in new_edges:
                new_g.add_edge(e[0], e[1], latency=10, bandwidth=1024)
            if nx.is_directed(new_g):
                for e in new_edges:
                    new_g.add_edge(e[1], e[0], latency=10, bandwidth=1024)
            G = new_g
        # print(add_delta, 'nodes was added', G)
        # print(G.nodes)
        if is_dqn:
            graph_size_delta = 0
        run_params['network'] = G

    runner = RunnerClass(router_type=router_type, params_override=params_override, run_params=run_params)
    if runner.context == 'network':
        factory = runner.world.factory
    else:
        factory = runner.world.factory.sub_factory

    df = _gen_episodes(
        router_type, one_out, factory, num_episodes, sinks=sinks, bar=bar, random_seed=random_seed,
        use_full_topology=use_full_topology, graph_size_delta=graph_size_delta)

    run_params['network'] = original_run_params

    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df
