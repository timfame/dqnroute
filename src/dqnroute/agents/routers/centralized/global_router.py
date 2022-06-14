import logging
import networkx as nx
from typing import List, Tuple, Dict

from .reward import RewardGlobal, NetworkRewardGlobal
from ..dqn import SharedBrainStorage, ConveyorAddInputMixin
from ...base import MasterHandler
from ....messages import *
from ....networks.global_q_network import GlobalQNetwork
from ....utils import *
from ....constants import DQNROUTE_LOGGER
from ....memory import *
from ....networks import *


logger = logging.getLogger(DQNROUTE_LOGGER)


class GlobalRouter(MasterHandler):
    def __init__(self, topology: nx.DiGraph, edge_weight='weight', **kwargs):
        super().__init__(**kwargs)
        self.topology = topology
        self.edge_weight = edge_weight

    def handleSlaveEvent(self, slave_id: AgentId, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, PkgEnqueuedEvent):
            assert event.recipient == slave_id, "Wrong recipient of PkgEnqueuedEvent!"

            return self.detectEnqueuedPkg(event.sender, event.pkg)

        elif isinstance(event, PkgProcessingEvent):
            assert event.recipient == slave_id, "Wrong recipient of PkgProcessingEvent!"

            pkg = event.pkg
            if pkg.dst == slave_id:
                return [PkgReceiveAction(pkg)]

            sender = event.sender
            neighbors = [n for n in self.topology.neighbors(slave_id)]

            next_node, msgs = self.route(sender, slave_id, pkg, neighbors)
            assert next_node in neighbors, "Resulting node is not among neighbors!"

            pkg.node_path.append(slave_id)

            # print('handle event in Router, to_nbr:', to_nbr, 'pkg:', pkg, 'additional_msgs:', additional_msgs)
            logger.debug('Routing pkg #{} on router {} to router {}'.format(pkg.id, slave_id[1], next_node[1]))

            return [PkgRouteAction(next_node, pkg)] + msgs

        else:
            return super().handleSlaveEvent(slave_id, event)

    def route(self, sender: AgentId, slave_id: AgentId, pkg: Package, neighbors: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        if len(neighbors) == 1:
            return neighbors[0], []

        path = nx.dijkstra_path(self.topology, slave_id, pkg.dst, weight=self.edge_weight)
        if path[1] in neighbors:
            return path[1], []

        else:
            min_nbr = None
            min_len = INFTY
            for nbr in neighbors:
                elen = self.topology[slave_id][nbr][self.edge_weight]
                plen = nx.dijkstra_path_length(self.topology, nbr, pkg.dst, weight=self.edge_weight)
                if elen + plen < min_len:
                    min_nbr = nbr
                    min_len = elen + plen
            assert min_nbr is not None, "!!sdfsdfs!!!"
            return min_nbr, []

    def addLink(self, src: AgentId, to: AgentId, params={}) -> List[Message]:
        msgs = super().addLink(to, params)
        if (src, to) not in self.topology.edges:
            self.topology.add_edge(to, src, **params)
            self.topology.add_edge(src, to, **params)
            self.topologyUpdate()
        return msgs

    def removeLink(self, src: AgentId, to: AgentId) -> List[Message]:
        msgs = super().removeLink(to)
        if (src, to) in self.topology.edges:
            self.topology.remove_edge(to, src)
            self.topology.remove_edge(src, to)
            self.topologyUpdate()
        return msgs

    def topologyUpdate(self):
        pass

    def detectEnqueuedPkg(self, slave_id: AgentId, pkg: Package) -> List[WorldEvent]:
        return []


class DQNGlobalRouter(GlobalRouter, RewardGlobal):
    """
    A router which implements the DQN-routing algorithm.
    """

    def __init__(self, batch_size: int, mem_capacity: int, nodes: List[AgentId],
                 embedding: Union[dict, Embedding], global_network: dict,
                 optimizer='rmsprop', brain=None, random_init=False, max_act_time=None,
                 additional_inputs=[], softmax_temperature: float = 1.5,
                 probability_smoothing: float = 0.0, load_filename: str = None,
                 use_single_neural_network: bool = False,
                 use_reinforce: bool = True,
                 use_combined_model: bool = False,
                 **kwargs):
        """
        Parameters added by Igor:
        :param softmax_temperature: larger temperature means larger entropy of routing decisions.
        :param probability_smoothing (from 0.0 to 1.0): if greater than 0, then routing probabilities will
            be separated from zero.
        :param load_filename: filename to load the neural network. If None, a new network will be created.
        :param use_single_neural_network: all routers will reference the same instance of the neural network.
            In particular, this very network will be influeced by training steps in all nodes.
        """
        super().__init__(**kwargs)
        batch_size, mem_capacity = 1, 1
        self.batch_size = batch_size
        self.memory = Memory(mem_capacity)
        self.additional_inputs = additional_inputs
        self.nodes = nodes
        self.max_act_time = max_act_time

        # changed by Igor: custom temperatures for softmax:
        self.min_temp = softmax_temperature
        # added by Igor: probability smoothing (0 means no smoothing):
        self.probability_smoothing = probability_smoothing

        self.use_reinforce = use_reinforce
        self.use_combined_model = use_combined_model

        # changed by Igor: brain loading process
        def load_brain():
            b = brain
            if b is None:
                b = self._makeBrain(embedding, global_network, additional_inputs=additional_inputs, **kwargs)
                if random_init:
                    b.init_xavier()
                else:
                    if load_filename is not None:
                        print('Loaded from pretrain')
                        b.change_label(load_filename)
                        b.restore()
            return b

        self.graph_size = None
        if use_single_neural_network:
            self.brain = SharedBrainStorage.load(load_brain, len(nodes))
        else:
            self.brain = load_brain()
        self.use_single_neural_network = use_single_neural_network

        self.optimizer = get_optimizer(optimizer)(self.brain.parameters())
        self.loss_func = nn.MSELoss()

    def save(self):
        self.brain.save()

    def route(self, sender: AgentId, slave_id: AgentId, pkg: Package, neighbors: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        if self.max_act_time is not None and self.env.time() > self.max_act_time:
            return super().route(sender, slave_id, pkg, neighbors)
        else:
            to, estimate, saved_state = self._act(slave_id, pkg, neighbors)
            reward = self.registerResentPkg(slave_id, pkg, estimate, to, saved_state)
            # print('Route. src:', slave_id, 'dst:', pkg.dst, 'predict:', to,  'estimate:', estimate)
            # print('Route. self.id:', self.id, 'slave_id:', slave_id, 'sender:', sender, 'neighbors:', neighbors, 'to:', to, 'estimate:', estimate, 'pkg:', pkg, 'reward:', reward)
            return to, [MasterEvent(slave_id, OutMessage(slave_id, sender, reward))] if sender[0] != 'world' else []

    def handleMsgFrom(self, sender: AgentId, msg: Message, slave_id: AgentId = None) -> List[Message]:
        # print('handleMsgFrom. self.id:', self.id, 'sender:', sender, 'msg:', msg)
        if isinstance(msg, RewardMsg):
            action, Q_new, prev_state = self.receiveReward(slave_id, msg)
            # print('RewardMsg, src:', slave_id, 'dst:', msg.pkg.dst, 'predict:', action, 'sender:', sender, 'Q_new:', Q_new)
            self.memory.add((prev_state, action[1], -Q_new, slave_id))

            if self.use_reinforce and len(self.memory.samples) >= self.memory.capacity:
                self._replay()
            return []
        else:
            return super().handleMsgFrom(sender, msg)

    def _makeBrain(self, embedding: Union[dict, Embedding], global_network: dict,
                   additional_inputs=[], dir_with_models: str = '', **kwargs):
        emb_dim = embedding['dim'] if type(embedding) == dict else embedding.dim
        model_args = {
            # 'scope': dir_with_models,
            'embedding_dim': emb_dim,
            "with_attn": False,
        }
        model_args = dict(**global_network, **model_args)
        print('make brain:', model_args)
        self.graph_size = global_network["n"]
        return GlobalQNetwork(**model_args)

    def _act(self, slave_id: AgentId, pkg: Package, neighbors: List[AgentId]):
        state = self._getNNState(slave_id, pkg, neighbors)
        prediction = self._predict(state)[0]
        distr = softmax(prediction, self.min_temp)
        estimate = -np.dot(prediction, distr)

        to = -1
        while ('router', to) not in neighbors:
            to = sample_distr(distr)

        return ('router', to), estimate, state

    def _predict(self, x):
        self.brain.eval()
        return self.brain(*map(torch.from_numpy, x)).clone().detach().numpy()

    def _train(self, x, y):
        self.brain.train()
        self.optimizer.zero_grad()
        output = self.brain(*map(torch.from_numpy, x))
        loss = self.loss_func(output, torch.from_numpy(y))
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def _getAddInput(self, tag, *args, **kwargs):
        if tag == 'amatrix':
            amatrix = nx.convert_matrix.to_numpy_array(
                self.topology, nodelist=self.nodes, weight=self.edge_weight,
                dtype=np.float32)
            gstate = np.ravel(amatrix)

            cur_sz = len(self.topology.nodes)
            new_sz = self.graph_size
            new_gstate = np.array([0.0] * (new_sz * new_sz), dtype=np.float32)
            for i in range(cur_sz):
                for j in range(cur_sz):
                    idx = i * cur_sz + j
                    new_idx = i * new_sz + j
                    new_gstate[new_idx] = gstate[idx]

            return new_gstate
        else:
            raise Exception('Unknown additional input: ' + tag)

    def _getNNState(self, slave_id: AgentId, pkg: Package, neighbors: List[AgentId]):
        n = len(self.nodes)
        addr = np.array(slave_id[1])
        dst = np.array(pkg.dst[1])

        neighbours = np.array(
            list(map(lambda v: v in neighbors, self.nodes)),
            dtype=np.float32)
        state = [addr, dst, neighbours]

        for inp in self.additional_inputs:
            tag = inp['tag']
            add_inp = self._getAddInput(tag)
            if tag == 'amatrix':
                add_inp[add_inp > 0] = 1
            state.append(add_inp)

        return tuple(state)

    def _sampleMemStacked(self):
        """
        Samples a batch of episodes from memory and stacks
        states, actions and values from a batch together.
        """
        i_batch = self.memory.sample(self.batch_size)
        batch = [b[1] for b in i_batch]

        states = stack_batch([b[0] for b in batch])
        actions = [b[1] for b in batch]
        values = [b[2] for b in batch]
        slave_ids = [b[3] for b in batch]

        return states, actions, values, slave_ids

    def _replay(self):
        """
        Fetches a batch of samples from the memory and fits against them.
        """
        states, actions, values, _ = self._sampleMemStacked()
        predict = self._predict(states)

        for i in range(self.batch_size):
            a = actions[i]
            predict[i][a] = values[i]

        self._train(states, predict)


class DQNGlobalRouterOO(DQNGlobalRouter):
    """
    Variant of DQN global router which uses Q-network with scalar output.
    """

    # def _makeBrain(self, additional_inputs=[], **kwargs):
    #     return QNetwork(len(self.nodes), additional_inputs=additional_inputs,
    #                     one_out=True, **kwargs)

    def _act(self, slave_id: AgentId, pkg: Package, neighbors: List[AgentId]):
        state = self._getNNState(slave_id, pkg, neighbors)
        # print('=========================================================')
        # for _i, n in enumerate(self.topology.nodes):
        #     print(f'Node #{_i} emb:\n\t {self._nodeRepr(n[1])}')
        # print('=========================================================')
        # print('ACT, state:', state)
        prediction = self._predict(state).flatten()
        # print('ACT, prediction:', prediction)
        distr = softmax(prediction, self.min_temp)
        # print('ACT, distr after softmax:', distr)
        # Igor: probability smoothing
        distr = (1 - self.probability_smoothing) * distr + self.probability_smoothing / len(distr)
        # print('ACT, distr after smooth:', distr)

        to_idx = sample_distr(distr)
        estimate = -np.dot(prediction, distr)

        saved_state = [s[to_idx] for s in state]
        # print('ACT, to_idx:', to_idx, 'saved_state:', saved_state)
        to = neighbors[to_idx]
        return to, estimate, saved_state

    def _nodeRepr(self, node):
        return np.array(node)

    def _getAddInput(self, tag, nbr):
        return super()._getAddInput(tag)

    def _getNNState(self, slave_id: AgentId, pkg: Package, neighbors: List[AgentId]):
        n = len(self.nodes)
        src = self._nodeRepr(slave_id[1])
        dst = self._nodeRepr(pkg.dst[1])

        get_add_inputs = lambda nbr: [self._getAddInput(inp['tag'], nbr)
                                      for inp in self.additional_inputs]

        input = [[src, dst, self._nodeRepr(v[1])] + get_add_inputs(v) for v in neighbors]

        # neighbors_embeddings = [self._nodeRepr(v[1]) for v in neighbors]
        # add_zeros = 3 - len(neighbors_embeddings)
        # for _ in range(add_zeros):
        #     neighbors_embeddings.append(np.array([0.0] * len(src), np.float32))
        # if len(neighbors_embeddings) > 3:
        #     neighbors_embeddings = neighbors_embeddings[:3]
        # input = [src, dst] + neighbors_embeddings
        # return tuple(input)

        # print('NN State:', input)
        return stack_batch(input)

    def _replay(self):
        states, actions, values, slave_ids = self._sampleMemStacked()

        # predict = self._predict(states)
        #
        # for i in range(self.batch_size):
        #     a = actions[i]
        #     idx = -1
        #     for n_id, n in enumerate(self.topology.neighbors(slave_ids[i])):
        #         if n[1] == a:
        #             idx = n_id
        #             break
        #     predict[i][idx] = values[i]
        #
        # self._train(states, predict)

        self._train(states, np.expand_dims(np.array(values, dtype=np.float32), axis=1))
        # self._train(states, np.expand_dims(np.array(values, dtype=np.float32), axis=0))


class DQNGlobalRouterEmb(DQNGlobalRouterOO):
    """
    Variant of DQN global Router which uses graph embeddings instead of
    one-hot label encodings.
    """

    def __init__(self, embedding: Union[dict, Embedding], edges_num: int, **kwargs):
        # Those are used to only re-learn the embedding when the topology is changed
        self.prev_num_nodes = 0
        self.prev_num_edges = 0
        self.init_edges_num = edges_num
        self.network_initialized = False

        if type(embedding) == dict:
            self.embedding = get_embedding(**embedding)
        else:
            self.embedding = embedding
        self.embeddings_fitted = False

        kwargs.update({"embedding": self.embedding})
        super().__init__(**kwargs)

    # def _makeBrain(self, additional_inputs=[], **kwargs):
    #     if not self.use_combined_model:
    #         return QNetwork(
    #             len(self.nodes), additional_inputs=additional_inputs,
    #             embedding_dim=self.embedding.dim, one_out=True, **kwargs
    #         )
    #     else:
    #         return CombinedNetwork(
    #             len(self.nodes), additional_inputs=additional_inputs,
    #             embedding_dim=self.embedding.dim, one_out=True, **kwargs
    #         )

    def _nodeRepr(self, node):
        return self.embedding.transform(node).astype(np.float32)

    def _getNNState(self, slave_id: AgentId, pkg: Package, neighbors: List[AgentId]):
        if not self.embeddings_fitted:
            self.topologyUpdate()
            self.embeddings_fitted = True
        return super()._getNNState(slave_id, pkg, neighbors)

    def topologyUpdate(self):
        self.embedding.fit(self.topology, weight=self.edge_weight)

    def networkStateChanged(self):
        num_nodes = len(self.topology.nodes)
        num_edges = len(self.topology.edges)

        if not self.network_initialized and num_nodes == len(self.nodes) and num_edges == self.init_edges_num:
            self.network_initialized = True

        if self.network_initialized and (num_edges != self.prev_num_edges or num_nodes != self.prev_num_nodes):
            self.prev_num_nodes = num_nodes
            self.prev_num_edges = num_edges
            self.embedding.fit(self.topology, weight=self.edge_weight)


class DQNGlobalRouterNetwork(NetworkRewardGlobal, DQNGlobalRouter):
    pass


class DQNGlobalRouterOONetwork(NetworkRewardGlobal, DQNGlobalRouterOO):
    pass


class DQNGlobalRouterEmbNetwork(NetworkRewardGlobal, DQNGlobalRouterEmb):
    pass
