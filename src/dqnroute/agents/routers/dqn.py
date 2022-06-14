import random
import math
import logging
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import pandas as pd
import pprint
import os

from typing import List, Tuple, Dict, Union
from ..base import *
from .link_state import *
from ...constants import DQNROUTE_LOGGER
from ...messages import *
from ...memory import *
from ...utils import *
from ...networks import *

logger = logging.getLogger(DQNROUTE_LOGGER)


class SharedBrainStorage:
    INSTANCE = None
    PROCESSED_NODES = 0

    @staticmethod
    def load(brain_loader: Callable[[], QNetwork], no_nodes: int) -> QNetwork:
        if SharedBrainStorage.INSTANCE is None:
            SharedBrainStorage.INSTANCE = brain_loader()
        SharedBrainStorage.PROCESSED_NODES += 1
        # print(f"Brain initialization: {SharedBrainStorage.PROCESSED_NODES} / {no_nodes} agents")
        result = SharedBrainStorage.INSTANCE
        if SharedBrainStorage.PROCESSED_NODES == no_nodes:
            # all nodes have been processes
            # prepare this class for possible reuse
            SharedBrainStorage.INSTANCE = None
            SharedBrainStorage.PROCESSED_NODES = 0
        return result


class DQNRouter(LinkStateRouter, RewardAgent):
    """
    A router which implements the DQN-routing algorithm.
    """

    def __init__(self, batch_size: int, mem_capacity: int, nodes: List[AgentId],
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
                b = self._makeBrain(additional_inputs=additional_inputs, **kwargs)
                if random_init:
                    b.init_xavier()
                else:
                    if load_filename is not None:
                        print('loadfilename:', load_filename)
                        rindex = load_filename.rindex('.')
                        current_filename = load_filename[:rindex] + f'_real_graph_size_{b.graph_size}' + \
                                           load_filename[rindex:]
                        b.change_label(current_filename)
                        b.restore()
            return b

        if use_single_neural_network:
            self.brain = SharedBrainStorage.load(load_brain, len(nodes))
        else:
            self.brain = load_brain()
        self.use_single_neural_network = use_single_neural_network

        self.optimizer = get_optimizer(optimizer)(self.brain.parameters())
        self.loss_func = nn.MSELoss()

    def save(self):
        self.brain.save()

    def route(self, sender: AgentId, pkg: Package, allowed_nbrs: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        if self.max_act_time is not None and self.env.time() > self.max_act_time:
            return super().route(sender, pkg, allowed_nbrs)
        else:
            to, estimate, saved_state = self._act(pkg, allowed_nbrs)
            reward = self.registerResentPkg(pkg, estimate, to, saved_state)
            # print('Route. src:', self.id, 'dst:', pkg.dst, 'predict:', to, 'state:', saved_state, 'estimate:',
            #       estimate)
            # print('Route. self.id:', self.id, 'sender:', sender, 'neighbors:', allowed_nbrs, 'to:', to, 'estimate:', estimate, 'pkg:', pkg, 'reward:', reward)
            return to, [OutMessage(self.id, sender, reward)] if sender[0] != 'world' else []

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[Message]:
        if isinstance(msg, RewardMsg):
            action, Q_new, prev_state = self.receiveReward(msg)
            # print('RewardMsg, sender:', sender, 'action:', action, 'Q_new:', Q_new)
            # print('RewardMsg, src:', self.id, 'dst:', msg.pkg.dst, 'predict:', action, 'state:', prev_state, 'sender:',
            #       sender, 'Q_new:', Q_new)
            self.memory.add((prev_state, action[1], -Q_new))
            if self.use_reinforce and len(self.memory.samples) >= self.memory.capacity:
                self._replay()
            # else:
            #     print('cap:', self.memory.capacity, len(self.memory.samples))
            return []
        else:
            return super().handleMsgFrom(sender, msg)

    def _makeBrain(self, additional_inputs=[], initial_size=None, **kwargs):
        if initial_size is None:
            initial_size = len(self.nodes)
        return QNetwork(initial_size, additional_inputs=additional_inputs, one_out=False, **kwargs)

    def _act(self, pkg: Package, allowed_nbrs: List[AgentId]):
        state = self._getNNState(pkg, allowed_nbrs)
        prediction = self._predict(state)[0]
        distr = softmax(prediction, self.min_temp)
        estimate = -np.dot(prediction, distr)

        to = -1
        while ('router', to) not in allowed_nbrs:
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

    def _getAddInput(self, tag, graph_size_delta=0, *args, **kwargs):
        if tag == 'amatrix':
            # print('getAddInput', self.network, self.nodes)
            amatrix = nx.convert_matrix.to_numpy_array(
                self.network, nodelist=self.nodes, weight=self.edge_weight,
                dtype=np.float32)
            gstate = np.ravel(amatrix)
            if graph_size_delta == 0:
                return gstate

            cur_sz = len(self.network.nodes)
            new_sz = cur_sz + graph_size_delta
            new_gstate = np.array([0.0] * (new_sz * new_sz), dtype=np.float32)
            for i in range(cur_sz):
                for j in range(cur_sz):
                    idx = i * cur_sz + j
                    new_idx = i * new_sz + j
                    new_gstate[new_idx] = gstate[idx]
            return new_gstate
        else:
            raise Exception('Unknown additional input: ' + tag)

    def _getNNState(self, pkg: Package, nbrs: List[AgentId], graph_size_delta=0):
        n = len(self.nodes)
        addr = np.array(self.id[1])
        dst = np.array(pkg.dst[1])

        neighbors = list(map(lambda v: v in nbrs, self.nodes))
        for _ in range(graph_size_delta):
            neighbors.append(False)
        neighbours = np.array(neighbors, dtype=np.float32)
        input = [addr, dst, neighbours]

        for inp in self.additional_inputs:
            tag = inp['tag']
            add_inp = self._getAddInput(tag, graph_size_delta=graph_size_delta)
            if tag == 'amatrix':
                add_inp[add_inp > 0] = 1
            input.append(add_inp)

        return tuple(input)

    def _sampleMemStacked(self):
        """
        Samples a batch of episodes from memory and stacks
        states, actions and values from a batch together.
        """
        i_batch = self.memory.sample(self.batch_size)
        batch = [b[1] for b in i_batch]

        states = stack_batch([l[0] for l in batch])
        actions = [l[1] for l in batch]
        values = [l[2] for l in batch]

        return states, actions, values

    def _replay(self):
        """
        Fetches a batch of samples from the memory and fits against them.
        """
        states, actions, values = self._sampleMemStacked()
        preds = self._predict(states)

        for i in range(self.batch_size):
            a = actions[i]
            preds[i][a] = values[i]

        self._train(states, preds)


class DQNRouterOO(DQNRouter):
    """
    Variant of DQN router which uses Q-network with scalar output.
    """

    def _makeBrain(self, additional_inputs=[], initial_size=None, **kwargs):
        if initial_size is None:
            initial_size = len(self.nodes)
        return QNetwork(initial_size, additional_inputs=additional_inputs,
                        one_out=True, **kwargs)

    def _act(self, pkg: Package, allowed_nbrs: List[AgentId]):
        state = self._getNNState(pkg, allowed_nbrs)
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
        to = allowed_nbrs[to_idx]

        return to, estimate, saved_state

    def _nodeRepr(self, node):
        return np.array(node)

    def _getAddInput(self, tag, nbr, graph_size_delta=0):
        return super()._getAddInput(tag, graph_size_delta=graph_size_delta)

    def _getNNState(self, pkg: Package, nbrs: List[AgentId], graph_size_delta=0):
        n = len(self.nodes)
        addr = self._nodeRepr(self.id[1])
        dst = self._nodeRepr(pkg.dst[1])

        get_add_inputs = lambda nbr: [self._getAddInput(inp['tag'], nbr, graph_size_delta=graph_size_delta)
                                      for inp in self.additional_inputs]

        input = [[addr, dst, self._nodeRepr(v[1])] + get_add_inputs(v) for v in nbrs]
        # print('NN State:', input)
        # print('Stacked NN State:', stack_batch(input))
        #
        # def unsqueeze(arr, min_d=2):
        #     if len(arr.shape) == 0:
        #         arr = np.array([arr])
        #     if len(arr.shape) < min_d:
        #         return arr.reshape(arr.shape[0], -1)
        #     return arr
        # print('concatenate:', np.concatenate([unsqueeze(y) for y in stack_batch(input)], axis=1))
        return stack_batch(input)

    def _replay(self):
        states, _, values = self._sampleMemStacked()
        # print(np.array(states, dtype=np.float32).shape, np.array(values, dtype=np.float32).shape)
        replay_loss = self._train(states, np.expand_dims(np.array(values, dtype=np.float32), axis=1))
        # replay_loss = self._train(states, np.expand_dims(np.array(values, dtype=np.float32), axis=0))
        # print('DQN loss:', replay_loss)


class DQNRouterEmb(DQNRouterOO):
    """
    Variant of DQNRouter which uses graph embeddings instead of
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

        super().__init__(**kwargs)

    def _makeBrain(self, additional_inputs=[], initial_size=None, **kwargs):
        if initial_size is None:
            initial_size = len(self.nodes)
        if not self.use_combined_model:
            return QNetwork(
                initial_size, additional_inputs=additional_inputs,
                embedding_dim=self.embedding.dim, one_out=True, **kwargs
            )
        else:
            return CombinedNetwork(
                initial_size, additional_inputs=additional_inputs,
                embedding_dim=self.embedding.dim, one_out=True, **kwargs
            )

    def _nodeRepr(self, node):
        return self.embedding.transform(node).astype(np.float32)

    def networkStateChanged(self):
        num_nodes = len(self.network.nodes)
        num_edges = len(self.network.edges)

        if not self.network_initialized and num_nodes == len(self.nodes) and num_edges == self.init_edges_num:
            self.network_initialized = True

        if self.network_initialized and (num_edges != self.prev_num_edges or num_nodes != self.prev_num_nodes):
            self.prev_num_nodes = num_nodes
            self.prev_num_edges = num_edges
            self.embedding.fit(self.network, weight=self.edge_weight)
            # self.log(pprint.pformat(self.embedding._X), force=self.id[1] == 0)


class DQNRouterNetwork(NetworkRewardAgent, DQNRouter):
    pass


class DQNRouterOONetwork(NetworkRewardAgent, DQNRouterOO):
    pass


class DQNRouterEmbNetwork(NetworkRewardAgent, DQNRouterEmb):
    pass


class ConveyorAddInputMixin:
    """
    Mixin which adds conveyor-specific additional NN inputs support
    """

    def _getAddInput(self, tag, nbr=None):
        if tag == 'work_status':
            return np.array(
                list(map(lambda n: self.network.nodes[n].get('works', False), self.nodes)),
                dtype=np.float32)
        if tag == 'working':
            nbr_works = 1 if self.network.nodes[nbr].get('works', False) else 0
            return np.array(nbr_works, dtype=np.float32)
        else:
            return super()._getAddInput(tag, nbr)


class DQNRouterConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNRouter):
    pass


class DQNRouterOOConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNRouterOO):
    pass


class DQNRouterEmbConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNRouterEmb):
    pass
