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
from ...base import *
from ..link_state import *
from ....constants import DQNROUTE_LOGGER
from ....messages import *
from ....memory import *
from ....utils import *
from ....networks import *
from .router import CentralizedRouter
from ..dqn import SharedBrainStorage, ConveyorAddInputMixin
from ..link_state import AbstractStateHandler

logger = logging.getLogger(DQNROUTE_LOGGER)


class AbstractStateCentralizedHandler(MasterHandler):
    """
    A router which implements a link-state protocol but the state is
    not necessarily link-state and can be abstracted out.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seq_num = 0
        self.announcements = {}

    def init(self, config) -> List[WorldEvent]:
        msgs = super().init(config)
        return msgs

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[WorldEvent]:
        if isinstance(msg, StateAnnouncementMsg):
            if msg.node == self.id:
                return []

            if msg.node not in self.announcements or self.announcements[msg.node].seq < msg.seq:
                self.announcements[msg.node] = msg
                broadcast, msgs = self.processNewAnnouncement(msg.node, msg.state)
                self.networkStateChanged()

                if broadcast:
                    msgs += self.broadcast(msg, exclude=[sender])
                return msgs
            return []

        else:
            return super().handleMsgFrom(sender, msg)

    def _announceState(self, node: AgentId) -> List[Message]:
        state = self.getState(node)
        print('announce state', state)
        if state is None:
            return []

        announcement = StateAnnouncementMsg(self.id, self.seq_num, state)
        self.seq_num += 1
        return self.broadcast(announcement)

    def networkStateChanged(self):
        """
        Check if relevant network state has been changed and perform
        some action accordingly.
        Do nothing by default; should be overridden in subclasses.
        """
        pass

    def getState(self, node: AgentId):
        """
        Should be overridden by subclasses. If returned state is `None`,
        no announcement is made.
        """
        raise NotImplementedError()

    def processNewAnnouncement(self, node: int, state) -> Tuple[bool, List[WorldEvent]]:
        raise NotImplementedError()


class LinkStateCentralizedRouter(CentralizedRouter, AbstractStateCentralizedHandler):

    def init(self, config) -> List[WorldEvent]:
        msgs = super().init(config)
        return msgs

    def addLink(self, u: AgentId, v: AgentId, params={}) -> List[WorldEvent]:
        msgs = super().addLink(u, v, params)
        # self.network.add_edge(u, v, **params)
        return msgs + self._announceState(u)

    def removeLink(self, u: AgentId, v: AgentId) -> List[WorldEvent]:
        msgs = super().removeLink(u, v)
        # self.network.remove_edge(u, v)
        return msgs + self._announceState(u)

    def getState(self, node: AgentId):
        return self.network.adj[node]

    def processNewAnnouncement(self, node: AgentId, neighbours) -> Tuple[bool, List[WorldEvent]]:
        changed = False

        if neighbours is None:
            neighbours = self.network.adj[node]

        for (m, params) in neighbours.items():
            if self.network.get_edge_data(node, m) != params:
                self.network.add_edge(node, m, **params)
                changed = True

        for m in set(self.network.nodes()) - set(neighbours.keys()):
            try:
                self.network.remove_edge(node, m)
                changed = True
            except nx.NetworkXError:
                pass

        return changed, []


class DQNCentralizedRouter(LinkStateCentralizedRouter, RewardAgent):
    """
    A router which implements the centralized DQN-routing algorithm.
    """

    def __init__(self, batch_size: int = None, mem_capacity: int = None, nodes: List[AgentId] = None,
                 optimizer='rmsprop', brain=None, random_init=False, max_act_time=None,
                 additional_inputs=[], softmax_temperature: float = 1.5,
                 probability_smoothing: float = 0.0, load_filename: str = None,
                 use_single_neural_network: bool = False,
                 use_reinforce: bool = True,
                 use_combined_model: bool = False,
                 **kwargs):

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
                        b.change_label(load_filename)
                        b.restore()
            return b

        if use_single_neural_network:
            self.brain = SharedBrainStorage.load(load_brain, len(nodes))
        else:
            self.brain = load_brain()
        self.use_single_neural_network = use_single_neural_network

        self.optimizer = get_optimizer(optimizer)(self.brain.parameters())
        self.loss_func = nn.MSELoss()

    def detectEnqueuedPkg(self, slave_id: AgentId):
        return []

    def routeFrom(self, sender: AgentId, slave_id: AgentId, pkg: Package) -> Tuple[AgentId, List[Message]]:
        neighbours = [v for _, v in self.network.edges(slave_id)]
        next_agent, estimate, saved_state = self._act(pkg, neighbours)
        reward = self.registerResentPkg(pkg, estimate, next_agent, saved_state)
        return next_agent, [MasterEvent(self.id, reward, sender=sender)] if sender[0] != 'world' else []

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[Message]:
        unpacked_msg = msg
        if isinstance(unpacked_msg, RewardMsg):
            action, Q_new, prev_state = self.receiveReward(unpacked_msg)
            self.memory.add((prev_state, action[1], -Q_new))

            if self.use_reinforce:
                self._replay()
            return []
        else:
            return super().handleMsgFrom(sender, unpacked_msg)

    def _try_unpack(self, msg):
        if not isinstance(msg, OutMessage):
            return msg
        try:
            return msg.inner_msg
        except KeyError:
            return msg

    def _makeBrain(self, additional_inputs=[], **kwargs):
        return QNetwork(len(self.nodes), additional_inputs=additional_inputs, one_out=False, **kwargs)

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

    def _getAddInput(self, tag, *args, **kwargs):
        if tag == 'amatrix':
            amatrix = nx.convert_matrix.to_numpy_array(
                self.network, nodelist=self.nodes, weight=self.edge_weight,
                dtype=np.float32)
            gstate = np.ravel(amatrix)
            return gstate
        else:
            raise Exception('Unknown additional input: ' + tag)

    def _getNNState(self, pkg: Package, nbrs: List[AgentId]):
        n = len(self.nodes)
        addr = np.array(self.id[1])
        dst = np.array(pkg.dst[1])

        neighbours = np.array(
            list(map(lambda v: v in nbrs, self.nodes)),
            dtype=np.float32)
        input = [addr, dst, neighbours]

        for inp in self.additional_inputs:
            tag = inp['tag']
            add_inp = self._getAddInput(tag)
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


class DQNCentralizedRouterOO(DQNCentralizedRouter):
    """
    Variant of DQN router which uses Q-network with scalar output.
    """

    def _makeBrain(self, additional_inputs=[], **kwargs):
        return QNetwork(len(self.nodes), additional_inputs=additional_inputs,
                        one_out=True, **kwargs)

    def _act(self, pkg: Package, allowed_nbrs: List[AgentId]):
        state = self._getNNState(pkg, allowed_nbrs)
        prediction = self._predict(state).flatten()
        distr = softmax(prediction, self.min_temp)

        # Igor: probability smoothing
        distr = (1 - self.probability_smoothing) * distr + self.probability_smoothing / len(distr)

        to_idx = sample_distr(distr)
        estimate = -np.dot(prediction, distr)

        saved_state = [s[to_idx] for s in state]
        to = allowed_nbrs[to_idx]
        return to, estimate, saved_state

    def _nodeRepr(self, node):
        return np.array(node)

    def _getAddInput(self, tag, nbr):
        return super()._getAddInput(tag)

    def _getNNState(self, pkg: Package, nbrs: List[AgentId]):
        self.try_fit_embedding()
        n = len(self.nodes)
        addr = self._nodeRepr(self.id[1])
        dst = self._nodeRepr(pkg.dst[1])

        get_add_inputs = lambda nbr: [self._getAddInput(inp['tag'], nbr)
                                      for inp in self.additional_inputs]

        input = [[addr, dst, self._nodeRepr(v[1])] + get_add_inputs(v) for v in nbrs]
        return stack_batch(input)

    def _replay(self):
        states, _, values = self._sampleMemStacked()
        self._train(states, np.expand_dims(np.array(values, dtype=np.float32), axis=0))

    def try_fit_embedding(self):
        pass


class DQNCentralizedRouterEmb(DQNCentralizedRouterOO):
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

    def _makeBrain(self, additional_inputs=[], **kwargs):
        if not self.use_combined_model:
            return QNetwork(
                len(self.nodes), additional_inputs=additional_inputs,
                embedding_dim=self.embedding.dim, one_out=True, **kwargs
            )
        else:
            return QNetwork(
                len(self.nodes), additional_inputs=additional_inputs,
                embedding_dim=self.embedding.dim, one_out=True, **kwargs
            )

    def _nodeRepr(self, node):
        return self.embedding.transform(node).astype(np.float32)

    def try_fit_embedding(self):
        num_nodes = len(self.network.nodes)
        num_edges = len(self.network.edges)
        if num_edges != self.prev_num_edges or num_nodes != self.prev_num_nodes:
            self.prev_num_nodes = num_nodes
            self.prev_num_edges = num_edges
            self.embedding.fit(self.network, weight=self.edge_weight)

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


class DQNCentralizedRouterNetwork(NetworkRewardAgent, DQNCentralizedRouter):
    pass


class DQNCentralizedRouterOONetwork(NetworkRewardAgent, DQNCentralizedRouterOO):
    pass


class DQNCentralizedRouterEmbNetwork(NetworkRewardAgent, DQNCentralizedRouterEmb):
    pass


class DQNCentralizedRouterConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNCentralizedRouter):
    pass


class DQNCentralizedRouterOOConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNCentralizedRouterOO):
    pass


class DQNCentralizedRouterEmbConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNCentralizedRouterEmb):
    pass
