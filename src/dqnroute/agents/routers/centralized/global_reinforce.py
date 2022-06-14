from .global_router import GlobalRouter
from .reward import RewardGlobal, NetworkRewardGlobal, ConveyorRewardGlobal
from .link_state import LSGlobalConveyorMixin
from ..link_state import *
from ...base import *
from ....constants import DQNROUTE_LOGGER
from ....messages import *
from ....memory import *
from ....utils import *
from ....networks import *

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from src.dqnroute.networks.actor_critic_networks import PPOActor


class PackageHistory:
    epoch_size = 40

    routers = defaultdict(dict)
    rewards = defaultdict(list)
    log_probs = defaultdict(list)

    finished_packages = set()
    started_packages = set()

    @staticmethod
    def addToHistory(pkg: Package, router, reward: float, log_prob):
        assert pkg.id not in PackageHistory.finished_packages

        PackageHistory.routers[pkg.id][router.id] = router
        PackageHistory.rewards[pkg.id].append(reward)
        PackageHistory.log_probs[pkg.id].append(log_prob)

    @staticmethod
    def finishHistory(pkg: Package):
        PackageHistory.finished_packages.add(pkg.id)

    @staticmethod
    def learn():
        print('Learn')
        eps = 1e-8
        gamma = 0.99

        torch.autograd.set_detect_anomaly(True)

        all_routers_needed = dict()
        # print(PackageHistory.finished_packages)
        # print(PackageHistory.started_packages)
        # print(PackageHistory.finished_packages & PackageHistory.started_packages)

        packages = PackageHistory.finished_packages & PackageHistory.started_packages

        for package_idx in packages:
            for router in PackageHistory.routers[package_idx].values():
                all_routers_needed[router.id] = router

        for router in all_routers_needed.values():
            router.actor.init_optimizer(router.actor.parameters())
            router.actor.optimizer.zero_grad()

        for package_idx in packages:
            rewards_package = PackageHistory.rewards[package_idx]
            log_probs_package = PackageHistory.log_probs[package_idx]

            if len(rewards_package) < 2:
                continue

            # print('package_idx:', package_idx)
            # print('rewards_package:', rewards_package)
            # print('log_probs_package:', log_probs_package)

            R = 0
            policy_losses = []
            returns = []

            for r in rewards_package[::-1]:
                R = r + gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            # print('returns:', returns)
            for log_prob, R in zip(log_probs_package, returns):
                policy_losses.append(-log_prob * R)

            for policy_loss in policy_losses:
                policy_loss.backward()

        for router in all_routers_needed.values():
            # torch.nn.utils.clip_grad_norm(router.actor.parameters(), 1.0)
            router.actor.optimizer.step()
            # pass

        PackageHistory.routers = defaultdict(dict)
        PackageHistory.rewards = defaultdict(list)
        PackageHistory.log_probs = defaultdict(list)

        PackageHistory.finished_packages = set()
        PackageHistory.started_packages = set()


class GlobalReinforce(GlobalRouter, RewardGlobal):
    def __init__(
            self,
            distance_function: str,
            nodes: List[AgentId],
            embedding: Union[dict, Embedding],
            edges_num: int,
            global_network: dict,
            max_act_time=None,
            additional_inputs=[],
            dir_with_models: str = '',
            load_filename: str = None,
            use_single_network: bool = False,
            **kwargs
    ):
        super(GlobalReinforce, self).__init__(**kwargs)

        self.distance_function = get_distance_function(distance_function)
        self.nodes = nodes
        self.init_edges_num = edges_num
        self.max_act_time = max_act_time
        self.additional_inputs = additional_inputs
        self.prev_num_nodes = 0
        self.prev_num_edges = 0
        self.network_initialized = False
        self.embeddings_fitted = False
        self.use_single_network = use_single_network

        if type(embedding) == dict:
            self.embedding = get_embedding(**embedding)
        else:
            self.embedding = embedding


        # Create net architecture
        model_args = {
            'scope': dir_with_models,
            'embedding_dim': embedding['dim']
        }
        model_args = dict(**global_network, **model_args)
        model = GlobalNetwork(**model_args)
        # Init net weights
        if load_filename is not None:
            # Get pretrained net from file
            print('RESTORING')
            model.change_label(load_filename)
            # actor_model._label = load_filename
            model.restore()
        else:
            print('NEW MODEL')
            # Create net from scratch
            model.init_xavier()

        self.actor = model
        self.memory = ReinforceMemory()

        self.horizon = 64
        self.discount_factor = 0.99  # gamma
        self.eps = 1e-6

    def route(self, sender: AgentId, slave_id: AgentId, pkg: Package, allowed_nbrs: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        if self.max_act_time is not None and self.env.time() > self.max_act_time:
            return super().route(sender, slave_id, pkg, allowed_nbrs)
        else:
            to, addr_idx, dst_idx, action_idx, action_log_prob = self._act(slave_id, pkg, allowed_nbrs)
            reward = self.registerResentPkg(
                slave_id,
                pkg,
                addr_idx=addr_idx,
                dst_idx=dst_idx,
                action_idx=action_idx,
                action_log_prob=action_log_prob,
                allowed_neighbours=allowed_nbrs
            )
            return to, [MasterEvent(slave_id, OutMessage(slave_id, sender, reward))] if sender[0] != 'world' else []

    def handleMsgFrom(self, sender: AgentId, msg: Message, slave_id: AgentId = None) -> List[WorldEvent]:
        if isinstance(msg, RewardMsg):
            addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, reward = self.receiveReward(slave_id, msg)
            # print(f'Pkg {msg.pkg.id, msg.pkg.dst[1]}. {self.id} -> {allowed_neighbours[action_idx]}, Sender: {sender}, Time: {self.env.time()}')
            # self.memory.add((addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, reward))
            print('routed and reward got:', msg.pkg.id)
            PackageHistory.addToHistory(msg.pkg, self, reward, action_log_prob)

            if len(PackageHistory.finished_packages) >= 64:
                PackageHistory.learn()

            return []
        else:
            return super().handleMsgFrom(sender, msg)

    def _act(self, slave_id: AgentId, pkg: Package, neighbors: List[AgentId]):
        # allowed_nbrs_emb = np.array(list(map(lambda neighbour: self._nodeRepr(neighbour[1]), allowed_nbrs)))
        allowed_nbrs_emb = list(map(lambda neighbour: self._nodeRepr(neighbour[1]), neighbors))
        allowed_nbrs_emb_tensor = torch.FloatTensor(allowed_nbrs_emb)
        addr_idx = slave_id[1]
        dst_idx = pkg.dst[1]
        addr_emb = torch.FloatTensor(self._nodeRepr(addr_idx))
        dst_emb = torch.FloatTensor(self._nodeRepr(dst_idx))

        neighbors_embeddings = [self._nodeRepr(v[1]) for v in neighbors]
        add_zeros = 3 - len(neighbors_embeddings)
        for _ in range(add_zeros):
            neighbors_embeddings.append(np.array([1000000.0] * len(addr_emb), np.float32))
        if len(neighbors_embeddings) > 3:
            neighbors_embeddings = neighbors_embeddings[:3]
        nb1_emb = torch.FloatTensor(neighbors_embeddings[0])
        nb2_emb = torch.FloatTensor(neighbors_embeddings[1])
        nb3_emb = torch.FloatTensor(neighbors_embeddings[2])

        # 1. Actor generates next embedding
        predicted_next_emb = self._actorPredict(addr_emb, dst_emb, nb1_emb, nb2_emb, nb3_emb)
        # predicted_next_emb.register_hook(lambda t: print(f'hook predicted_next_emb:\n {t}'))

        # 2. Compute distances from next embedding (see step 1.) and allowed neighbours
        dist_to_nbrs = self.distance_function(allowed_nbrs_emb_tensor, predicted_next_emb)
        nbrs_prob = Categorical(F.softmax(1 / (dist_to_nbrs + self.eps), dim=0))
        # TODO debug

        # 3. Get sample from allowed neighbours based on probability
        next_nbr_idx = nbrs_prob.sample()
        action_log_prob = nbrs_prob.log_prob(next_nbr_idx)
        action_idx = next_nbr_idx.item()

        to = neighbors[action_idx]
        return to, addr_idx, dst_idx, action_idx, action_log_prob

    def _actorPredict(self, addr_emb, dst_emb, nb1_emb, nb2_emb, nb3_emb):
        return self.actor.forward(addr_emb, dst_emb, nb1_emb, nb2_emb, nb3_emb)

    def networkStateChanged(self):
        num_nodes = len(self.topology.nodes)
        num_edges = len(self.topology.edges)

        if not self.network_initialized and num_nodes == len(self.nodes) and num_edges == self.init_edges_num:
            self.network_initialized = True

        if self.network_initialized and (num_edges != self.prev_num_edges or num_nodes != self.prev_num_nodes):
            self.prev_num_nodes = num_nodes
            self.prev_num_edges = num_edges
            self.embedding.fit(self.topology, weight=self.edge_weight)
            # self.log(pprint.pformat(self.embedding._X), force=self.id[1] == 0)

    def _nodeRepr(self, node):
        if not self.embeddings_fitted:
            self.topologyUpdate()
            self.embeddings_fitted = True
        return self.embedding.transform(node).astype(np.float32)

    def topologyUpdate(self):
        self.embedding.fit(self.topology, weight=self.edge_weight)


class GlobalReinforceNetwork(NetworkRewardGlobal, GlobalReinforce):
    def registerResentPkg(self, slave_id: AgentId, pkg: Package, **kwargs) -> RewardMsg:
        addr_idx = kwargs['addr_idx']
        dst_idx = kwargs['dst_idx']
        action_idx = kwargs['action_idx']
        action_log_prob = kwargs['action_log_prob']
        allowed_neighbours = kwargs['allowed_neighbours']

        reward_data = self._getRewardData(slave_id, pkg, None)

        if slave_id not in self._pending_pkgs:
            self._pending_pkgs[slave_id] = {}
        self._pending_pkgs[slave_id][pkg.id] = (addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, reward_data)

        self._last_tuple[slave_id] = (addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, reward_data)

        return self._mkReward(slave_id, pkg, reward_data)

    def _mkReward(self, slave_id: AgentId, pkg: Package, reward_data) -> NetworkRewardMsg:
        # Q_estimate is ignored in our setting
        time_processed = reward_data
        return NetworkRewardMsg(slave_id, pkg, 0, time_processed)

    def receiveReward(self, slave_id: AgentId, msg: RewardMsg):
        try:
            addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, old_reward_data = \
                self._pending_pkgs[slave_id].pop(msg.pkg.id)
        except KeyError:
            addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, old_reward_data = \
                self._last_tuple[slave_id]

        reward = self._computeReward(slave_id, msg, old_reward_data)
        return addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, reward

    def _computeReward(self, slave_id: AgentId, msg: ConveyorRewardMsg, old_reward_data):
        time_sent = old_reward_data
        time_processed = msg.reward_data
        time_gap = time_processed - time_sent
        return -time_gap


class GlobalReinforceConveyor(LSGlobalConveyorMixin, ConveyorRewardGlobal, GlobalReinforce):
    def registerResentPkg(self, slave_id: AgentId, pkg: Package, **kwargs) -> RewardMsg:
        addr_idx = kwargs['addr_idx']
        dst_idx = kwargs['dst_idx']
        action_idx = kwargs['action_idx']
        action_log_prob = kwargs['action_log_prob']
        allowed_neighbours = kwargs['allowed_neighbours']

        reward_data = self._getRewardData(slave_id, pkg, None)

        if pkg.id not in self._pending_pkgs:
            self._pending_pkgs = {}
        self._pending_pkgs[pkg.id][slave_id] = (addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, reward_data)

        return self._mkReward(slave_id, pkg, reward_data)

    def _mkReward(self, slave_id: AgentId, bag: Bag, reward_data) -> ConveyorRewardMsg:
        # Q_estimate is ignored in our setting
        time_processed, energy_gap = reward_data
        return ConveyorRewardMsg(slave_id, bag, 0, time_processed, energy_gap)

    def receiveReward(self, slave_id: AgentId, msg: RewardMsg):
        addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, old_reward_data = \
            self._pending_pkgs[msg.pkg.id].pop(slave_id)

        reward = self._computeReward(slave_id, msg, old_reward_data)
        return addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, reward

    def _computeReward(self, slave_id: AgentId, msg: ConveyorRewardMsg, old_reward_data):
        time_sent, _ = old_reward_data
        time_processed, energy_gap = msg.reward_data
        time_gap = time_processed - time_sent
        return -(time_gap + self._e_weight * energy_gap)
