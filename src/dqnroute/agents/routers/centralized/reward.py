import logging
import random

from copy import deepcopy
from typing import List, Tuple, Callable
from ....messages import *
from ....utils import *
from ....constants import DQNROUTE_LOGGER


logger = logging.getLogger(DQNROUTE_LOGGER)


class RewardGlobal(object):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pending_pkgs = {}
        self._debug_pkgs = {}
        self._last_tuple = {}

    def registerResentPkg(self, slave_id: AgentId, pkg: Package, Q_estimate: float, action, data, **kwargs) -> RewardMsg:
        rdata = self._getRewardData(slave_id, pkg, data)
        if pkg.id not in self._pending_pkgs:
            self._pending_pkgs[pkg.id] = {}
        self._pending_pkgs[pkg.id][slave_id] = (action, rdata, data)

        # Igor Buzhinsky's hack to suppress a no-key exception in receiveReward
        self._last_tuple[slave_id] = action, rdata, data

        return self._mkReward(slave_id, pkg, Q_estimate, rdata)

    def receiveReward(self, slave_id: AgentId, msg: RewardMsg):
        try:
            action, old_reward_data, saved_data = self._pending_pkgs[msg.pkg.id].pop(slave_id)
        except KeyError:
            self.log(f'not our package: {msg.pkg}, slave_id: {slave_id}, path:\n  {msg.pkg.node_path}\n', force=True)
            action, old_reward_data, saved_data = self._last_tuple[slave_id]
        reward = self._computeReward(slave_id, msg, old_reward_data)
        return action, reward, saved_data

    def _computeReward(self, slave_id: AgentId, msg: RewardMsg, old_reward_data):
        raise NotImplementedError()

    def _mkReward(self, slave_id: AgentId, pkg: Package, Q_estimate: float, reward_data) -> RewardMsg:
        raise NotImplementedError()

    def _getRewardData(self, slave_id: AgentId, pkg: Package, data):
        raise NotImplementedError()


class NetworkRewardGlobal(RewardGlobal):
    """
    Agent which receives and processes rewards in computer networks.
    """

    def _computeReward(self, slave_id: AgentId, msg: NetworkRewardMsg, time_sent: float):
        time_received = msg.reward_data
        return msg.Q_estimate + (time_received - time_sent)

    def _mkReward(self, slave_id: AgentId, pkg: Package, Q_estimate: float, time_sent: float) -> NetworkRewardMsg:
        return NetworkRewardMsg(slave_id, pkg, Q_estimate, time_sent)

    def _getRewardData(self, slave_id: AgentId, pkg: Package, data):
        return self.env.time()


class ConveyorRewardGlobal(RewardGlobal):
    """
    Agent which receives and processes rewards in conveyor networks
    """

    def __init__(self, energy_reward_weight, **kwargs):
        super().__init__(**kwargs)
        self._e_weight = energy_reward_weight

    def _computeReward(self, slave_id: AgentId, msg: ConveyorRewardMsg, old_reward_data):
        time_sent, _ = old_reward_data
        time_processed, energy_gap = msg.reward_data
        time_gap = time_processed - time_sent

        # self.log('time gap: {}, nrg gap: {}'.format(time_gap, energy_gap), True)
        return msg.Q_estimate + time_gap + self._e_weight * energy_gap

    def _mkReward(self, slave_id: AgentId, bag: Bag, Q_estimate: float, reward_data) -> ConveyorRewardMsg:
        time_processed, energy_gap = reward_data
        return ConveyorRewardMsg(slave_id, bag, Q_estimate, time_processed, energy_gap)

    def _getRewardData(self, slave_id: AgentId, bag: Bag, data):
        cur_time = self.env.time()
        # delay = self.conv_stop_delay
        # consumption = self.env.energy_consumption()
        # stop_time = self.env.get_scheduled_stop()
        # time_gap = delay - max(0, stop_time - cur_time)
        # energy_gap = consumption * time_gap
        energy_gap = self.env.get_total_nrg() - self.env.get_prev_total_nrg()
        return cur_time, energy_gap / 5000  # zhang consumption for 1 sec