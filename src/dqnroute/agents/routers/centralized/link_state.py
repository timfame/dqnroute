import random
import pprint
import networkx as nx

from copy import deepcopy
from typing import List, Tuple, Dict
from ...base import *
from ....messages import *
from ....constants import INFTY



class LSGlobalConveyorMixin(object):
    """
    Mixin for state routers which are working in a conveyor
    environment. Does not inherit `LinkStateRouter` in order
    to maintain sane parent's MRO. Only for usage as a mixin
    in other classes.
    """

    def __init__(self, conv_stop_delay: float, **kwargs):
        super().__init__(**kwargs)
        self.conv_stop_delay = conv_stop_delay
        for node in self.topology:
            self.topology.nodes[node]['works'] = False

    def _conveyorWorks(self, node) -> bool:
        return self.topology.nodes[node].get('works', False)

    def _setConveyorWorkStatus(self, sender: AgentId, works: bool) -> List[Message]:
        if self._conveyorWorks(sender) != works:
            self.topology.nodes[sender]['works'] = works
            return self._announceState()
        return []

    def detectEnqueuedPkg(self, sender: AgentId, pkg: Package) -> List[WorldEvent]:
        msgs = super().detectEnqueuedPkg(sender, pkg)

        allowed_nbrs = only_reachable(self.topology, pkg.dst, self.topology.successors(sender))
        to, _ = self.route(sender, pkg, allowed_nbrs)
        if isinstance(self, RewardAgent):
            self._pending_pkgs.pop(pkg.id)

        return msgs + [PkgRoutePredictionAction(to, pkg)]

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[Message]:
        if isinstance(msg, ConveyorStartMsg):
            return self._setConveyorWorkStatus(sender, True)
        elif isinstance(msg, ConveyorStopMsg):
            return self._setConveyorWorkStatus(sender, False)
        else:
            return super().handleMsgFrom(sender, msg)

    def getState(self):
        sub = super().getState()
        return {'sub': sub, 'works': self._conveyorWorks()}

    def processNewAnnouncement(self, node: int, state) -> Tuple[bool, List[WorldEvent]]:
        sub_ok, msgs = super().processNewAnnouncement(node, state['sub'])
        works_changed = self._conveyorWorks(node) != state['works']
        self.topology.nodes[node]['works'] = state['works']
        return sub_ok or works_changed, msgs