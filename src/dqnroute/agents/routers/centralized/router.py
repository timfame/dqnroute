import random
import networkx as nx

from typing import List, Tuple, Dict
from dqnroute.agents.base import *
from dqnroute.messages import *


class CentralizedRouter(MasterHandler):
    """
    Centralized router.
    """
    def __init__(self, network: nx.DiGraph, edge_weight='weight', **kwargs):
        super().__init__(**kwargs)
        self.network = network
        self.edge_weight = edge_weight

    def handleSlaveEvent(self, slave_id: AgentId, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, PkgEnqueuedEvent):
            assert event.recipient == slave_id, "Wrong recipient of PkgEnqueuedEvent!"
            self.detectEnqueuedPkg(slave_id)
            return []

        elif isinstance(event, PkgProcessingEvent):
            assert event.recipient == slave_id, "Wrong recipient of PkgProcessingEvent!"
            pkg = event.pkg
            sender = event.sender
            if pkg.dst == slave_id:
                return [PkgReceiveAction(pkg)]
            else:
                to_nbr, msgs = self.routeFrom(sender, slave_id, pkg)
                logger.debug(f'Routing pkg #{pkg.id} on router {slave_id[1]} to router {to_nbr[1]}')
                pkg.node_path.append(slave_id)
                # print('handle event in CentralizedRouter, to_nbr:', to_nbr, 'pkg:', pkg, 'additional_msgs:', msgs)
                return [PkgRouteAction(to_nbr, pkg)] + msgs

        elif isinstance(event, LinkUpdateEvent):
            assert slave_id in [event.u, event.v], "Wrong recipient of LinkUpdateEvent!"
            if isinstance(event, AddLinkEvent):
                return self.addLink(event.u, event.v, event.params)
            elif isinstance(event, RemoveLinkEvent):
                return self.removeLink(event.u, event.v)
            return []

        else:
            return super().handleSlaveEvent(slave_id, event)

    def detectEnqueuedPkg(self, slave_id: AgentId):
        pass

    def addLink(self, u: AgentId, v: AgentId, params={}) -> List[WorldEvent]:
        self.network.add_edge(u, v, **params)
        return []

    def removeLink(self, u: AgentId, v: AgentId) -> List[WorldEvent]:
        self.network.remove_edge(u, v)
        return []

    def routeFrom(self, sender: AgentId, slave_id: AgentId, pkg: Package) -> Tuple[AgentId, List[Message]]:
        raise NotImplementedError()

    def networkStateChanged(self):
        """
        Check if relevant network state has been changed and perform
        some action accordingly.
        Do nothing by default; should be overridden in subclasses.
        """
        pass


class GlobalDynamicRouter(CentralizedRouter):
    """
    Router which routes packets accordingly to global-dynamic routing
    strategy (path is weighted as a sum of queue lenghts on the nodes)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for node in self.network.nodes:
            self.network.nodes[node]['q_len'] = 0

    def detectEnqueuedPkg(self, slave_id: AgentId):
        self.network.nodes[slave_id]['q_len'] += 1

    def routeFrom(self, sender: AgentId, slave_id: AgentId, pkg: Package) -> Tuple[AgentId, List[Message]]:
        w_func = lambda u, v, ps: self.network.nodes[v]['q_len']
        path = nx.dijkstra_path(self.network, slave_id, pkg.dst, weight=w_func)
        self.network.nodes[slave_id]['q_len'] -= 1
        return path[1], []
