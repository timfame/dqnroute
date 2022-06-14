import networkx as nx

from ..common import *
from .common import MultiAgentEnv
from ..factory.router import RouterFactory
from simpy import Environment, Event, Resource, Process


logger = logging.getLogger(DQNROUTE_LOGGER)


class NetworkEnvironment(MultiAgentEnv):
    """
    Class which simulates the behavior of computer network.
    """
    def __init__(self, data_series: EventSeries, pkg_process_delay: int = 0, **kwargs):
        self.pkg_process_delay = pkg_process_delay
        self.data_series = data_series

        super().__init__(**kwargs)

        self.link_queues = {}
        self.router_queues = {}
        for router_id in self.conn_graph.nodes:
            self.link_queues[router_id] = {}
            for _, nbr in self.conn_graph.edges(router_id):
                self.link_queues[router_id][nbr] = Resource(self.env, capacity=1)
            self.router_queues[router_id] = Resource(self.env, capacity=1)

    def makeConnGraph(self, network_cfg, **kwargs) -> nx.Graph:
        if type(network_cfg) == list:
            return make_network_graph(network_cfg)
        elif type(network_cfg) == dict:
            return gen_network_graph(network_cfg['generator'])
        elif type(network_cfg) == nx.Graph:
            return network_cfg.copy()
        else:
            raise Exception(f'Invalid network config: {network_cfg}')

    def makeHandlerFactory(self, **kwargs):
        return RouterFactory(context='network', **kwargs)

    def handleAction(self, from_agent: AgentId, action: Action) -> Event:
        if isinstance(action, PkgRouteAction):
            # print('Env: handle action:', action, 'from_agent:', from_agent)
            to_agent = action.to
            if not self.conn_graph.has_edge(from_agent, to_agent):
                raise Exception("Trying to route to a non-neighbor")

            self.env.process(self._edgeTransfer(from_agent, to_agent, action.pkg))
            return Event(self.env).succeed()

        elif isinstance(action, PkgReceiveAction):
            # print('Env: handle action:', action, 'from_agent:', from_agent, 'time:', self.env.now,
            #       'spent_time:', self.env.now - action.pkg.start_time)
            logger.debug(f"Package #{action.pkg.id} received at node {from_agent[1]} at time {self.env.now}")

            from ...agents.routers.centralized.global_reinforce import PackageHistory
            PackageHistory.finishHistory(action.pkg)

            self.data_series.logEvent('time', self.env.now, self.env.now - action.pkg.start_time)
            return Event(self.env).succeed()

        else:
            return super().handleAction(from_agent, action)

    def handleWorldEvent(self, event: WorldEvent) -> Event:
        # print('Env: handle world event:', event)
        if isinstance(event, PkgEnqueuedEvent):
            self.env.process(self._inputQueue(event.sender, event.recipient, event.pkg))
            return self.passToAgent(event.recipient, event)
        else:
            return super().handleWorldEvent(event)

    def _edgeTransfer(self, from_agent: AgentId, to_agent: AgentId, pkg: Package):
        logger.debug(f"Package #{pkg.id} hop: {from_agent[1]} -> {to_agent[1]}")

        edge_params = self.conn_graph[from_agent][to_agent]
        latency = edge_params['latency']
        bandwidth = edge_params['bandwidth']

        # print(f'Edge transfer, from: {from_agent}, to: {to_agent}, pkg: {pkg}, latency: {latency}, '
        #       f'bandwidth: {bandwidth}, pkg_size: {pkg.size}, time_now: {self.env.now}')

        with self.link_queues[from_agent][to_agent].request() as req:
            yield req
            yield self.env.timeout(pkg.size / bandwidth)

        yield self.env.timeout(latency)
        self.handleWorldEvent(PkgEnqueuedEvent(from_agent, to_agent, pkg))

    def _inputQueue(self, from_agent: AgentId, to_agent: AgentId, pkg: Package):
        with self.router_queues[to_agent].request() as req:
            yield req
            yield self.env.timeout(self.pkg_process_delay)
        self.passToAgent(to_agent, PkgProcessingEvent(from_agent, to_agent, pkg))
