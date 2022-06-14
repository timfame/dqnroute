import networkx as nx

from ...constants import LOG_DATA_DIR
from ..common import *
from .common import MultiAgentEnv
from ...event_series import EventSeries, event_series, MultiEventSeries
from ..factory.router import RouterFactory
from simpy import Environment, Event, Resource, Process
from .common import SimulationRunner
from ..environment.network import NetworkEnvironment


class NetworkRunner(SimulationRunner):
    """
    Class which constructs and runs scenarios in computer network simulation
    environment.
    """
    context = 'network'

    def __init__(self, data_dir=LOG_DATA_DIR + '/network', **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)
        self.all_nodes = None
        self.all_edges = None
        self.broken_edges = None
        self.changes = 0

    def makeDataSeries(self, series_period, series_funcs):
        return MultiEventSeries(time=event_series(series_period, series_funcs))

    def makeMultiAgentEnv(self, created_topology: nx.Graph = None, **kwargs) -> MultiAgentEnv:
        network_cfg = self.run_params['network'] if created_topology is None else created_topology
        return NetworkEnvironment(env=self.env, data_series=self.data_series,
                                  network_cfg=network_cfg,
                                  routers_cfg=self.run_params['settings']['router'],
                                  **self.run_params['settings']['router_env'], **kwargs)

    def relevantConfig(self):
        ps = self.run_params
        ss = ps['settings']
        return (ps['network'], ss['pkg_distr'], ss['router_env'],
                ss['router'].get(self.world.factory.router_type, {}))

    def makeRunId(self, random_seed):
        return '{}-{}'.format(self.world.factory.router_type, random_seed)

    def runProcess(self, random_seed=None):
        if random_seed is not None:
            set_random_seed(random_seed)

        self.all_nodes, self.all_edges, self.broken_edges = self.global_change()

        pkg_distr = self.run_params['settings']['pkg_distr']

        topology_changes_logs = []

        pkg_id = 1
        for period in pkg_distr["sequence"]:
            try:
                action = period["action"]
                pause = period.get("pause", 0)
                is_random = period.get("random", False)

                if action == 'break_link':
                    if is_random:
                        i = random.randint(0, len(self.all_edges) - 1)
                        u, v, ps = self.all_edges.pop(i)
                        self.broken_edges.append((u, v, ps))
                    else:
                        u = ('router', period["u"])
                        v = ('router', period["v"])

                    yield self.world.handleWorldEvent(RemoveLinkEvent(u, v))
                elif action == 'restore_link':
                    if is_random:
                        if len(self.broken_edges) == 0:
                            continue

                        i = random.randint(0, len(self.broken_edges) - 1)
                        u, v, ps = self.broken_edges.pop(i)
                        self.all_edges.append((u, v, ps))
                    else:
                        u = ('router', period["u"])
                        v = ('router', period["v"])

                    yield self.world.handleWorldEvent(AddLinkEvent(u, v, params=self.world.conn_graph.edges[u, v]))
                elif action == 'change_topology':
                    size_delta = int(period['size_delta'])
                    new_topology = self.changeTopology(size_delta)
                    topology_changes_logs.append((self.env.now, size_delta))
                    print('CHANGE. time: ', self.env.now, 'size_delta:', size_delta)
                    yield self.changeWorld(new_topology, pause)

                yield self.env.timeout(pause)

            except KeyError:
                delta = period["delta"]
                try:
                    sources = [('router', v) for v in period["sources"]]
                except KeyError:
                    sources = self.all_nodes

                try:
                    dests = [('router', v) for v in period["dests"]]
                except KeyError:
                    dests = self.all_nodes

                simult_sources = period.get("simult_sources", 1)

                for i in range(0, period["pkg_number"] // simult_sources):
                    srcs = random.sample(sources, simult_sources)
                    for src in srcs:
                        dst = random.choice(dests)

                        from dqnroute.agents.routers.centralized.global_reinforce import PackageHistory
                        PackageHistory.started_packages.add(pkg_id)

                        pkg = Package(pkg_id, DEF_PKG_SIZE, dst, self.env.now, None)  # create empty packet
                        logger.debug(f"Sending random pkg #{pkg_id} from {src} to {dst} at time {self.env.now}")
                        # print(f"Runner: Sending random pkg #{pkg_id} from {src} to {dst} at time {self.env.now}")
                        # print('sending:', self.changes)
                        yield self.world.handleWorldEvent(PkgEnqueuedEvent(('world', 0), src, pkg))
                        pkg_id += 1
                    yield self.env.timeout(delta)

    def changeTopology(self, size_delta) -> nx.Graph:
        delete_delta = -size_delta if size_delta < 0 else 0
        add_delta = size_delta if size_delta > 0 else 0

        def get_random_router_node(g):
            return list(g.nodes)[random.randint(0, len(g.nodes) - 1)]

        if delete_delta > 0 or add_delta > 0:
            G = self.world.conn_graph.copy()
            for _ in range(delete_delta):
                new_g = G.copy()
                node_to_delete = get_random_router_node(new_g)
                new_g.remove_node(node_to_delete)
                is_connected = nx.is_strongly_connected if new_g.is_directed() else nx.is_connected
                while not is_connected(new_g):
                    new_g = G.copy()
                    node_to_delete = get_random_router_node(new_g)
                    new_g.remove_node(node_to_delete)

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

        return G

    def global_change(self):
        all_nodes = list(self.world.conn_graph.nodes)
        all_edges = list(self.world.conn_graph.edges(data=True))
        broken_edges = []
        for node in all_nodes:
            self.world.passToAgent(node, WireInMsg(-1, InitMessage({})))
        return all_nodes, all_edges, broken_edges

    def changeWorld(self, new_topology, pause):
        print('saving')
        if self.world.factory.centralized():
            pass
            # self.world.factory.master_handler.save()
        else:
            some_handler = next(iter(self.world.handlers.values()))
            some_handler.save()
        print('CHANGE WORLD')
        self.changes += 1
        self.world = self.makeMultiAgentEnv(created_topology=new_topology, **self.world_kwargs)
        self.all_nodes, self.all_edges, self.broken_edges = self.global_change()
        return self.env.timeout(pause)

