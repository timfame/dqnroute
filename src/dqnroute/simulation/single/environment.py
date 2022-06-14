import networkx as nx

from ...agents import get_router_class
from ...utils import make_network_graph, gen_network_graph


class Environment:

    def __init__(self, data_series, network_cfg, routers_cfg, router_type, pkg_process_delay: int = 0, **kwargs):
        self.topology = self.createTopology(network_cfg)
        self.pkg_process_delay = pkg_process_delay
        self.data_series = data_series

        self.router = get_router_class(router_type)(routers_cfg[router_type])

    def createTopology(self, network_cfg) -> nx.Graph:
        if type(network_cfg) == list:
            return make_network_graph(network_cfg)
        elif type(network_cfg) == dict:
            return gen_network_graph(network_cfg['generator'])
        elif type(network_cfg) == nx.Graph:
            return network_cfg
        else:
            raise Exception(f'Invalid network config: {network_cfg}')

    def processPackage(self, src, dst):
        pass
