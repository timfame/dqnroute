from .common import HandlerFactory
from ..common import *
from ...agents import get_router_class
from ...agents.routers import LinkStateRouter
from ..training import TrainingRouterClass
from copy import deepcopy


class RouterFactory(HandlerFactory):
    def __init__(self, router_type, routers_cfg, context=None,
                 topology_graph=None, training_router_type=None, **kwargs):
        RouterClass = get_router_class(router_type, context)
        self.context = context
        self.router_cfg = routers_cfg.get(router_type, {})
        self.edge_weight = 'latency' if context == 'network' else 'length'
        self._dyn_env = None

        if training_router_type is None:
            self.training_mode = False
            self.router_type = router_type
            self.RouterClass = RouterClass
        else:
            self.training_mode = True
            TrainerClass = get_router_class(training_router_type, context)
            self.router_type = f'training__{router_type}__{training_router_type}'
            self.RouterClass = TrainingRouterClass(RouterClass, TrainerClass, **kwargs)

        kwargs.update({"topology_graph": topology_graph})
        super().__init__(**kwargs)

        if self.training_mode:
            dummy = RouterClass(
                **self._handlerArgs(('router', 0), neighbours=[], random_init=True))
            self.brain = dummy.brain
            self.router_cfg['brain'] = self.brain

    def dynEnv(self):
        if self._dyn_env is None:
            return DynamicEnv(time=lambda: self.env.now)
        else:
            return self._dyn_env

    def useDynEnv(self, env):
        self._dyn_env = env

    def makeMasterHandler(self) -> MasterHandler:
        G = self.topology_graph
        kwargs = {}
        kwargs.update({
            'env': self.dynEnv(),
            'network': G,
            'topology': G,
            'edge_weight': self.edge_weight,
            'nodes': sorted(list(G.nodes())),
            'edges_num': len(G.edges()),  # small hack to make link-state initialization simpler
        })
        kwargs.update(self.router_cfg)
        return self.RouterClass(**kwargs)

    def _handlerArgs(self, agent_id, **kwargs):
        G = self.topology_graph
        kwargs.update({
            'env': self.dynEnv(),
            'id': agent_id,
            'edge_weight': self.edge_weight,
            'nodes': sorted(list(G.nodes())),
            'edges_num': len(G.edges()),  # small hack to make link-state initialization simpler
        })
        kwargs.update(self.router_cfg)

        if issubclass(self.RouterClass, LinkStateRouter):
            kwargs['adj_links'] = G.adj[agent_id]
        return kwargs

    def makeHandler(self, agent_id: AgentId, **kwargs) -> MessageHandler:
        assert agent_id[0] == 'router', "Only routers are allowed in computer network"
        return self.RouterClass(**self._handlerArgs(agent_id, **kwargs))

    def centralized(self):
        return issubclass(self.RouterClass, MasterHandler)
