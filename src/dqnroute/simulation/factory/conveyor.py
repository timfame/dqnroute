from ...constants import *
from ..common import *
from .common import HandlerFactory
from .router import RouterFactory


class ConveyorFactory(HandlerFactory):
    def __init__(self, router_type, routers_cfg, topology,
                 conn_graph, conveyor_cfg, energy_consumption, max_speed,
                 conveyor_models, oracle=True, **kwargs):
        self.router_type = router_type
        self.router_cfg = routers_cfg.get(router_type, {})
        self.conveyor_cfg = conveyor_cfg
        self.topology = topology
        self.conveyor_cfg = conveyor_cfg
        self.energy_consumption = energy_consumption
        self.max_speed = max_speed

        self.conveyor_models = conveyor_models
        self.oracle = oracle

        stop_delay = self.conveyor_cfg['stop_delay']
        try:
            routers_cfg[router_type]['conv_stop_delay'] = stop_delay
        except KeyError:
            routers_cfg[router_type] = {'conv_stop_delay': stop_delay}

        self.RouterClass = get_router_class(router_type, 'conveyors', oracle=oracle)

        if not self.centralized():
            r_topology, _, _ = conv_to_router(topology)
            self.sub_factory = RouterFactory(
                router_type, routers_cfg,
                conn_graph=r_topology.to_undirected(),
                topology_graph=r_topology,
                context='conveyors', **kwargs)

        super().__init__(conn_graph=conn_graph, **kwargs)

        self.conveyor_dyn_envs = {}
        time_func = lambda: self.env.now
        energy_func = lambda: self.energy_consumption

        for conv_id in self.conveyor_models.keys():
            dyn_env = self.dynEnv()
            dyn_env.register_var('prev_total_nrg', 0)
            dyn_env.register_var('total_nrg', 0)
            self.conveyor_dyn_envs[conv_id] = dyn_env

    def centralized(self):
        return issubclass(self.RouterClass, MasterHandler)

    def dynEnv(self):
        time_func = lambda: self.env.now
        energy_func = lambda: self.energy_consumption
        return DynamicEnv(time=time_func, energy_consumption=energy_func)

    def makeMasterHandler(self) -> MasterHandler:
        dyn_env = self.dynEnv()
        G = self.topology_graph
        cfg = {**self.router_cfg, **self.conveyor_cfg}
        cfg.update({
            'network':   G,
            'nodes':     sorted(list(G.nodes())),
            'edges_num': len(G.edges())
        })
        if self.oracle:
            cfg['conveyor_models'] = self.conveyor_models
        else:
            conv_lengths = {cid: model.length for (cid, model) in self.conveyor_models.items()}
            cfg['conv_lengths'] = conv_lengths
        return self.RouterClass(env=dyn_env, topology=self.topology, max_speed=self.max_speed, **cfg)

    def makeHandler(self, agent_id: AgentId, neighbours: List[AgentId], **kwargs) -> MessageHandler:
        a_type = agent_type(agent_id)
        conv_idx = conveyor_idx(self.topology, agent_id)

        if conv_idx != -1:
            dyn_env = self.conveyor_dyn_envs[conv_idx]
        else:
            # only if it's sink
            dyn_env = self.dynEnv()

        common_args = {
            'env': dyn_env,
            'id': agent_id,
            'neighbours': neighbours,
            'topology': self.topology,
            'router_factory': self.sub_factory,
            'oracle': self.oracle
        }

        if a_type == 'conveyor':
            if self.oracle:
                common_args['model'] = self.conveyor_models[conv_idx]
                common_args['all_models'] = self.conveyor_models

            return SimpleRouterConveyor(max_speed=self.max_speed,
                                        length=self.conveyor_models[conv_idx].length,
                                        **common_args,
                                        **self.conveyor_cfg)
        elif a_type == 'source':
            return RouterSource(**common_args)
        elif a_type == 'sink':
            return RouterSink(**common_args)
        elif a_type == 'diverter':
            return RouterDiverter(**common_args)
        else:
            raise Exception('Unknown agent type: ' + a_type)