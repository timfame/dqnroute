import logging

from ...constants import *
from ...messages import *
from ...utils import *
from ...event_series import EventSeries, event_series, MultiEventSeries
from .common import SimulationRunner
from ..environment.common import MultiAgentEnv
from ..environment.conveyor import ConveyorsEnvironment


logger = logging.getLogger(DQNROUTE_LOGGER)


class ConveyorsRunner(SimulationRunner):
    """
    Class which constructs and runs scenarios in conveyor network
    simulation environment.
    """
    context = 'conveyors'

    def __init__(self, data_dir=LOG_DATA_DIR + '/conveyors', omit_training: bool = False, **kwargs):
        """
        :param omit_training: whether to skip simulation & training when run.
        """
        super().__init__(data_dir=data_dir, **kwargs)
        self.omit_training = omit_training

    def makeDataSeries(self, series_period, series_funcs):
        time_series = event_series(series_period, series_funcs)
        energy_series = event_series(series_period, series_funcs)
        collisions_series = event_series(series_period, series_funcs)
        return MultiEventSeries(time=time_series, energy=energy_series, collisions=collisions_series)

    def makeMultiAgentEnv(self, **kwargs) -> MultiAgentEnv:
        run_settings = self.run_params['settings']
        return ConveyorsEnvironment(env=self.env, data_series=self.data_series,
                                    conveyors_layout=self.run_params['configuration'],
                                    routers_cfg=run_settings['router'],
                                    conveyor_cfg=run_settings['conveyor'],
                                    **run_settings['conveyor_env'], **kwargs)

    def relevantConfig(self):
        ps = self.run_params
        ss = ps['settings']
        return (ps['configuration'], ss['bags_distr'], ss['conveyor_env'],
                ss['conveyor'], ss['router'].get(self.world.factory.router_type, {}))

    def makeRunId(self, random_seed: int) -> str:
        return f'{self.world.factory.router_type}-{random_seed}'

    def runProcess(self, random_seed: int = None):
        if random_seed is not None:
            if self.world.factory.centralized():
                seed = random_seed + 42
            else:
                seed = random_seed
            set_random_seed(seed)

        all_nodes = list(self.world.conn_graph.nodes)
        for node in all_nodes:
            self.world.passToAgent(node, WireInMsg(-1, InitMessage({})))

        bag_distr = self.run_params['settings']['bags_distr']
        sources = list(self.run_params['configuration']['sources'].keys())
        sinks = self.run_params['configuration']['sinks']

        # Little pause in order to let all initialization messages settle
        yield self.env.timeout(1)

        # added by Igor to support loading already trained models
        if self.omit_training:
            return

        bag_id = 1

        for period in bag_distr['sequence']:
            try:
                action = period['action']
                conv_idx = period['conv_idx']
                pause = period.get('pause', 0)

                if pause > 0:
                    yield self.env.timeout(pause)

                if action == 'conv_break':
                    yield self.world.handleWorldEvent(ConveyorBreakEvent(conv_idx))
                elif action == 'conv_restore':
                    yield self.world.handleWorldEvent(ConveyorRestoreEvent(conv_idx))
                else:
                    raise Exception('Unknown action: ' + action)

                if pause > 0:
                    yield self.env.timeout(pause)

            except KeyError:
                # adding a tiny noise to delta
                delta = period['delta'] + round(np.random.normal(0, 0.5), 2)

                cur_sources = period.get('sources', sources)
                cur_sinks = period.get('sinks', sinks)
                simult_sources = period.get("simult_sources", 1)
                # print(period, cur_sources)
                # assert False

                for i in range(0, period['bags_number'] // simult_sources):
                    srcs = random.sample(cur_sources, simult_sources)
                    for (j, src) in enumerate(srcs):
                        if j > 0:
                            mini_delta = round(abs(np.random.normal(0, 0.5)), 2)
                            yield self.env.timeout(mini_delta)

                        dst = random.choice(cur_sinks)

                        # fix to make reinforce work
                        from dqnroute.agents.routers.reinforce import PackageHistory
                        PackageHistory.started_packages.add(bag_id)

                        bag = Bag(bag_id, ('sink', dst), self.env.now, None)
                        logger.debug(f"Sending random bag #{bag_id} from {src} to {dst} at time {self.env.now}")
                        yield self.world.handleWorldEvent(BagAppearanceEvent(src, bag))

                        bag_id += 1
                    yield self.env.timeout(delta)
