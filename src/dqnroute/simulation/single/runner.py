import random
import yaml
from typing import Union, Optional, List

import networkx as nx

from .environment import Environment
from ...event_series import EventSeries, MultiEventSeries, event_series
from ...utils import dict_merge, make_network_graph, gen_network_graph, set_random_seed


class Runner:

    def __init__(self, run_params: Union[dict, str], data_dir: str, params_override={},
                 data_series: Optional[EventSeries] = None, series_period: int = 500,
                 series_funcs: List[str] = ['count', 'sum', 'min', 'max'], **kwargs):
        if type(run_params) == str:
            with open(run_params) as f:
                run_params = yaml.safe_load(f)
        run_params = dict_merge(run_params, params_override)

        if data_series is None:
            data_series = self.makeDataSeries(series_period, series_funcs)

        self.run_params = run_params
        self.data_series = data_series
        self.data_dir = data_dir

        self.env = Environment(data_series=self.data_series,
                               network_cfg=self.run_params['network'],
                               routers_cfg=self.run_params['settings']['router'],
                               **self.run_params['settings']['router_env'], **kwargs)

    def makeDataSeries(self, series_period, series_funcs):
        return MultiEventSeries(time=event_series(series_period, series_funcs))

    def run(self, random_seed=None):
        if random_seed is not None:
            set_random_seed(random_seed)

        pkg_distr = self.run_params['settings']['pkg_distr']
        pkg_id = 1
        for period in pkg_distr["sequence"]:
            topology = self.env.topology

            delta = period["delta"]
            try:
                sources = [('router', v) for v in period["sources"]]
            except KeyError:
                sources = list(topology.nodes)

            try:
                dests = [('router', v) for v in period["dests"]]
            except KeyError:
                dests = list(topology.nodes)

            simult_sources = period.get("simult_sources", 1)
            for i in range(0, period["pkg_number"] // simult_sources):
                srcs = random.sample(sources, simult_sources)
                for src in srcs:
                    dst = random.choice(dests)

                    pkg = Package(pkg_id, DEF_PKG_SIZE, dst, self.env.now, None)  # create empty packet
                    logger.debug(f"Sending random pkg #{pkg_id} from {src} to {dst} at time {self.env.now}")
                    # print(f"Runner: Sending random pkg #{pkg_id} from {src} to {dst} at time {self.env.now}")
                    yield self.world.handleWorldEvent(PkgEnqueuedEvent(('world', 0), src, pkg))
                    pkg_id += 1
                yield self.env.timeout(delta)
