import yaml

from simpy import Environment, Event, Interrupt
from ...agents import *
from ...utils import *

from ..environment.common import MultiAgentEnv


class SimulationRunner:
    """
    Class which constructs an environment from given settings and runs it.
    """

    def __init__(self, run_params: Union[dict, str], data_dir: str, params_override = {},
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

        # Makes a world simulation
        self.env = Environment()

        self.world_kwargs = kwargs.copy()
        self.world = self.makeMultiAgentEnv(**self.world_kwargs)

    def runDataPath(self, random_seed) -> str:
        cfg = self.relevantConfig()
        return f'{self.data_dir}/{data_digest(cfg)}-{self.makeRunId(random_seed)}.csv'

    def run(self, random_seed=None, ignore_saved=False,
            progress_step=None, progress_queue=None, **kwargs) -> EventSeries:
        """
        Runs the environment, optionally reporting the progress to a given queue.
        """
        data_path = self.runDataPath(random_seed)
        run_id = self.makeRunId(random_seed)

        if not ignore_saved and os.path.isfile(data_path):
            self.data_series.load(data_path)
            if progress_queue is not None:
                progress_queue.put((run_id, self.data_series.maxTime()))
                progress_queue.put((run_id, None))

        else:
            self.env.process(self.runProcess(random_seed))

            if progress_queue is not None:
                if progress_step is None:
                    self.env.run()
                    progress_queue.put((run_id, progress_step))
                else:
                    next_step = progress_step
                    while self.env.peek() != float('inf'):
                        self.env.run(until=next_step)
                        progress_queue.put((run_id, progress_step))
                        next_step += progress_step
                    progress_queue.put((run_id, None))
            else:
                self.env.run()

            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            self.data_series.save(data_path)

        return self.data_series

    def makeDataSeries(self, series_period, series_funcs):
        """
        Makes a data series if one is not given directly
        """
        raise NotImplementedError()

    def makeMultiAgentEnv(self, **kwargs) -> MultiAgentEnv:
        """
        Initializes a world environment.
        """
        raise NotImplementedError()

    def makeSingleAgentEnv(self, **kwargs) -> MultiAgentEnv:
        """
        Initializes a world environment.
        """
        raise NotImplementedError()

    def relevantConfig(self):
        """
        Defines a part of `run_params` which is used to calculate
        run hash (for data saving).
        """
        raise NotImplementedError()

    def makeRunId(self, random_seed):
        """
        Run identificator, which depends on random seed and some run params.
        """
        raise NotImplementedError()

    def runProcess(self, random_seed):
        """
        Generator which generates a series of test scenario events in
        the world environment.
        """
        raise NotImplementedError()