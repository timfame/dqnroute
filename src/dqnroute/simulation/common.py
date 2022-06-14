import networkx as nx
import os
import yaml
import pprint

from typing import *
from simpy import Environment, Event, Interrupt
from ..event_series import EventSeries
from ..messages import *
from ..agents import *
from ..utils import *


##
# Small run utilities
#

def run_simulation(RunnerClass, return_runner=False, **kwargs):
    runner = RunnerClass(**kwargs)
    data_series = runner.run(**kwargs)
    df = data_series.getSeries(add_avg=True)

    if return_runner:
        return df, runner
    return df


def mk_job_id(router_type, seed):
    return '{}-{}'.format(router_type, seed)


def un_job_id(job_id):
    [router_type, s_seed] = job_id.split('-')
    return router_type, int(s_seed)


def add_cols(df, **cols):
    for (col, val) in cols.items():
        df.loc[:, col] = val


class DummyProgressbarQueue:
    def __init__(self, bar):
        self.bar = bar

    def put(self, val):
        _, delta = val
        if delta is not None:
            self.bar.update(delta)
