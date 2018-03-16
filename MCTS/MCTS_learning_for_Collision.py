import pandas as pd
import numpy as np
import pykep as pk
import tqdm

import sys
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from simulator import Simulator, read_space_objects
from api import Environment, MAX_FUEL_CONSUMPTION
from MCTS_simple import DecisionTree

from copy import copy

start_time = 6599.95
step = 0.0001
end_time = 6600.05

n_actions = 3
n_iterations = 10
time_space = np.arange(start_time, end_time, step)

osc = read_space_objects(parent_dir + "/data/collision.osc", "osc")
protected = osc[0]
debris = [osc[1]]

start_time_mjd2000 = pk.epoch(start_time, "mjd2000")
max_fuel_cons = MAX_FUEL_CONSUMPTION
fuel_level = protected.get_fuel()

action_table = DecisionTree(
    protected, debris, start_time, end_time, step, n_actions=n_actions)
action_table.train(n_iterations, print_out=True)
# TODO - maneuvers to print_out with the time
print(action_table.get_action_table())
print(action_table.get_total_reward())
action_table.save_action_table('action_table.csv')
