import numpy as np
from copy import copy
import pykep as pk
import pandas as pd

import sys
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from api import Environment, MAX_FUEL_CONSUMPTION
from simulator import Simulator, read_space_objects
from agent import TableAgent as Agent


np.random.seed(0)

# TODO - check functions


def get_random_on_interval(space, num, replace=False):
    result = np.sort(np.random.choice(space, num, replace))
    result = result - space[0]
    result[1:] = result[1:] - result[:-1]
    return result


def get_random_time_to_req(time_space, n_actions, last_nan=True):
    if last_nan:
        n_actions -= 1
    if n_actions != 0:
        time_to_req = get_random_on_interval(time_space, n_actions)
    else:
        return(np.array([np.nan]))
    if last_nan:
        time_to_req = np.append(time_to_req, np.nan)
    return time_to_req


def generate_session(protected, debris, agent, start_time, end_time, step):
    start_time_mjd2000 = pk.epoch(start_time, "mjd2000")
    env = Environment(copy(protected), copy(debris), start_time_mjd2000)
    simulator = Simulator(agent, env, update_r_p_step=None, print_out=False)
    # reward
    reward = simulator.run(end_time, step, visualize=False)
    return reward


def get_random_actions(n_rnd_actions, max_time, max_fuel, last_nan=False, inaction=True):
    '''
    Agrs:
        max_fuel (float): max fuel consumption in this action
    '''
    dV = np.empty((n_rnd_actions, 3))

    if last_nan:
        time_to_req = np.full((n_rnd_actions, 1), np.nan)
    else:
        time_to_req = np.random.uniform(high=max_time, size=(n_rnd_actions, 1))

    if inaction:
        n_rnd_actions -= 1
        dV[-1] = np.zeros(3)

    for i in range(n_rnd_actions):
        fuel_cons = np.random.uniform(max_fuel)
        dV[i] = get_random_on_interval(np.linspace(
            0, fuel_cons), 3, True) - fuel_cons / 3

    actions = np.hstack((dV, time_to_req))

    return actions


def add_action_to_action_table(action_table, action):

    if action_table.size:
        action_table = np.vstack([action_table, action])
        action_table[-2, -1] = action_table[-1, -1]
        action_table[-1, -1] = np.nan
    else:
        action_table = action.reshape((1, -1))

    return action_table


class DecisionTree:

    def __init__(self, protected, debris, start_time, end_time, step,
                 max_fuel_cons=10, fuel_level=20, n_actions=3):

        self.env = Environment(copy(protected), copy(debris), start_time)
        self.protected = protected
        self.debris = debris
        self.start_time = start_time
        self.end_time = end_time
        self.step = step
        self.n_actions = n_actions
        self.nodes = np.array([])
        self.fuel_level = fuel_level
        self.action_table = np.empty((0, 4))
        self.total_reward = None
        self.max_fuel_cons = max_fuel_cons

    def train(self, n_iterations=10):
        max_time = self.end_time - self.start_time
        c_fuel_level = self.fuel_level
        for i in range(self.n_actions):
            print("explored action :", i)
            reward = -float("inf")
            # TODO - this should be the first one
            # TODO - do this strategy, but add MCTS to chose dV on each step
            # TODO - do MCTS strategy (not just step-by-step)
            max_fuel = min(self.max_fuel_cons, c_fuel_level)
            if i == 0:
                actions = get_random_actions(
                    n_iterations, max_time, max_fuel, last_nan=True, inaction=True)
            else:
                actions = get_random_actions(
                    n_iterations, max_time, max_fuel, last_nan=False, inaction=True)
            # TODO - get_random_actions min(остаток топлива и макс возможная
            # трата)
            # TODO - все принты в лог
            # TODO - убрать смещение
            # print("mean:", np.mean(actions, axis=0), "\n")
            for j in range(n_iterations):
                # TODO - fuel construction for whole table
                # time to req from previous action learn after learn action
                temp_action_table = add_action_to_action_table(
                    self.action_table, actions[j])
                agent = Agent(temp_action_table)
                # TODO - don't generate all session
                r = generate_session(
                    self.protected, self.debris, agent, self.start_time, self.end_time, self.step)

                print("action:", j, actions[j], "reward:", r)
                if r > reward:
                    # TODO - choose several best actions?
                    best_action = actions[j]
                    reward = r
                    c_fuel_level = self.protected.get_fuel()
                    # state and other should be saves somehow
            print('best action:', best_action, "reward:", reward)

            self.total_reward = reward
            self.action_table = add_action_to_action_table(
                self.action_table, best_action)
            if not np.isnan(best_action[-1]):
                max_time -= best_action[-1]
            print("action table:\n", self.action_table, "\n")

        # сразу выучивает, что что-то надо делать. Это не правильно.
        # или несколько действий еще вперед неслучайных проигрывать,
        # или доигрывать не до конца
        # или и то и то (можно опционально это делать)

    def get_action_table(self):
        return self.action_table

    def get_total_reward(self):
        return self.total_reward

    def save_action_table(self, path):
        header = "dVx,dVy,dVz,time to request"
        np.savetxt(path, self.action_table, delimiter=',', header=header)
