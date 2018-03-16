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

# TODO - check functions and do tests


def generate_session(protected, debris, agent, start_time, end_time, step):
    """Simulation.

    Args:
        protected (SpaceObject): protected space object in Environment.
        debris ([SpaceObject, ]): list of other space objects.
        agent (Agent): agent, to do actions in environment.
        start_time (float): start time of simulation provided as mjd2000.
        end_time (float): end time of simulation provided as mjd2000.
        step (float): time step in simulation.

    Returns:
        reward: reward of the session.

    """
    start_time_mjd2000 = pk.epoch(start_time, "mjd2000")
    env = Environment(copy(protected), copy(debris), start_time_mjd2000)
    simulator = Simulator(agent, env, update_r_p_step=None, print_out=False)
    reward = simulator.run(end_time, step, visualize=False)
    return reward


def get_random_on_interval(space, n, replace=False):
    """Returns n random indents from some space.

    Args:
        space (np.array): ordered array.
        n (int): number of indents.

    Returns:
        result (np.array): indents.

    """
    result = np.sort(np.random.choice(space, n, replace))
    result = result - space[0]
    result[1:] = result[1:] - result[:-1]
    return result


def get_random_time_to_req(time_space, n_actions, last_nan=True):
    """Array of random times to request.

    Args:
        time_space (np.array): ordered time array.
        n_actions (int): number of actions.
        last_nan (bool): last time to request is np.nan if True.

    Returns:
        time_to_req (np.array): array of random times to request.

    """
    if last_nan:
        n_actions -= 1
    if n_actions != 0:
        time_to_req = get_random_on_interval(time_space, n_actions)
    else:
        return(np.array([np.nan]))
    if last_nan:
        time_to_req = np.append(time_to_req, np.nan)
    return time_to_req


def get_random_actions(n_rnd_actions, max_time, max_fuel_cons, nan_time_to_req=False, inaction=True):
    '''Random actions (not action table).

    Agrs:
        n_rnd_actions (int): number of random actions.
        max_time (float): maximum time to request. 
        max_fuel_cons (float): maximum allowable fuel consumption.
        nan_time_to_req (bool): array of times to request is np.nan (for example, for last action).
        inaction (bool): one action of random actions is inaction.

    Returns:
        actions (np.array): array of random actions.

    '''
    dV = np.empty((n_rnd_actions, 3))

    if nan_time_to_req:
        time_to_req = np.full((n_rnd_actions, 1), np.nan)
    else:
        time_to_req = np.random.uniform(high=max_time, size=(n_rnd_actions, 1))

    if inaction:
        n_rnd_actions -= 1
        dV[-1] = np.zeros(3)

    for i in range(n_rnd_actions):
        fuel_cons = np.random.uniform(max_fuel_cons)
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
    """Simple MCTS Method for Reinforcement Learning.

    TODO:
        look one step ahead.
        self.train

    """

    def __init__(self, protected, debris, start_time, end_time, step,
                 max_fuel_cons=10, fuel_level=20, n_actions=3):
        """
        Agrs:
            protected (SpaceObject): protected space object in Environment.
            debris ([SpaceObject, ]): list of other space objects.
            start_time (float): start time of simulation provided as mjd2000.
            end_time (float): end time of simulation provided as mjd2000.
            step (float): time step in simulation.
            max_fuel_cons (float): maximum allowable fuel consumption per action.
            fuel_level (float): total fuel level.
            n_actions (int): total number of actions.

        """
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

    def train(self, n_iterations=10, print_out=False):
        """Training agent policy (self.action_table).

        Args:
            n_iterations (int): number of iterations for choice an action.
            print_out (bool): print information during the training.

        TODO:
            MCTS strategy (not just step-by-step)?
            deal with bias
            fuel construction for whole table
            check max_fuel_cons
            don't generate whole session all the time
            time to req from previous action learn after learn action
            choose several best actions?
            do not finish the simulation?
            good print_out or remove it
            parallel
            log
            test

        """
        max_time = self.end_time - self.start_time
        c_fuel_level = self.fuel_level
        for i in range(self.n_actions):
            reward = -float("inf")

            max_fuel = min(self.max_fuel_cons, c_fuel_level)
            if i == 0:
                actions = get_random_actions(
                    n_iterations, max_time, max_fuel, True, True)
            else:
                actions = get_random_actions(
                    n_iterations, max_time, max_fuel, False, True)
            for j in range(n_iterations):
                temp_action_table = add_action_to_action_table(
                    self.action_table, actions[j])
                agent = Agent(temp_action_table)
                r = generate_session(
                    self.protected, self.debris, agent, self.start_time, self.end_time, self.step)

                if r > reward:
                    # TODO - choose several best actions?
                    best_action = actions[j]
                    reward = r
                    c_fuel_level = self.protected.get_fuel()
                    # state and other should be saves somehow
                if print_out:
                    print('action:', i, "iteration:",
                          j, actions[j], "reward:", r)

            self.total_reward = reward
            self.action_table = add_action_to_action_table(
                self.action_table, best_action)
            if not np.isnan(best_action[-1]):
                max_time -= best_action[-1]
            if print_out:
                print("explored action :", i)
                print('best action:', best_action, "reward:", reward)
                print("action table:\n", self.action_table, "\n")

    def get_action_table(self):
        return self.action_table

    def get_total_reward(self):
        return self.total_reward

    def save_action_table(self, path):
        header = "dVx,dVy,dVz,time to request"
        np.savetxt(path, self.action_table, delimiter=',', header=header)
