import numpy as np
from copy import copy
import pykep as pk
import pandas as pd

import sys
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from api import Environment
from simulator import Simulator
from agent import TableAgent as Agent

PI = 3.1415

np.random.seed(0)

# TODO - tests


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


def get_random_dV(fuel_cons):
    """Returns random x, y, z accelerations at a given fuel consumption.

    Args:
        fuel_cons (float): fuel consumption.

    Returns:
        dV (np.array): accelerations.

    """
    # using Spherical coordinate system
    r = fuel_cons
    theta = np.random.uniform(0, PI)
    phi = np.random.uniform(0, 2 * PI)

    dVx = r * np.sin(theta) * np.cos(phi)
    dVy = r * np.sin(theta) * np.sin(phi)
    dVz = r * np.cos(theta)

    dV = np.array([dVx, dVy, dVz])

    return dV


def get_random_actions(n_rnd_actions, max_time, max_fuel_cons, fuel_level, inaction=True,
                       p_skip=0.2, p_skip_coef=0.1):
    '''Random actions.

    Agrs:
        n_rnd_actions (int): number of random actions.
        max_time (float): maximum time to request.
        max_fuel_cons (float): maximum allowable fuel consumption per action.
        fuel_level (float): total fuel level.
        nan_time_to_req (bool): array of times to request is np.nan (for example, for last action).
        inaction (bool): first action of random actions is inaction.
        p_skip (float): probability of inaction (besides force first inaction).
        p_skip_coef (float): coefficient of inaction probability increase (from 0 to 1).

    Returns:
        actions (np.array): array of random actions.

    '''
    dV_arr = np.empty((n_rnd_actions - inaction, 3))

    for i in range(n_rnd_actions - inaction):
        if np.random.uniform() < p_skip:
            dV_arr[i] = [0, 0, 0]
        else:
            fuel_cons = np.random.uniform(
                0, min(max_fuel_cons, fuel_level))
            fuel_level -= fuel_cons
            dV_arr[i] = get_random_dV(fuel_cons)
        p_skip = 1 - (1 - p_skip) * (1 - p_skip_coef)

    if inaction:
        dV_arr = np.vstack((np.zeros((1, 3)), dV_arr))

    time_to_req = np.random.uniform(0.001, max_time, (n_rnd_actions, 1))

    actions = np.hstack((dV_arr, time_to_req))

    return actions


def add_action_to_action_table(action_table, action):
    if action_table.size:
        action_table = np.vstack([action_table, action])
    else:
        action_table = action.reshape((1, -1))

    return action_table


class DecisionTree:
    """Simple MCTS Method for Reinforcement Learning.

    """

    def __init__(self, protected, debris, start_time, end_time, step,
                 max_fuel_cons=10, fuel_level=20):
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
        self.investigated_time = self.start_time
        self.step = step
        self.fuel_level = fuel_level
        self.max_fuel_cons = max_fuel_cons
        self.action_table = np.empty((0, 4))
        self.total_reward = None
        self.max_time_to_req = 0.05

    def train(self, n_iterations=10, n_steps_ahead=2, print_out=False):
        """Training agent policy (self.action_table).

        Args:
            n_iterations (int): number of iterations for choice an action.
            n_steps_ahead (int): number of actions ahead to evaluate.
            print_out (bool): print information during the training.

        TODO:
            MCTS strategy (not just step-by-step)?
            deal with bias
            don't generate whole session all the time
            time to req from previous action learn after learn action
            choose several best actions?
            do not finish the simulation?
            good print_out or remove it
            parallel
            log
            test
            good name for method - is it MCTS
            add min time to req?
            model description!
            combine get_best_current_action and get_best_actions_if_current_passed into one function.

        """
        skipped = False
        while (self.investigated_time < self.end_time) & (self.fuel_level > 0):
            if not skipped:
                current_best_action, current_best_reward = self.get_best_current_action(
                    n_iterations, n_steps_ahead, print_out=print_out)
            skip_best_action, skip_best_reward, skip_best_next_action = self.get_best_actions_if_current_passed(
                n_iterations, n_steps_ahead, print_out)
            # it is more advantageous to maneuver.
            if current_best_reward > skip_best_reward:
                skipped = False
                best_action = current_best_action
            # it is more advantageous to skip the maneuver.
            # in this case, at the next step the current_best_action is
            # skip_best_next_action.
            else:
                skipped = True
                best_action = skip_best_action
                current_best_action, current_best_reward = skip_best_next_action, skip_best_reward
            self.action_table = add_action_to_action_table(
                self.action_table, best_action)
            self.fuel_level -= np.linalg.norm(best_action[:3])
            if print_out:
                print("inv time:", self.investigated_time)
                print("best cur r:", current_best_reward, '\n',
                      "best cur a:", current_best_action)
                print("best skip r:", skip_best_reward, '\n',
                      "best skip a:", skip_best_action, '\n',
                      "best skip next a:", skip_best_next_action)
                print("best action:", best_action, '\n',
                      "fuel level:", self.fuel_level, '\n',
                      "skipped:", skipped, '\n',
                      "total AT:\n", self.action_table, '\n\n\n')
            self.investigated_time += best_action[3]

    def get_best_current_action(self, n_iterations, n_steps_ahead, print_out):
        """Returns the best of random actions with given parameters.

        Args:
            n_iterations (int): number of iterations for choice an action.
            n_steps_ahead (int): number of actions ahead to evaluate.
            print_out (bool): print information during the training.

        Returns:
            best_action (np.array): best action action among considered.
            best_reward (float): reward of the session that contained the best action.

        """
        best_reward = -float("inf")
        for i in range(n_iterations):
            temp_action_table = get_random_actions(
                n_rnd_actions=n_steps_ahead + 1,
                max_time=self.max_time_to_req,
                max_fuel_cons=self.max_fuel_cons,
                fuel_level=self.fuel_level,
                inaction=False)
            action_table = np.vstack((self.action_table, temp_action_table))
            agent = Agent(action_table)
            r = generate_session(
                self.protected, self.debris, agent, self.start_time, self.end_time, self.step)
            if print_out:
                print("current iter:", i)
                print("AT:", action_table)
                print("r:", r, '\n')
            if r > best_reward:
                best_reward = r
                best_action = temp_action_table[0]

        return best_action, best_reward

    def get_best_actions_if_current_passed(self, n_iterations, n_steps_ahead, print_out):
        """Returns the best of random actions with given parameters, provided that firts action skipped.

        Args:
            n_iterations (int): number of iterations for choice an action.
            n_steps_ahead (int): number of actions ahead to evaluate.
            print_out (bool): print information during the training.


        Returns:
            best_action (np.array): best action action among considered (empty action just with time to request).
            best_reward (float): reward of the session that contained the best action.
            best_next_action (np.array): best next action action among considered.

        """
        best_reward = -float("inf")
        for i in range(n_iterations):
            temp_action_table = get_random_actions(
                n_rnd_actions=n_steps_ahead + 2,
                max_time=self.max_time_to_req,
                max_fuel_cons=self.max_fuel_cons,
                fuel_level=self.fuel_level,
                inaction=True)
            action_table = np.vstack((
                self.action_table, temp_action_table))
            agent = Agent(action_table)
            r = generate_session(
                self.protected, self.debris, agent, self.start_time, self.end_time, self.step)
            if print_out:
                print("skip iter:", i)
                print("AT:", action_table)
                print("r:", r, '\n')
            if r > best_reward:
                best_action = temp_action_table[0]
                best_next_action = temp_action_table[1]
                best_reward = r

        return best_action, best_reward, best_next_action

    def get_action_table(self):
        return self.action_table

    def get_total_reward(self):
        agent = Agent(self.action_table)
        self.total_reward = generate_session(
            self.protected, self.debris, agent, self.start_time, self.end_time, self.step)
        return self.total_reward

    def save_action_table(self, path):
        header = "dVx,dVy,dVz,time to request"
        np.savetxt(path, self.action_table, delimiter=',', header=header)
