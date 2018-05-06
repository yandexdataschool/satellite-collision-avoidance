# Reinforcement Learning - Monte-Carlo Tree Search method.

import numpy as np
from copy import copy
import pykep as pk

from ...api import Environment
from ...simulator import Simulator
from ...agent import TableAgent as Agent

from ..train_utils import generate_session


def get_random_dV(fuel_cons):
    """Returns random x, y, z accelerations at a given fuel consumption.

    Args:
        fuel_cons (float): fuel consumption.

    Returns:
        dV (np.array): accelerations.

    """
    # using Spherical coordinate system
    r = fuel_cons
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)

    dVx = r * np.sin(theta) * np.cos(phi)
    dVy = r * np.sin(theta) * np.sin(phi)
    dVz = r * np.cos(theta)

    dV = np.array([dVx, dVy, dVz])

    return dV


def get_random_actions(n_rnd_actions, max_time, max_fuel_cons, fuel_level=None, inaction=True,
                       p_skip=0.2, p_skip_coef=0.1):
    '''Random actions.

    Agrs:
        n_rnd_actions (int): number of random actions.
        max_time (float): maximum time to request.
        max_fuel_cons (float): maximum allowable fuel consumption per action.
        fuel_level (float): total fuel level (not taken into account if None).
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
            if fuel_level:
                fuel_cons = np.random.uniform(
                    0, min(max_fuel_cons, fuel_level))
                fuel_level -= fuel_cons
            else:
                fuel_cons = np.random.uniform(
                    0, max_fuel_cons)
            dV_arr[i] = get_random_dV(fuel_cons)
        p_skip = 1 - (1 - p_skip) * (1 - p_skip_coef)

    if inaction:
        dV_arr = np.vstack((np.zeros((1, 3)), dV_arr))

    time_to_req = np.random.uniform(0.001, max_time, (n_rnd_actions, 1))

    actions = np.hstack((dV_arr, time_to_req))

    return actions


def add_action_to_action_table(action_table, action):
    # TODO - docstrings
    if action_table.size:
        action_table = np.vstack([action_table, action])
        action_table[-2, -1] = action_table[-1, -1]
        action_table[-1, -1] = np.nan
    else:
        action_table = action.reshape((1, -1))

    return action_table


class DecisionTree:
    """MCTS Method for Reinforcement Learning."""

    def __init__(self, protected, debris, start_time, end_time, step,
                 max_fuel_cons, fuel_level, max_time_to_req=0.05):
        """
        Agrs:
            protected (SpaceObject): protected space object in Environment.
            debris ([SpaceObject, ]): list of other space objects.
            start_time (float): start time of simulation provided as mjd2000.
            end_time (float): end time of simulation provided as mjd2000.
            step (float): time step in simulation.
            max_fuel_cons (float): maximum allowable fuel consumption per action.
            fuel_level (float): total fuel level.

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
        self.max_time_to_req = max_time_to_req

    def train_simple(self, n_iterations=10, print_out=False):
        """Training agent policy (self.action_table).

        Args:
            n_iterations (int): number of iterations for choice an action.
            print_out (bool): print information during the training.

        TODO:
            describe train_simple and train difference
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
        if print_out:
            self.print_start_train()
        while (self.investigated_time < self.end_time) & (self.fuel_level > 0):
            reward = -float("inf")
            fuel = min(self.max_fuel_cons, self.fuel_level)
            actions = get_random_actions(
                n_iterations, self.max_time_to_req, self.max_fuel_cons, None, True, 0, 0)
            print("actions", actions)
            for a in actions:
                temp_action_table = add_action_to_action_table(
                    self.action_table, a)
                agent = Agent(temp_action_table)
                r = generate_session(
                    self.protected, self.debris, agent, self.start_time, self.end_time, self.step)
                if print_out:
                    print("action:", a)
                    print("r:", r, '\n')
                if r > reward:
                    best_action = a
                    reward = r
                    self.fuel_level -= np.linalg.norm(best_action[:3])

            self.action_table = add_action_to_action_table(
                self.action_table, best_action)
            if print_out:
                print("inv time:", self.investigated_time)
                print("best r:", reward, '\n',
                      "best a:", best_action, '\n',
                      "fuel level:", self.fuel_level, '\n',
                      "total AT:\n", self.action_table, '\n\n\n')
            self.investigated_time += best_action[3]

        if print_out:
            self.print_end_train()

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
        if print_out:
            self.print_start_train()
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

            if self.action_table.size:
                self.action_table = np.vstack([self.action_table, best_action])
            else:
                self.action_table = best_action.reshape((1, -1))

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
        if print_out:
            self.print_end_train()

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

    def print_start_train(self):
        print("Start training.\n")

    def print_end_train(self):
        print("Training completed.\nTotal reward:", self.get_total_reward(),
              "\nAction Table:\n", self.action_table)

    def save_action_table(self, path):
        header = "dVx,dVy,dVz,time to request"
        np.savetxt(path, self.action_table, delimiter=',', header=header)
