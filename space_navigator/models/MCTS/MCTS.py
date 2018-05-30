# Reinforcement Learning - based on Monte-Carlo Tree Search methods.

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
    """MCTS based method for Reinforcement Learning."""

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
            max_time_to_req (float): maximum time for requesting the next maneuver.

        TODO:
            get_best_actions_if_current_passed_with_return using get_best_current_action_with_return.
            generate_session_with_env.
            log, parallel, tests.
            model description: readme and notebook tutorials.
            better print out.

        """
        self.env = Environment(copy(protected), copy(
            debris), start_time, end_time)
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

    def train(self, n_iterations=10, n_steps_ahead=0, n_eval=0, print_out=False):
        """Training agent policy (self.action_table).

        Args:
            n_iterations (int): number of iterations for choice an action.
            n_steps_ahead (int): number of actions ahead to evaluate.
            n_eval (int): number of sessions with random policies for evaluating the action.
            print_out (bool): print information during the training.

        TODO:
            if n_steps_ahead > 0:
                evaluate action by more than one sessions.
                do some tree with exploration like MCTS?
            don't generate whole session all the time.
            time to req from previous action learn after learn action?
            do not finish the simulation?
            if n_steps_ahead=0 skip action probability => 0
        """
        if print_out:
            self.print_start_train()
        while (self.investigated_time < self.end_time) & (self.fuel_level > 0):
            best_action, best_reward = self.get_best_action(
                n_iterations, n_steps_ahead, n_eval, print_out)
            if self.action_table.size:
                self.action_table = np.vstack([self.action_table, best_action])
            else:
                self.action_table = best_action.reshape((1, -1))
            self.fuel_level -= np.linalg.norm(best_action[:3])
            if print_out:
                print("inv time:", self.investigated_time)
                print("best r:", best_reward, '\n',
                      "best a:", best_action)
                print("fuel level:", self.fuel_level, '\n',
                      "total AT:\n", self.action_table, '\n\n\n')
            self.investigated_time += best_action[3]
        if print_out:
            self.print_end_train()

    def train_with_reverse(self, n_iterations=10, print_out=False):
        """Training agent policy (self.action_table).

        Args:
            n_iterations (int): number of iterations for choice an action.
            print_out (bool): print information during the training.

        TODO:
            the number of turns around the orbit?
            see self.train().
            do this algorithm like self.train?
        """
        if print_out:
            self.print_start_train()
        skipped = False
        while (self.investigated_time < self.end_time) & (self.fuel_level > 0):
            if not skipped:
                current_best_action, current_best_reward = self.get_best_current_action_with_return(
                    n_iterations, print_out=print_out)
            skip_best_action, skip_best_reward, skip_best_next_action = self.get_best_actions_if_current_passed_with_return(
                n_iterations, print_out=print_out)
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
                self.action_table = best_action.reshape((-1, 4))

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
            self.investigated_time += np.sum(best_action[:, 3])

        if print_out:
            self.print_end_train()

    def get_best_action(self, n_iterations, n_steps_ahead, n_eval, print_out):
        """Returns the best of random actions with given parameters.

        Args:
            n_iterations (int): number of iterations for choice an action.
            n_steps_ahead (int): number of actions ahead to evaluate.
            n_eval (int): number of sessions with random policies for evaluating the action.
            print_out (bool): print information during the training.

        Returns:
            best_action (np.array): best action action among considered.
            best_reward (float): reward of the session that contained the best action.

        """
        best_reward = -float("inf")
        actions = get_random_actions(
            n_rnd_actions=n_iterations,
            max_time=self.max_time_to_req,
            max_fuel_cons=self.max_fuel_cons,
            fuel_level=None,  # TODO - test
            inaction=True,
            p_skip=0.1,  # ~percent of empty actions
            p_skip_coef=0)
        if print_out:
            print("actions candidates:\n", actions, "\n")
        for i in range(n_iterations):
            a = actions[i]
            if print_out:
                print("-----action #", i, ":", a)
            a_rewards = list()
            for j in range(n_eval):
                eval_action_table = get_random_actions(
                    n_rnd_actions=n_steps_ahead,
                    max_time=self.max_time_to_req,
                    max_fuel_cons=self.max_fuel_cons,
                    fuel_level=self.fuel_level,
                    inaction=False)
                action_table = np.vstack(
                    (self.action_table, a, eval_action_table))
                agent = Agent(action_table)
                r = generate_session(
                    self.protected, self.debris, agent, self.start_time, self.end_time, self.step)
                a_rewards.append(r)
                if print_out:
                    print("action table:\n", action_table)
                    print("reward:", r, '\n')
            # TODO - try average reward
            best_a_reward = np.max(a_rewards)
            if best_a_reward > best_reward:
                best_reward = best_a_reward
                best_action = a
            if print_out:
                print("best action reward:", best_a_reward, '\n')

        return best_action, best_reward

    def get_best_current_action_with_return(self, n_iterations, print_out):
        """Returns the best of random actions with given parameters including reverse action.

        Args:
            n_iterations (int): number of iterations for choice an action.
            print_out (bool): print information during the training.

        Returns:
            best_actions (np.array with shape=(2, 4)): best actions (maneuver with reverse) among considered.
            best_reward (float): reward of the session that contained the best action.

        TODO:
            better getting of period.

        """
        best_reward = -float("inf")
        for i in range(n_iterations):
            # random maneuver
            temp_action_table = get_random_actions(
                n_rnd_actions=1,
                max_time=self.max_time_to_req,
                max_fuel_cons=self.max_fuel_cons,
                fuel_level=self.fuel_level / 2,
                inaction=False,
                p_skip=0,
                p_skip_coef=0)
            # add reverse maneuver
            action_table = np.vstack(
                (self.action_table, temp_action_table))
            agent = Agent(action_table)
            _, temp_env = generate_session(self.protected, self.debris, agent,
                                           self.start_time, self.end_time, self.step, return_env=True)
            temp_action_table[0, 3] = temp_env.protected.get_orbital_period()
            temp_action_table = np.vstack(
                (temp_action_table, -temp_action_table))
            temp_action_table[1, 3] = np.nan
            action_table = np.vstack(
                (self.action_table, temp_action_table))
            agent = Agent(action_table)
            r = generate_session(
                self.protected, self.debris, agent, self.start_time, self.end_time, self.step)
            if print_out:
                print("current iter:", i)
                print("AT:", action_table)
                print("r:", r, '\n')
            if r > best_reward:
                best_reward = r
                best_actions = temp_action_table

        return best_actions, best_reward

    def get_best_actions_if_current_passed_with_return(self, n_iterations, print_out):
        """Returns the best of random actions with given parameters, provided that firts action skipped.

        Args:
            n_iterations (int): number of iterations for choice an action.
            print_out (bool): print information during the training.

        Returns:
            best_action (np.array): best action action among considered (empty action just with time to request).
            best_reward (float): reward of the session that contained the best action.
            best_next_actions (np.array with shape=(2, 4)): best next actions (maneuver with reverse) among considered.


        TODO:
            compair reward with get_best_current_action_with_return.
            better getting of period.

        """
        best_reward = -float("inf")
        for i in range(n_iterations):
            # random maneuver
            temp_action_table = get_random_actions(
                n_rnd_actions=2,
                max_time=self.max_time_to_req,
                max_fuel_cons=self.max_fuel_cons,
                fuel_level=self.fuel_level / 2,
                inaction=True,
                p_skip=0,
                p_skip_coef=0)
            # add reverse maneuver
            action_table = np.vstack(
                (self.action_table, temp_action_table))
            agent = Agent(action_table)
            _, temp_env = generate_session(self.protected, self.debris, agent,
                                           self.start_time, self.end_time, self.step, return_env=True)
            temp_action_table[1, 3] = temp_env.protected.get_orbital_period()
            temp_action_table = np.vstack(
                (temp_action_table, -temp_action_table[1]))
            temp_action_table[2, 3] = np.nan
            action_table = np.vstack(
                (self.action_table, temp_action_table))
            agent = Agent(action_table)
            r = generate_session(
                self.protected, self.debris, agent, self.start_time, self.end_time, self.step)
            if print_out:
                print("skip iter:", i)
                print("AT:", action_table)
                print("r:", r, '\n')
            if r > best_reward:
                best_action = temp_action_table[0]
                best_next_actions = temp_action_table[1:]
                best_reward = r

        return best_action, best_reward, best_next_actions

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
