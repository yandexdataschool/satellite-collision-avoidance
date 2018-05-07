# Reinforcement Learning - Cross-Entropy method.

import numpy as np
from copy import copy
import pykep as pk

import matplotlib.pyplot as plt

from ...api import Environment, MAX_FUEL_CONSUMPTION
from ...simulator import Simulator
from ...agent import TableAgent as Agent
from ...utils import read_space_objects

from ..train_utils import generate_session, constrain_action


def random_action_table(mu_table, sigma_table, max_fuel_cons, fuel_level):
    """Returns random action table using normal distributions under the given parameters.

    Args:
        mu_table (np.array): action table as table of expectation.
        sigma_table (np.array): table of sigmas.
        max_fuel_cons (float): maximum allowable fuel consumption per action.
        fuel_level (float): total fuel level.

    Returns:
        action_table (np.array): random table of actions.

    """
    n_actions = mu_table.shape[0]
    action_table = np.zeros_like(mu_table)
    for i in range(n_actions - 1):
        for j in range(4):
            action_table[i, j] = np.random.normal(
                mu_table[i, j], sigma_table[i, j])
        action_table[i] = constrain_action(action_table[i], max_fuel_cons)
        fuel_level -= np.sum(action_table[i, :3])
        max_fuel_cons = min(max_fuel_cons, fuel_level)
    for j in range(3):
        action_table[-1, j] = np.random.normal(
            mu_table[-1, j], sigma_table[-1, j])
    action_table[-1] = constrain_action(action_table[-1], max_fuel_cons)
    action_table[-1, -1] = np.nan
    return action_table


class ShowProgress:
    """Displays training progress."""

    def __init__(self):
        self.fig = plt.figure(figsize=[12, 6])
        self.fig.show()
        self.fig.canvas.draw()

    def plot(self, batch_rewards, log):  # , :
        """Displays training progress.

        Args:
            batch_rewards (list of floats): batch of rewards.
            log ([[mean_reward, max_reward, policy_reward, threshold], ]): training log.

        TODO:
            text "mean reward = %.3f, threshold=%.3f" % (log[-1][0], log[-1][-1]) with plt.text?
            reward_range=[-1500, 10]) ?
            labels
            threshold to subplot 0
            some report on the plot (iteration, n_iterations...)
            reward_range (list): [min_reward, max_reward] for chart?
            plt.hist(batch_rewards, range=reward_range)? 
            percentile?

        """
        mean_rewards = list(zip(*log))[0]
        max_rewards = list(zip(*log))[1]
        policy_rewards = list(zip(*log))[2]
        threshold = log[-1][-1]

        plt.subplot(1, 2, 1)
        plt.cla()
        plt.plot(mean_rewards, label='Mean rewards')
        plt.plot(max_rewards, label='Max rewards')
        plt.plot(policy_rewards, label='Policy rewards')
        plt.legend(loc=2, prop={'size': 10})
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.cla()
        plt.hist(batch_rewards)
        plt.vlines(threshold,  [0], [len(batch_rewards)],
                   label="threshold", color='red')
        plt.legend(loc=2, prop={'size': 10})
        plt.grid()

        self.fig.canvas.draw()


class CrossEntropy:
    """Cross-Entropy Method for Reinforcement Learning."""

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

        self.env = Environment(copy(protected), copy(debris), start_time, end_time)
        self.protected = protected
        self.debris = debris
        self.start_time = start_time
        self.end_time = end_time
        self.max_time = self.end_time - self.start_time
        self.step = step
        self.n_actions = n_actions
        self.fuel_level = fuel_level
        self.action_table = np.zeros((self.n_actions, 4))
        self.action_table[:, 3] = self.max_time / n_actions
        self.action_table[-1, -1] = np.nan
        self.sigma_table = np.ones((self.n_actions, 4))
        self.sigma_table[:, 3] = 0.01
        self.sigma_table[-1, -1] = np.nan
        self.max_fuel_cons = max_fuel_cons

    def train(self, n_iterations=10, n_sessions=20, n_best_actions=4,
              learning_rate=0.7, sigma_coef=0.9, learning_rate_coef=0.9,
              print_out=False, show_progress=False):
        """Training agent policy (self.action_table).

        Args:
            n_iterations (int): number of iterations.
            n_sessions (int): number of sessions per iteration.
            n_best_actions (int): number of best actions provided by iteration for update policy.
            learning_rate (float): learning rate for stability.
            sigma_coef (float): coefficient of changing sigma per iteration.
            learning_rate_coef (float): coefficient of changing learning rate per iteration.
            print_out (bool): print information during the training.
            show_progress (bool): show training chart.

        TODO:
            percentile + perc coef
            stop if reward change < epsilon
            careful update
            good print_out or remove it
            lists => np.arrays
            save table sometimes during the learning?
            parallel
            log
            test

        """
        if print_out | show_progress:
            agent = Agent(self.action_table)
            self.total_reward = generate_session(self.protected, self.debris,
                                                 agent, self.start_time, self.end_time, self.step)
            if print_out:
                self.print_start_train()
            if show_progress:
                progress = ShowProgress()
                log = [[self.total_reward] * 4]
                progress.plot([self.total_reward], log)
        for i in range(n_iterations):
            batch_rewards = []
            action_tables = []
            for j in range(n_sessions):
                if print_out:
                    print('iter:', i + 1, "session:", j + 1)
                action_table = random_action_table(
                    self.action_table, self.sigma_table, self.max_fuel_cons, self.fuel_level)
                agent = Agent(action_table)
                action_tables.append(action_table)
                reward = generate_session(self.protected, self.debris,
                                          agent, self.start_time, self.end_time, self.step)
                batch_rewards.append(reward)
                if print_out:
                    print('action_table:\n', action_table)
                    print('reward:\n', reward)
            best_rewards_indices = np.argsort(batch_rewards)[-n_best_actions:]
            best_rewards = np.array(batch_rewards)[best_rewards_indices]
            best_action_tables = np.array(action_tables)[best_rewards_indices]
            new_action_table = np.mean(best_action_tables, axis=0)
            self.action_table = (
                new_action_table * learning_rate
                + self.action_table * (1 - learning_rate)
            )
            self.sigma_table *= sigma_coef
            learning_rate *= learning_rate
            if print_out | show_progress:
                self.update_total_reward()
                if print_out:
                    print('best rewards:', best_rewards)
                    print('new action table:', new_action_table)
                    print('action table:', self.action_table)
                    print('policy reward:', self.get_total_reward(), '\n')
                if show_progress:
                    mean_reward = np.mean(batch_rewards)
                    max_reward = best_rewards[-1]
                    policy_reward = self.get_total_reward()
                    threshold = best_rewards[0]
                    log.append([mean_reward, max_reward,
                                policy_reward, threshold])
                    progress.plot(batch_rewards, log)
        if not (print_out | show_progress):
            self.update_total_reward()
        if print_out:
            self.print_end_train()

    def set_action_table(self, action_table):
        # TODO - try to set MCTS action_table and train (tune) it.
        self.action_table = action_table

    def get_action_table(self):
        return self.action_table

    def update_total_reward(self):
        agent = Agent(self.action_table)
        self.total_reward = generate_session(self.protected, self.debris,
                                             agent, self.start_time, self.end_time, self.step)

    def get_total_reward(self):
        return self.total_reward

    def print_start_train(self):
        print("Start training.\nInitial action table:\n", self.action_table,
              "\nInitial reward:", self.total_reward, "\n")

    def print_end_train(self):
        print("Training completed.\nTotal reward:", self.total_reward,
              "\nAction Table:\n", self.action_table)

    def save_action_table(self, path):
        # TODO - save reward here?
        header = "dVx,dVy,dVz,time to request"
        np.savetxt(path, self.action_table, delimiter=',', header=header)
