# Reinforcement Learning - Cross-Entropy method.

import numpy as np
import pykep as pk

from tqdm import trange
import time
import matplotlib.pyplot as plt

from ...api import Environment, MAX_FUEL_CONSUMPTION
from ...simulator import Simulator
from ...agent import TableAgent as Agent
from ...utils import read_space_objects

from ..train_utils import (
    generate_session, constrain_action,
    print_start_train, print_end_train
)


def random_action_table(mu_table, sigma_table, fuel_level):
    """Returns random action table using normal distributions under the given parameters.

    Args:
        mu_table (np.array): action table as table of expectation.
        sigma_table (np.array): table of sigmas.
        fuel_level (float): total fuel level.

    Returns:
        action_table (np.array): random table of actions.

    """
    n_actions = mu_table.shape[0]
    action_table = np.zeros_like(mu_table)
    max_fuel_cons = MAX_FUEL_CONSUMPTION
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

    def plot(self, rewards_batch, log):
        """Displays training progress.

        Args:
            rewards_batch (list of floats): batch of rewards.
            log ([[mean_reward, max_reward, policy_reward, threshold], ]): training log.

        TODO:
            text "mean reward = %.3f, threshold=%.3f" % (log[-1][0], log[-1][-1]) with plt.text?
            reward_range=[-1500, 10]) ?
            labels
            threshold to subplot 0
            some report on the plot (iteration, n_iterations...)
            reward_range (list): [min_reward, max_reward] for chart?
            plt.hist(rewards_batch, range=reward_range)?
            percentile?

        """
        mean_rewards = list(zip(*log))[0]
        max_rewards = list(zip(*log))[1]
        policy_rewards = list(zip(*log))[2]
        threshold = log[-1][-1]

        plt.subplot(1, 2, 1)
        plt.cla()
        plt.title('Rewards history')
        plt.plot(mean_rewards, label='Mean rewards')
        plt.plot(max_rewards, label='Max rewards')
        plt.plot(policy_rewards, label='Policy rewards')
        plt.xlabel("iteration")
        plt.ylabel("reward")
        plt.legend(loc=4, prop={'size': 10})
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.cla()
        plt.title('Histogram of sessions')
        plt.hist(rewards_batch)
        plt.vlines(threshold,  [0], [len(rewards_batch)],
                   label="threshold", color='red')
        plt.xlabel("reward")
        plt.ylabel("number of sessions")
        plt.legend(loc=2, prop={'size': 10})
        plt.grid()

        self.fig.canvas.draw()

    def save_fig(self, log):
        fig = plt.figure(figsize=[10, 6])
        mean_rewards = list(zip(*log))[0]
        max_rewards = list(zip(*log))[1]
        policy_rewards = list(zip(*log))[2]
        plt.title('Rewards history')
        plt.plot(mean_rewards, label='Mean rewards')
        plt.plot(max_rewards, label='Max rewards')
        plt.plot(policy_rewards, label='Policy rewards')
        plt.xlabel("iteration")
        plt.ylabel("reward")
        plt.legend(loc=4, prop={'size': 10})
        plt.grid()
        fig.savefig("./training/CE/CE_graphics.png")


class CrossEntropy:
    """Cross-Entropy Method for Reinforcement Learning."""

    def __init__(self, env, step, n_actions=3):
        """
        Agrs:
            env (Environment): environment with given parameteres.
            step (float): time step in simulation.
            n_actions (int): total number of actions.

        TODO:
            sigma to args.
            path to save plots.
            variable step propagation step.
            generate_session => generate_session_with_env

        """

        self.env = env
        self.step = step
        self.n_actions = n_actions

        self.protected = env.protected
        self.debris = env.debris

        self.start_time = self.env.init_params["start_time"].mjd2000
        self.end_time = self.env.init_params["end_time"].mjd2000
        duration = self.end_time - self.start_time

        self.action_table = np.zeros((self.n_actions, 4))
        self.action_table[:, 3] = duration / (n_actions + 1)

        self.action_table[-1, -1] = np.nan
        self.sigma_table = np.ones((self.n_actions, 4))
        self.sigma_table[:, 3] = 0.01
        self.sigma_table[-1, -1] = np.nan

        self.fuel_level = self.env.init_fuel
        self.policy_reward = -float("inf")

    def train(self, n_iterations=20, n_sessions=100, lr=0.7, percentile=80,
              sigma_decay=0.98, lr_decay=0.98, percentile_growth=1.005,
              print_out=False, show_progress=False):
        """Training agent policy (self.action_table).

        Args:
            n_iterations (int): number of iterations.
            n_sessions (int): number of sessions per iteration.
            n_best_actions (int): number of best actions provided by iteration for update policy.
            lr (float): learning rate for stability.
            sigma_decay (float): coefficient of changing sigma per iteration.
            lr_decay (float): coefficient of changing learning rate per iteration.
            percentile_growth (float): coefficient of changing percentile.
            print_out (bool): print information during the training.
            show_progress (bool): show training chart.

        TODO:
            stop if reward change < epsilon
            careful update
            good print_out or remove it
            lists => np.arrays
            save table sometimes during the learning?
            parallel
            log
            test

        """

        if (print_out | show_progress):
            if print_out:
                train_start_time = time.time()
                self.policy_reward = self.get_reward()
                print_start_train(self.policy_reward, self.action_table)
            if show_progress:
                if not print_out:
                    self.policy_reward = self.get_reward()
                progress = ShowProgress()
                log = [[self.policy_reward] * 4]
                progress.plot([self.policy_reward], log)

        for i in range(n_iterations):
            rewards_batch = []
            action_tables = []
            for j in trange(n_sessions):
                action_table = random_action_table(
                    self.action_table, self.sigma_table, self.fuel_level)
                agent = Agent(action_table)
                reward = generate_session(self.protected, self.debris, agent,
                                          self.start_time, self.end_time, self.step)
                action_tables.append(action_table)
                rewards_batch.append(reward)
            rewards_batch = np.array(rewards_batch)
            reward_threshold = np.percentile(rewards_batch, percentile)
            best_rewards_indices = rewards_batch >= reward_threshold
            best_rewards = rewards_batch[best_rewards_indices]
            best_action_tables = np.array(action_tables)[best_rewards_indices]
            new_action_table = np.mean(best_action_tables, axis=0)
            self.action_table = new_action_table * lr + \
                self.action_table * (1 - lr)
            self.sigma_table *= sigma_decay
            lr *= lr_decay
            temp_percentile = percentile * percentile_growth
            if temp_percentile <= 100:
                percentile = temp_percentile

            if print_out | show_progress:
                self.policy_reward = self.get_reward()
                mean_reward = np.mean(rewards_batch)
                max_reward = best_rewards[-1]
                if print_out:
                    s = (f"iter #{i}:"
                         + f"\nPolicy Reward: {self.policy_reward}"
                         + f"\nMean Reward:   {mean_reward}"
                         + f"\nMax Reward:    {max_reward}"
                         + f"\nThreshold:     {reward_threshold}")
                    print(s)
                if show_progress:
                    log.append([mean_reward, max_reward,
                                self.policy_reward, reward_threshold])
                    progress.plot(rewards_batch, log)

        if not (print_out | show_progress):
            self.policy_reward = self.get_reward()

        if print_out:
            train_time = time.time() - train_start_time
            print_end_train(self.policy_reward, train_time, self.action_table)
        if show_progress:
            progress.save_fig(log)

    def set_action_table(self, action_table):
        # TODO - try to set MCTS action_table and train (tune) it.
        # TODO - use copy
        self.action_table = action_table

    def get_reward(self):
        agent = Agent(self.action_table)
        return generate_session(self.protected, self.debris, agent,
                                self.start_time, self.end_time, self.step)

    def save_action_table(self, path):
        # TODO - save reward here?
        header = "dVx,dVy,dVz,time to request"
        np.savetxt(path, self.action_table, delimiter=',', header=header)
