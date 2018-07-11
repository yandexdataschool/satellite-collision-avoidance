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

from ..base_model import BaseTableModel
from ..train_utils import (
    generate_session, constrain_action,
    print_start_train, print_end_train
)


def random_action_table(mu_table, sigma_table, fuel_level, dV_angle):
    """Returns random action table using normal distributions under the given parameters.

    Args:
        mu_table (np.array): action table as table of expectation.
        sigma_table (np.array): table of sigmas.
        fuel_level (float): total fuel level.

    Returns:
        action_table (np.array): random table of actions.

    """
    n_maneuvers = mu_table.shape[0]
    action_table = np.zeros_like(mu_table)
    max_fuel_cons = MAX_FUEL_CONSUMPTION
    for i in range(n_maneuvers - 1):
        for j in range(4):
            if dV_angle == "auto":
                action_table[i, j] = np.random.normal(
                    mu_table[i, j], sigma_table[i, j])
            if dV_angle == "collinear":
                # how to do collinear for n_maneuvers > 0
                pass
            if dV_angle == "coplanar":

                pass
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

    def plot(self, rewards_batch, log_rewards):
        """Displays training progress.

        Args:
            rewards_batch (list of floats): batch of rewards.
            log_rewards ([[mean_reward, max_reward, policy_reward, threshold], ]): training log_rewards.

        TODO:
            text "mean reward = %.3f, threshold=%.3f" % (log_rewards[-1][0], log_rewards[-1][-1]) with plt.text?
            reward_range=[-1500, 10]) ?
            labels
            threshold to subplot 0
            some report on the plot (iteration, n_iterations...)
            reward_range (list): [min_reward, max_reward] for chart?
            plt.hist(rewards_batch, range=reward_range)?
            percentile?

        """
        mean_rewards = list(zip(*log_rewards))[0]
        max_rewards = list(zip(*log_rewards))[1]
        policy_rewards = list(zip(*log_rewards))[2]
        threshold = log_rewards[-1][-1]

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

    def save_fig(self, log_rewards):
        fig = plt.figure(figsize=[10, 6])
        mean_rewards = list(zip(*log_rewards))[0]
        max_rewards = list(zip(*log_rewards))[1]
        policy_rewards = list(zip(*log_rewards))[2]
        plt.title('Rewards history')
        plt.plot(mean_rewards, label='Mean rewards')
        plt.plot(max_rewards, label='Max rewards')
        plt.plot(policy_rewards, label='Policy rewards')
        plt.xlabel("iteration")
        plt.ylabel("reward")
        plt.legend(loc=4, prop={'size': 10})
        plt.grid()
        fig.savefig("./training/CE/CE_graphics.png")


class CrossEntropy(BaseTableModel):
    """Cross-Entropy Method for Reinforcement Learning."""

    def __init__(self, env, step, reverse=True, first_maneuver_time="early",
                 n_maneuvers=2, lr=0.7, percentile=80,
                 sigma_dV=None, sigma_t=None):
        """
        Agrs:
            env (Environment): environment with given parameteres.
            step (float): time step in simulation.
            n_maneuvers (int): total number of maneuvers.

        TODO:
            sigma to args.
            path to save plots.
            variable step propagation step.
            generate_session => generate_session_with_env

        """
        super().__init__(env, step, reverse, first_maneuver_time)

        if n_maneuvers < 1:
            raise ValueError(
                f"n_maneuvers = {n_maneuvers}, must be greater than 0.")
        if reverse and n_maneuvers != 2:
            raise ValueError(
                f"if reverse==True, n_maneuvers = {n_maneuvers} must be equal to 2.")

        self.start_time = env.init_params["start_time"].mjd2000
        self.end_time = env.init_params["end_time"].mjd2000
        duration = self.end_time - self.start_time
        # for first action: dV = 0, time_to_req >= 0.
        n_actions = n_maneuvers + 1

        # action table
        self.action_table = np.zeros((n_actions, 4))
        self.action_table[:, 3] = duration / (n_actions)
        self.action_table[-1, -1] = np.nan

        # sigma table
        sigma_dV = sigma_dV or 1
        sigma_t = sigma_t or duration / 5
        self.sigma_table = np.vstack((
            np.zeros((1, 4)), sigma_dV * np.ones((n_maneuvers, 4))
        ))
        self.sigma_table[:, 3] = sigma_t
        self.sigma_table[-1, -1] = np.nan

        if first_maneuver_time == "early":
            self.action_table[0] = np.array(
                [0, 0, 0, self.time_to_first_maneuver])
            self.action_table[1:-1, 3] = (
                duration - self.time_to_first_maneuver) / (n_actions)
            self.sigma_table[0, 3] = 0

        self.lr = lr
        self.percentile = percentile

        self.fuel_level = env.init_fuel
        self.policy_reward = -float("inf")

        self.progress = None

    def iteration(self, print_out=False, n_sessions=30,
                  sigma_decay=0.98, lr_decay=0.98, percentile_growth=1.005,
                  show_progress=False,
                  dV_angle="auto"):
        # TODO - take into account coplanar and collinear
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
            dV_angle (str): "coplanar", "collinear" or "auto".

        TODO:
            stop if reward change < epsilon
            careful update
            good print_out or remove it
            lists => np.arrays
            save table sometimes during the learning?
            parallel
            log
            test
            обратный маневр - пока ровно через виток,
                но потом можно перебирать разное время
                или ближайший виток после последней опасности/самый поздний до конца сессии  

        """
        # progress
        if show_progress:
            if not self.progress:
                self.progress = ShowProgress()
            if not print_out:
                self.policy_reward = self.get_reward(self.action_table)
            log_rewards = [[self.policy_reward] * 4]
            self.progress.plot([self.policy_reward], log_rewards)

        # iteration
        rewards_batch = []
        action_tables = []
        for _ in trange(n_sessions):
            if self.reverse:
                temp_action_table = random_action_table(
                    self.action_table[:2, :], self.sigma_table[:2, :],
                    self.fuel_level / 2, dV_angle
                )
                # TODO - better getting of orbital_period
                # TODO - first action = 0, but time2teq!=0 and trainable
                # sigma = 0
                agent = Agent(temp_action_table)
                _, temp_env = generate_session(self.protected, self.debris, agent,
                                               self.start_time, self.start_time + self.step, self.step, return_env=True)
                time_to_req = temp_env.protected.get_orbital_period()
                action_table = np.vstack(
                    (temp_action_table, -temp_action_table))
                action_table[0, 3] = time_to_req

            else:
                action_table = random_action_table(
                    self.action_table, self.sigma_table,
                    self.fuel_level, dV_angle
                )

            reward = self.get_reward(action_table)

            action_tables.append(action_table)
            rewards_batch.append(reward)

        rewards_batch = np.array(rewards_batch)
        reward_threshold = np.percentile(rewards_batch, self.percentile)
        best_rewards_indices = rewards_batch >= reward_threshold
        best_rewards = rewards_batch[best_rewards_indices]
        best_action_tables = np.array(action_tables)[best_rewards_indices]

        if self.reverse:
            new_action_table = None
            pass
        else:
            new_action_table = np.mean(best_action_tables, axis=0)
            self.action_table = new_action_table * self.lr + \
                self.action_table * (1 - self.lr)

        self.sigma_table *= sigma_decay
        self.lr *= lr_decay
        temp_percentile = self.percentile * percentile_growth
        if temp_percentile <= 100:
            self.percentile = temp_percentile

        if print_out | show_progress:
            self.policy_reward = self.get_reward(self.action_table)
            mean_reward = np.mean(rewards_batch)
            max_reward = best_rewards[-1]
            if print_out:
                s = (  # f"iter #{i}:"
                    f"Policy Reward: {self.policy_reward}"
                     + f"\nMean Reward:   {mean_reward}"
                     + f"\nMax Reward:    {max_reward}"
                     + f"\nThreshold:     {reward_threshold}")
                print(s)
                print(self.action_table)
            if show_progress:
                log_rewards.append([mean_reward, max_reward,
                                    self.policy_reward, reward_threshold])
                self.progress.plot(rewards_batch, log_rewards)
                # TODO - remove?
                self.progress.save_fig(log_rewards)

    def set_action_table(self, action_table):
        # TODO - try to set MCTS action_table and train (tune) it.
        # TODO - use copy        if reverse == True and n_maneuvers != 2:
        # TODO - manage with reverse
        if self.reverse:
            raise ValueError("reverse==True")
        self.action_table = action_table
