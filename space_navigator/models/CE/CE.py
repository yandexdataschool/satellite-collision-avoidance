# Reinforcement Learning - Cross-Entropy method.

import numpy as np
import pykep as pk

from tqdm import trange
import time
import matplotlib.pyplot as plt

from ...api import Environment, MAX_FUEL_CONSUMPTION, fuel_consumption
from ...simulator import Simulator
from ...agent import TableAgent as Agent
from ...utils import read_space_objects

from ..base_model import BaseTableModel
from ..train_utils import (
    orbital_period_after_actions, position_after_actions,
    constrain_action, projection,
)


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
                duration - self.time_to_first_maneuver) / n_actions
            self.sigma_table[0, 3] = 0

        self.lr = lr
        self.percentile = percentile

        self.fuel_level = env.init_fuel
        self.policy_reward = -float("inf")

        self.progress = None
        self.log_rewards = None

    def iteration(self, print_out=False, n_sessions=30,
                  sigma_decay=0.98, lr_decay=0.98, percentile_growth=1.005,
                  show_progress=False,
                  dV_angle="auto"):
        # TODO - take into account coplanar and collinear
        # TODO - не двигаться если reward хуже (опционально)
        # dV_angle="collinear"
        """Training agent policy (self.action_table).

        Args:
            n_sessions (int): number of sessions per iteration.
            n_best_actions (int): number of best actions provided by iteration for update policy.
            lr (float): learning rate for stability.
            sigma_decay (float): coefficient of changing sigma per iteration.
            lr_decay (float): coefficient of changing learning rate per iteration.
            percentile_growth (float): coefficient of changing percentile.
            print_out (bool): print iteration information.
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
            # Добавить ранюю остановку? в методах RL не двигаться если нет
            # улучшений?


        """
        # progress
        if show_progress:
            if not self.progress:
                self.progress = ShowProgress()
                if not print_out:
                    self.policy_reward = self.get_reward(self.action_table)
                self.log_rewards = [[self.policy_reward] * 4]
                self.progress.plot([self.policy_reward], self.log_rewards)

        # iteration
        rewards_batch = []
        action_tables = []
        for _ in trange(n_sessions):
            action_table = self._get_random_action_table(dV_angle)
            reward = self.get_reward(action_table)
            action_tables.append(action_table)
            rewards_batch.append(reward)

        rewards_batch = np.array(rewards_batch)
        reward_threshold = np.percentile(rewards_batch, self.percentile)
        best_rewards_indices = rewards_batch >= reward_threshold
        best_rewards = rewards_batch[best_rewards_indices]
        best_action_tables = np.array(action_tables)[best_rewards_indices]

        new_action_table = np.mean(best_action_tables, axis=0)
        self.action_table = new_action_table * self.lr + \
            self.action_table * (1 - self.lr)
        if self.reverse:
            time_to_reverse = orbital_period_after_actions(
                self.action_table[:2], self.env, self.step)
            self.action_table[1, 3] = time_to_reverse

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
                print(f"Policy Reward: {self.policy_reward}"
                      + f"\nMean Reward:   {mean_reward}"
                      + f"\nMax Reward:    {max_reward}"
                      + f"\nThreshold:     {reward_threshold}"
                      # + f"\nAction Table:\n{self.action_table}"
                      )
            if show_progress:
                self.log_rewards.append([mean_reward, max_reward,
                                         self.policy_reward, reward_threshold])
                self.progress.plot(rewards_batch, self.log_rewards)
                # TODO - remove?
                # self.progress.save_fig(log_rewards)

    def set_action_table(self, action_table):
        # TODO - try to set MCTS action_table and train (tune) it.
        # TODO - use copy        if reverse == True and n_maneuvers != 2:
        # TODO - manage with reverse
        # TODO - more Exceptions
        if np.count_nonzero(action_table[0, :3]) != 0:
            raise ValueError("first action must be empty")
        if self.reverse:
            if action_table.shape[0] != 3:
                raise ValueError("if reverse -  it has to be only 3 actions")
        self.action_table = action_table

    def _get_random_action_table(self, dV_angle):
        """Returns random action table using normal distributions under the given parameters.

        Args:
            mu_table (np.array): action table as table of expectation.
            sigma_table (np.array): table of sigmas.
            fuel_level (float): total fuel level.

        Returns:
            action_table (np.array): random table of actions.

        """
        rnd_action_table = np.zeros_like(self.action_table)
        max_fuel = MAX_FUEL_CONSUMPTION
        fuel_level = self.fuel_level / (1 + (self.reverse == True))
        for i in range(self.action_table.shape[0] - (self.reverse == True)):
            rnd_action_table[i] = np.random.normal(
                self.action_table[i], self.sigma_table[i])
            if dV_angle in ["complanar", "collinear"] and i != 0:
                dV = rnd_action_table[i, :3]
                action_epoch = pk.epoch(
                    self.env.init_params[
                        "start_time"].mjd2000 + np.sum(rnd_action_table[:i, 3]),
                    "mjd2000",
                )
                pos, V = position_after_actions(
                    rnd_action_table[:i], self.env, self.step, action_epoch)
                pos, V = np.array(pos), np.array(V)
                if dV_angle == "complanar":
                    # у нас уже есть какой-то семплированный из распраделения маневр
                    # и хочется только спроецировать его на плоскость
                    A = np.vstack((pos, V)).T
                    dV = projection(A, dV)

                if dV_angle == "collinear":
                    norm_V = np.linalg.norm(V)
                    norm_dV = np.linalg.norm(dV)
                    cos_a = np.dot(V, dV) / (norm_V * norm_dV)
                    dV = V * np.sign(cos_a) * norm_dV / norm_V
                rnd_action_table[i, :3] = dV
            elif dV_angle != "auto" and i != 0:
                raise ValueError(f"unknown dV_angle type: {dV_angle}")

            rnd_action_table[i] = constrain_action(
                rnd_action_table[i], max_fuel)

            fuel_level -= fuel_consumption(rnd_action_table[i, :3])
            max_fuel = min(max_fuel, fuel_level)

        if self.reverse:
            time_to_reverse = orbital_period_after_actions(
                rnd_action_table[:-1], self.env, self.step)
            rnd_action_table[-2, -1] = time_to_reverse
            rnd_action_table[-1, :3] = -rnd_action_table[-2, :3]

        rnd_action_table[-1, -1] = np.nan
        return rnd_action_table
