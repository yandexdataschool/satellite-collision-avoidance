import os

import numpy as np
import pykep as pk
from copy import copy

import matplotlib.pyplot as plt

from ..api import Environment
from ..simulator import Simulator
from ..agent import TableAgent as Agent


def generate_session_with_env(action_table, env):
    """ Play full simulation. 
    Args:
        action_table (np.array((n_actions, action_size)): action table for agent.

    Returns:
        reward (float): reward after end of simulation.
    """
    agent = Agent(action_table)
    simulator = Simulator(agent, env)
    reward = simulator.run()
    env.reset()
    return reward


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
    end_time_mjd2000 = pk.epoch(end_time, "mjd2000")
    env = Environment(copy(protected), copy(debris),
                      start_time_mjd2000, end_time_mjd2000)
    simulator = Simulator(agent, env, update_r_p_step=None, print_out=False)
    reward = simulator.run(step, visualize=False)
    return reward


class ProgressPlotter(object):
    """ Save training results into images. """

    def __init__(self, base_dir, model):
        self.base_dir = base_dir
        self.model = model
        self.rewards = model.get_rewards_history()

    def plot_all_rewards(self, path):
        fig = plt.figure(figsize=[14, 10])
        ax = fig.add_subplot(111)
        plt.title("Rewards for all episodes")
        plt.xlabel("Session")
        plt.ylabel("Reward")
        ax.grid()
        ax.plot(self.rewards.flatten())
        fig.savefig(os.path.join(self.base_dir, path), dpi=fig.dpi)

    def plot_mean_reward_per_iteration(self, path):
        fig = plt.figure(figsize=[14, 10])
        ax = fig.add_subplot(111)
        plt.title("Mean Reward per iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        ax.grid()
        ax.plot(np.mean(self.rewards, axis=1))
        fig.savefig(os.path.join(self.base_dir, path), dpi=fig.dpi)


class ProgressLogger(object):
    # TODO: create Logger for reward/iterations/parameteres/
    pass
