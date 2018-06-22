import os

import numpy as np
import pykep as pk
from copy import copy

import matplotlib.pyplot as plt

from ..api import Environment
from ..api import fuel_consumption
from ..simulator import Simulator


def generate_session_with_env(agent, env):
    """ Play full simulation. 
    Args:
        agent (Agent): agent to do actions.
        env (Environment): environment to simulate session with.

    Returns:
        reward (float): reward after end of simulation.
    """
    simulator = Simulator(agent, env)
    reward = simulator.run(log=False)
    env.reset()
    return reward


def generate_session(protected, debris, agent, start_time, end_time, step, return_env=False):
    """Simulation.

    Args:
        protected (SpaceObject): protected space object in Environment.
        debris ([SpaceObject, ]): list of other space objects.
        agent (Agent): agent, to do actions in environment.
        start_time (float): start time of simulation provided as mjd2000.
        end_time (float): end time of simulation provided as mjd2000.
        step (float): time step in simulation.
        return_env (bool): return the environment at the end of the session.

    Returns:
        reward: reward of the session.

    """
    start_time_mjd2000 = pk.epoch(start_time, "mjd2000")
    end_time_mjd2000 = pk.epoch(end_time, "mjd2000")
    protected_copy, debris_copy = copy(protected), copy(debris)
    env = Environment(protected_copy, debris_copy,
                      start_time_mjd2000, end_time_mjd2000)
    simulator = Simulator(agent, env, step)
    reward = simulator.run(log=False)
    if return_env:
        return reward, env
    return reward


def constrain_action(action, max_fuel_cons, min_time=None, max_time=None):
    """Changes the action in accordance with the restrictions.

    Args:
        action (np.array): action.
        max_fuel_cons (float): maximum allowable fuel consumption.

    Returns:
        action (np.array): changed action.

    TODO:
        time constrain (max and min time to request)

    """
    fuel_cons = fuel_consumption(action[:3])
    if fuel_cons > max_fuel_cons:
        action[:3] *= max_fuel_cons / fuel_cons
    if min_time is not None and max_time is not None:
        action[3] = max(min_time, min(max_time, action[3]))
    else:
        action[3] = max(0.001, action[3])
    return action


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
