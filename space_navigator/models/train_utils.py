import os

import numpy as np
import pykep as pk
from copy import copy

import matplotlib.pyplot as plt

from ..api import Environment
from ..api import fuel_consumption
from ..simulator import Simulator
from ..agent import TableAgent


def generate_session_with_env(agent, env, step):
    """ Play full simulation. 
    Args:
        agent (Agent): agent to do actions.
        env (Environment): environment to simulate session with.

    Returns:
        reward (float): reward after end of simulation.
    """
    simulator = Simulator(agent, env, step)
    reward = simulator.run(log=False)
    env.reset()
    return reward


def get_orbital_period_in_the_end_of_session(agent, env, step):
    # TODO - better getting of orbital_period
    simulator = Simulator(agent, env, step)
    simulator.run(log=False)
    period = env.protected.get_orbital_period()
    env.reset()
    return period


# TODO - delete generate_session
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


def print_start_train(reward, action_table):
    #TODO - remove
    print("Start training.\n\nInitial action table:\n", action_table,
          "\nInitial Reward:", reward, "\n")


def print_end_train(reward, train_time, action_table):
    #TODO - remove
    print("\nTraining completed in {:.5} sec.".format(train_time))
    print(f"Total Reward: {reward}")
    print(f"Action Table:\n{action_table}")


def time_to_first_collision_in_env(env, step):
    # TODO - test
    agent = TableAgent()
    simulator = Simulator(agent, env, step)
    reward = simulator.run(log=False)
    collision_moments = env.get_collision_data()
    if collision_moments:
        time = collision_moments[0]['epoch'] - env.get_start_time()
    else:
        # TODO - if None - empty action table for all models
        time = None
    env.reset()
    return time


def time_to_early_first_maneuver(env, step):
    # TODO - test
    time_to_collision = time_to_first_collision_in_env(env, step)
    if time_to_collision:
        orbital_period = env.protected.get_orbital_period()
        print("ot:", orbital_period)
        print("tc:", time_to_collision)
        if time_to_collision < orbital_period / 2:
            time = 0
        else:
            time = (time_to_collision - orbital_period / 2) % orbital_period
    else:
        time = None
    print("time:", time)
    return time
