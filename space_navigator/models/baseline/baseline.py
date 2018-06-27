# Baseline - selection of a prograde/retrograde maneuver.

import numpy as np
import pykep as pk
from tqdm import trange
import time

from ...api import Environment, MAX_FUEL_CONSUMPTION
from ...simulator import Simulator
from ...agent import TableAgent as Agent

from ..train_utils import (
    generate_session, print_start_train, print_end_train
)


class Baseline:
    """Prograde/Retrograde maneuver.

    TODO:
        change method name
        user defined number of orbital periods before collision (0.5; 1.5; ..)

    """

    def __init__(self, env, step):
        """
        Agrs:
            env (Environment): environment with given parameteres.

        """
        self.env = env
        self.protected = env.protected
        self.debris = env.debris

        self.start_time = self.env.init_params["start_time"].mjd2000
        self.end_time = self.env.init_params["end_time"].mjd2000
        self.step = step

        self.fuel_level = self.env.init_fuel
        self.max_fuel = min(MAX_FUEL_CONSUMPTION, self.fuel_level / 2)

        self.action_table = np.array([])
        self.policy_reward = None

    def train(self, num=100, print_out=False):
        """Training agent policy (self.action_table).

        Args:
            num (int): number of samples to generate.
            print_out (bool): print information during the training.

        TODO:
            limit "c" by end_time

        """
        if print_out:
            train_start_time = time.time()
            print_start_train(self.get_reward(), self.action_table)

        _, V = self.protected.position(self.start_time)
        c = self.max_fuel / np.linalg.norm(V)
        space = np.linspace(-c, c, num)
        dV_arr = np.vstack([V[i] * space for i in range(3)]).T
        best_reward = -float("inf")

        for i in trange(num):
            dV = dV_arr[i]
            temp_action_table = np.hstack((dV, np.nan))
            agent = Agent(temp_action_table)
            _, temp_env = generate_session(self.protected, self.debris, agent,
                                           self.start_time, self.start_time + self.step, self.step, return_env=True)
            time_to_req = temp_env.protected.get_orbital_period()
            temp_action_table = np.vstack((
                np.hstack((dV, time_to_req)),
                np.hstack((-dV, np.nan))
            ))
            agent = Agent(temp_action_table)
            r = generate_session(
                self.protected, self.debris, agent, self.start_time, self.end_time, self.step)

            if r > best_reward:
                best_reward = r
                self.action_table = temp_action_table

        if print_out:
            train_time = time.time() - train_start_time
            print_end_train(self.get_reward(), train_time, self.action_table)

    def get_reward(self):
        agent = Agent(self.action_table)
        return generate_session(
            self.protected, self.debris, agent, self.start_time, self.end_time, self.step)

    def save_action_table(self, path):
        header = "dVx,dVy,dVz,time to request"
        np.savetxt(path, self.action_table, delimiter=',', header=header)
