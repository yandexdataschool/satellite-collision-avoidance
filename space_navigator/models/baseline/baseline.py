# Baseline - selection of a prograde/retrograde maneuver.

import numpy as np
from copy import copy
import pykep as pk

from ...api import Environment
from ...simulator import Simulator
from ...agent import TableAgent as Agent

from ..train_utils import generate_session


class Baseline:
    """Prograde/Retrograde maneuver.

    TODO:
        change method name
        user defined number of orbital periods before collision (0.5; 1.5; ..)

    """

    def __init__(self, protected, debris, start_time, end_time, step,
                 max_fuel_cons, fuel_level):
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
        self.env = Environment(
            copy(protected), copy(debris), start_time, end_time)
        self.protected = protected
        self.debris = debris
        self.start_time = start_time
        self.end_time = end_time
        self.step = step
        self.max_fuel = min(max_fuel_cons, fuel_level / 2)
        self.action_table = None
        self.total_reward = None

    def train(self, num=100, print_out=False):
        """Training agent policy (self.action_table).

        Args:
            num (int): number of samples to generate.
            print_out (bool): print information during the training.

        TODO:
            limit "c" by end_time

        """
        if print_out:
            self.print_start_train()

        _, V = self.protected.position(self.start_time)
        c = self.max_fuel / np.linalg.norm(V)
        space = np.linspace(-c, c, num)
        dV_arr = np.vstack([V[i] * space for i in range(3)]).T
        best_reward = -float("inf")
        if print_out:
            print("c: {}\nV: {}\n".format(c, V))

        for dV in dV_arr:
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
                s = "dV: {}\naction table:\n{}\nr: {}\n".format(
                    dV, temp_action_table, r)
                print(s)

        if print_out:
            self.print_end_train()

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
