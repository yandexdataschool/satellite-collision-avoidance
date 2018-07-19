# Baseline - selection of a prograde/retrograde maneuver.

import numpy as np
import pykep as pk
from tqdm import trange
import time

from ...api import Environment, MAX_FUEL_CONSUMPTION
from ...simulator import Simulator
from ...agent import TableAgent as Agent

from ..base_model import BaseTableModel
from ..train_utils import orbital_period_after_actions


class Baseline(BaseTableModel):
    """Prograde/Retrograde maneuvers."""

    def __init__(self, env, step, reverse=True):
        """
        Agrs:
            env (Environment): environment with given parameteres.
            step (float): time step in simulation.
            reverse (bool): 
                if True: there are selected exactly 2 maneuvers
                    while the second of them is reversed to the first one;
                if False: one maneuver.

        Returns:
            stop (bool): whether to stop training after iteration.

        """
        super().__init__(env, step, reverse, first_maneuver_time="early")

        self.start_time = self.env.init_params["start_time"].mjd2000
        self.end_time = self.env.init_params["end_time"].mjd2000

        self.first_action = np.array(
            [0, 0, 0, self.time_to_first_maneuver])

    def iteration(self, print_out=False, n_sessions=100):
        """Training iteration.

        Args:
            print_out (bool): print iteration information.
            n_sessions (int): number of sessions to generate.

        """
        max_fuel = self.env.init_fuel / 2 if self.reverse else self.env.init_fuel
        max_fuel = min(MAX_FUEL_CONSUMPTION, max_fuel)
        _, V = self.protected.position(self.start_time)
        c = max_fuel / np.linalg.norm(V)
        space = np.linspace(-c, c, n_sessions)
        dV_arr = np.vstack([V[i] * space for i in range(3)]).T

        for i in trange(n_sessions):
            dV = dV_arr[i]
            temp_action_table = np.vstack(
                (self.first_action, np.hstack((dV, np.nan)))
            )
            if self.reverse:
                time_to_reverse = orbital_period_after_actions(
                    temp_action_table, self.env, self.step)
                temp_action_table = np.vstack(
                    (temp_action_table, -temp_action_table[-1])
                )
                temp_action_table[1, 3] = time_to_reverse

            temp_reward = self.get_reward(temp_action_table)

            if temp_reward > self.policy_reward:
                self.policy_reward = temp_reward
                self.action_table = temp_action_table
        stop = True
        return True
