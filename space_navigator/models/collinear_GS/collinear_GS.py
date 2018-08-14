# Collinear Grid Search - selection of a prograde/retrograde maneuver.

import numpy as np
import pykep as pk
from tqdm import trange
import time

from ...api import Environment, MAX_FUEL_CONSUMPTION
from ...simulator import Simulator
from ...agent import TableAgent as Agent

from ..base_model import BaseTableModel
from ..train_utils import orbital_period_after_actions


class CollinearGridSearch(BaseTableModel):
    """Provides prograde/retrograde maneuvers through Grid Search."""

    def __init__(self, env, step, reverse=True, first_maneuver_direction="auto"):
        """
        Agrs:
            env (Environment): environment with given parameteres.
            step (float): time step in simulation.
            reverse (bool): 
                if True: there are selected exactly 2 maneuvers
                    while the second of them is reversed to the first one;
                if False: one maneuver.
            first_maneuver_direction (str): first maneuver is collinear
                to the velocity vector and could be:
                    "forward" (co-directed)
                    "backward" (oppositely directed)
                    "auto" (just collinear).
        """
        super().__init__(env, step, reverse, first_maneuver_time="early")

        self.start_time = self.env.get_start_time()
        self.first_maneuver_direction = first_maneuver_direction
        self.first_action = np.array(
            [0, 0, 0, self.time_to_first_maneuver])

    def iteration(self, print_out=False, n_sessions=100):
        """Training iteration.

        Args:
            print_out (bool): print iteration information.
            n_sessions (int): number of sessions to generate.

        Returns:
            stop (bool): whether to stop training after iteration.

        """
        stop = True
        if self.time_to_first_maneuver is None:
            # check there are no collisions
            return stop

        max_fuel = self.env.init_fuel / 2 if self.reverse else self.env.init_fuel
        max_fuel = min(MAX_FUEL_CONSUMPTION, max_fuel)
        first_maneuver_epoch = pk.epoch(
            self.start_time.mjd2000 + self.time_to_first_maneuver, "mjd2000")
        _, V = self.protected.position(first_maneuver_epoch)
        max_dV = max_fuel / np.linalg.norm(V)

        if self.first_maneuver_direction == "auto":
            space = np.linspace(-max_dV, max_dV, n_sessions + n_sessions % 2)
        elif self.first_maneuver_direction == "forward":
            space = np.linspace(0, max_dV, n_sessions + 1)[1:]
        elif self.first_maneuver_direction == "backward":
            space = np.linspace(-max_dV, 0, n_sessions + 1)[:-1]
        else:
            raise ValueError("Invalid first maneuver direction type")

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

        return stop
