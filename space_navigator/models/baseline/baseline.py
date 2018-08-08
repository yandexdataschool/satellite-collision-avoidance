# Baseline - selection of a prograde/retrograde maneuver.

import numpy as np
import pykep as pk
import time
from tqdm import trange
from copy import copy

from ...api import Environment, MAX_FUEL_CONSUMPTION
from ...simulator import Simulator
from ...agent import TableAgent as Agent

from ..base_model import BaseTableModel
from ..collinear_GS.collinear_GS import CollinearGridSearch
from ..train_utils import (
    orbital_period_after_actions, change_orbit,
    collision_data, generate_session,
)


class Baseline(BaseTableModel):
    """Provides prograde/retrograde maneuvers collision-by-collision."""

    def __init__(self, env, step, reverse=True):
        """
        Agrs:
            env (Environment): environment with given parameteres.
            step (float): time step in simulation.
            reverse (bool):
                if True: there are selected exactly 2 maneuvers
                    while the second of them is reversed to the first one;
                if False: one maneuver.

        TODO:
            tests - compare with CollinearGridSearch
            return to the initial orbit

        """
        super().__init__(env, step, reverse, first_maneuver_time="early")

        self.start_time = self.env.get_start_time()
        self.end_time = self.env.get_end_time()

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
        agent = Agent()
        collisions = collision_data(self.env, self.step, agent)
        action_table = copy(self.action_table)
        env = copy(self.env)
        start_time = copy(self.start_time)

        while True:
            if not collisions:
                break

            # next collision info
            epsilon = 0.01
            next_collision_epoch = pk.epoch(
                collisions[0]['epoch'] + epsilon, "mjd2000")
            debris_id = collisions[0]['debris_id']

            # new narrowed environment with only next collision
            narrowed_env = Environment(
                env.protected, [env.debris[debris_id]],
                start_time, next_collision_epoch,
            )

            # Collinear Grid Search model for narrowed environment
            model_GS = CollinearGridSearch(
                narrowed_env, self.step, reverse=False)  # TODO: reverse=reverse
            model_GS.train(1, False, n_sessions)
            action_table = model_GS.action_table

            # update temporary environment
            t_man = start_time.mjd2000 + model_GS.time_to_first_maneuver
            t_man = pk.epoch(t_man, "mjd2000")
            start_time = t_man.mjd2000 + self.step
            start_time = pk.epoch(start_time, "mjd2000")
            new_env = Environment(
                protected=copy(env.protected),
                debris=copy(env.debris),
                start_time=start_time,
                end_time=self.end_time,
            )
            if len(action_table) > 0:
                # change trajectory of protected object
                # according to obtained maneuver
                new_env.protected.maneuver(action_table[1, :3], t_man)
                # add action to the actions table
                action_table[-1, 3] = self.step
                self.action_table = np.vstack(
                    (self.action_table, action_table))
            env = new_env

            # print iteration info
            if print_out:
                print(f"Current epoch: {start_time.mjd2000}")
                print("Collision information:")
                print(f"    Debris name: {collisions[0]['debris_name']};")
                print(f"    Collision epoch: {collisions[0]['epoch']} (mjd2000);")
                print(f"    Distance: {collisions[0]['distance']} (meters);")
                print(f"    Probability: {collisions[0]['probability']}.")

            # update information about collisions
            agent = Agent()
            new_collisions = collision_data(env, self.step, agent)
            if new_collisions:
                # check if collision with last piece of debris
                # is still possible
                if new_collisions[0]['debris_id'] == new_collisions[0]['debris_id']:
                    # check if it is not another turn around the Earth
                    atol = 0.01  # (days) 0.01 days == 14.4 minutes
                    if np.isclose(
                            new_collisions[0]['epoch'],
                            collisions[0]['epoch'],
                            atol=atol,
                    ):
                        # print("11111111111111111111111", new_collisions)
                        new_collisions = new_collisions[1:] or []
                        # print("11111111111111111111111", new_collisions)
            collisions = new_collisions

        if self.reverse:
            # TODO
            change_orbit()

        stop = True
        return stop
