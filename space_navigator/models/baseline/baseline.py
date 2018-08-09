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

        TODO:
            tests - compare with CollinearGridSearch
            return to the initial orbit

        """
        super().__init__(env, step, reverse, first_maneuver_time="early")

        self.start_time = self.env.get_start_time()
        self.end_time = self.env.get_end_time()
        self.min_time_to_next_maneuver = 0.01  # (days)  # TODO: check
        self.first_maneuver_direction = first_maneuver_direction
        self.maneuvers_direction = first_maneuver_direction
        self._avoided_collisions = []

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
            # print("----", "n coll", len(collisions))
            if not collisions:
                break

            # print collision info
            if print_out:
                print(f"\nCurrent time: {start_time.mjd2000}")
                print(f"Collision with: {collisions[0]['debris_name']}.")
                print("Collision information before maneuver:")
                print(f"    Time: {collisions[0]['epoch']} (mjd2000);")
                print(f"    Distance: {collisions[0]['distance']} (meters);")
                print(f"    Probability: {collisions[0]['probability']}.")

            # next collision info before maneuver
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
            if print_out:
                print("Training:")
            model_GS = CollinearGridSearch(
                narrowed_env, self.step, False, self.maneuvers_direction)
            model_GS.train(1, False, n_sessions)
            action_table = model_GS.action_table

            # update environment
            if len(action_table) > 0:
                t_man = start_time.mjd2000 + model_GS.time_to_first_maneuver
                t_man = pk.epoch(t_man, "mjd2000")
                start_time = t_man.mjd2000 + self.min_time_to_next_maneuver
                start_time = pk.epoch(start_time, "mjd2000")
                new_env = Environment(
                    protected=copy(env.protected),
                    debris=copy(env.debris),
                    start_time=start_time,
                    end_time=self.end_time,
                )
                # change trajectory of protected object
                # according to obtained maneuver
                new_env.protected.maneuver(action_table[1, :3], t_man)
                # add action to the actions table
                action_table[-1, 3] = self.step
                self.action_table = np.vstack(
                    (self.action_table, action_table))
                env = new_env

            # update maneuvers direction
            if self.maneuvers_direction == 'auto' and len(action_table) > 0:
                _, V = env.protected.position(
                    start_time.mjd2000 + self.action_table[0, 3])
                is_forward = np.sum(action_table[1, :3] / V) >= 0
                self.maneuvers_direction = 'forward' if is_forward else 'backward'

            # new collisions info after maneuver
            if len(action_table) > 0:
                agent = Agent()
                new_collisions = collision_data(env, self.step, agent)
                if new_collisions:
                    # skip avoided collisions
                    # TODO: optimize skipping
                    skip_list = []
                    for i, new_coll in enumerate(new_collisions):
                        for avoided_coll in self._avoided_collisions:
                            # check debris id
                            if new_coll[0]['debris_id'] == avoided_coll[0]['debris_id']:
                                # check if it is not
                                # another turn around the Earth
                                atol = 0.01  # (days) 0.01 days == 14.4 minutes
                                if np.isclose(
                                    new_collisions[0]['epoch'],
                                    collisions[0]['epoch'],
                                    atol=atol,
                                ):
                                    skip_list.append(i)
                    new_collisions = list(np.delete(new_collisions, skip_list))
            else:
                # skip collision if it is better without maneuver
                self._avoided_collisions.append(collisions[0])
                new_collisions = collisions[1:] or []

            # print collision info after maneuver
            if print_out:
                if len(action_table) > 0:
                    maneuver = action_table[-1]
                    maneuver[3] = self.start_time.mjd2000 + np.sum(
                        self.action_table[:-1, 3])
                    print(f"Maneuver: {maneuver}")
                else:
                    print("Maneuver: no maneuvers.")

            # update collisions information
            collisions = new_collisions

        if self.reverse:
            # TODO
            change_orbit()

        stop = True
        return stop
