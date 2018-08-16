# Generate random collision situation.
import argparse
import sys

import numpy as np
import pykep as pk


from .generator import Generator
from ..api import SpaceObject
from ..models.baseline import Baseline


class GeneratorAntiBaseline(Generator):
    """ Generates such a random collision situation,
        with which the baseline model could not manage effectively.
    """

    def __init__(self, start_time, step, first_maneuver_direction="auto",
                 n_sessions=100, min_time_to_next_maneuver=0.01):

        if isinstance(start_time, (int, float)):
            start_time = pk.epoch(start_time, "mjd2000")
        elif isinstance(start_time, pk.core._core.epoch):
            pass
        else:
            raise TypeError("Invalid start_time type")

        self.start_time = start_time
        self.curr_time = self.start_time

        self.step = step
        self.first_maneuver_direction = first_maneuver_direction
        self.maneuvers_direction = first_maneuver_direction
        self.n_sessions = n_sessions
        self.min_time_to_next_maneuver = min_time_to_next_maneuver

        self.protected = None
        self.debris = []

        self._env = None

    def add_debris(self, max_time_to_collision=0.5, *args, **kwargs):
        """Add debris object on the orbit will be provided by baseline model.

        Args:
            min_time_to_collision after last maneuver or start_time
        """
        if not self.protected:
            raise Exception("no protected object")

        # generate debris object
        end_time = self.curr_time.mjd2000 + max_time_to_collision
        generator = Generator(self.curr_time, end_time)
        generator.protected = self.protected
        generator.add_debris(*args, **kwargs)
        self.debris += generator.debris
        self._env = generator.get_env()

        # baseline maneuver
        # TODO: add debris for forward and backward maneuvers
        print("start", self.curr_time.mjd2000)
        print("end", end_time)
        print("coll", generator.collision_epochs[0].mjd2000)
        model = Baseline(self._env, self.step, reverse=False,
                         first_maneuver_direction=self.first_maneuver_direction)
        model.train(1, False, self.n_sessions)
        maneuvers = model.get_maneuvers()
        print(maneuvers)
        if maneuvers:
            for maneuver in maneuvers:
                time = maneuver["t_man"].mjd2000
                assert self.curr_time.mjd2000 <= time <= end_time, "maneuver time is not in the interval"
                self._env.protected.maneuver(
                    maneuver["action"], maneuver["t_man"])

            # update parameters
            self.curr_time = pk.epoch(
                self.curr_time.mjd2000 + time + self.min_time_to_next_maneuver)
            self.maneuvers_direction = model.maneuvers_direction
        else:
            assert False, "no man"
            self.curr_time = generator.collision_epochs[0]
            pass
