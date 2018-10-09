import numpy as np
import pykep as pk
import time

import time

from .train_utils import generate_session_with_env, time_before_early_first_maneuver

from ..api import Environment, MAX_FUEL_CONSUMPTION
from ..simulator import Simulator
from ..agent import TableAgent
from ..utils import read_space_objects


class BaseTableModel:
    """Base class for table models."""

    def __init__(self, env, step, reverse=True,
                 first_maneuver_time="early"):
        """
        Agrs:
            env (Environment): environment with given parameteres.
            step (float): time step in simulation.
            reverse (bool): if True, there are selected exactly 2 maneuvers
                while the second of them is reversed to the first one.
            first_maneuver_time (str): time to the first maneuver. Could be:
                "early": max time to the first maneuver, namely
                    max(0, 0.5, 1.5, 2.5 ... orbital_periods before collision);
                "auto".

        TODO:
            add user's maneuver time?
        """
        self.env = env
        self.step = step
        self.reverse = reverse

        self.first_maneuver_time = first_maneuver_time
        if first_maneuver_time == "early":
            time_to_first_maneuver = time_before_early_first_maneuver(
                self.env, self.step)
        else:
            time_to_first_maneuver = None
        self.time_to_first_maneuver = time_to_first_maneuver

        self.action_table = np.empty((0, 4))
        self.policy_reward = self.get_reward()

        self.protected = env.protected
        self.debris = env.debris

    def train(self, n_iterations=5, print_out=False, *args, **kwargs):
        """Training agent policy (self.action_table).

        Args:
            n_iterations (int): number of iterations.
            print_out (bool): print information during the training.
            *args and **kwargs: iteration arguments, depend on method (inheritor class).

        TODO:
            add early stopping
            add log
            decorate by print_out and log?
        """
        if print_out:
            train_start_time = time.time()
            self.print_start_train()

        i = 0
        while i < n_iterations:
            if print_out:
                print(f"\niteration: {i+1}/{n_iterations}")
            stop = self.iteration(print_out, *args, **kwargs)
            if stop:
                break
            i += 1

        if print_out:
            train_time = time.time() - train_start_time
            self.print_end_train(train_time)

    def iteration(self, print_out, *args, **kwargs):
        pass

    def get_reward(self, action_table=None):
        if action_table is None:
            action_table = self.action_table
        agent = TableAgent(action_table)
        return generate_session_with_env(agent, self.env, self.step)

    def get_action_table(self):
        return self.action_table

    def save_action_table(self, path):
        # TODO - save reward here?
        header = "dVx,dVy,dVz,time to request"
        np.savetxt(path, self.action_table, delimiter=',', header=header)

    def print_start_train(self):
        print(f"\nStart training.\n\nInitial action table:\n{self.action_table}")
        print(f"Initial Reward: {self.policy_reward}")

    def print_end_train(self, train_time):
        self.policy_reward = self.get_reward(self.action_table)
        print("\nTraining completed in {:.5} sec.".format(train_time))
        print(f"Total Reward: {self.policy_reward}")
        print(f"Action Table:\n{self.action_table}")
