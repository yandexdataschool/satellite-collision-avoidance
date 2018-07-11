import numpy as np
import pykep as pk
import time

import time

from .train_utils import generate_session_with_env, time_to_early_first_maneuver

from ..api import Environment, MAX_FUEL_CONSUMPTION
from ..simulator import Simulator
from ..agent import TableAgent
from ..utils import read_space_objects


# def rrr(func):
#     print("ffff")

#     @wraps(func)
#     def wrapper(self, *args, **kwargs):
#         print("111111111111/n/n/n11/n/n/n")
#         func(*args, **kwargs)
#         print("33333333333")

#     print("eeeeee")
#     return wrapper


class BaseTableModel:

    def __init__(self, env, step, reverse=True,
                 first_maneuver_time="early"):
        """
        first_maneuver_time:
            early - либо за максимальный полу-виток, либо сразу (если столкновения раньше)
            auto - сам выбирает (для baseline перебирает все полувитки)
        """
        self.env = env
        self.step = step
        self.reverse = reverse

        self.first_maneuver_time = first_maneuver_time
        if first_maneuver_time == "early":
            time_to_first_maneuver = time_to_early_first_maneuver(
                self.env, self.step)
        else:
            time_to_first_maneuver = None
        self.time_to_first_maneuver = time_to_first_maneuver

        self.action_table = None
        self.policy_reward = -float("inf")

        self.protected = env.protected
        self.debris = env.debris

        # ИЛИ то что ниже ИЛИ декоратор - печатать старт и конец
        # print starn
        # print end
        # @rrr
    def train(self, n_iterations=5, print_out=False, *args, **kwargs):
        # ""
        # ,
        #           coplanar=True, collinear=True, reverse=True,
        #           maneuver_time="early",
        #           print_out=False):
        # time to maneuver - max, time, auto
        # компланарность/коллинеарность
        # пороги
        """

                """
        if print_out:
            train_start_time = time.time()
            self.print_start_train()
        for i in range(n_iterations):
            if print_out:
                print(f"\niteration: {i+1}/{n_iterations}")
            self.iteration(print_out, *args, **kwargs)
            # Добавить ранюю остановку? в методах RL не двигаться если нет
            # улучшений?
        if print_out:
            train_time = time.time() - train_start_time
            self.print_end_train(train_time)

    def iteration(self, print_out, *args, **kwargs):  # , *args, **kwargs):
        pass

    def get_reward(self, action_table):
        agent = TableAgent(action_table)
        # return generate_session(self.protected, self.debris, agent,
        #                         self.start_time, self.end_time, self.step)
        return generate_session_with_env(agent, self.env, self.step)

    def save_action_table(self, path):
        # TODO - save reward here?
        header = "dVx,dVy,dVz,time to request"
        np.savetxt(path, self.action_table, delimiter=',', header=header)

    def print_start_train(self):
        print(f"\nStart training.\n\nInitial action table:\n{self.action_table}")
        print(f"Initial Reward: {self.get_reward(self.action_table)}")

    def print_end_train(self, train_time):
        print("\nTraining completed in {:.5} sec.".format(train_time))
        print(f"Total Reward: {self.get_reward(self.action_table)}")
        print(f"Action Table:\n{self.action_table}")
