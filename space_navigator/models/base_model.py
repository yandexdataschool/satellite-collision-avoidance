import numpy as np
import pykep as pk
import time

from functools import wraps

# from .train_utils import generate_session_with_env

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

    def __init__(self, env, step, reverse):

        self.env = env
        self.step = step
        self.reverse = reverse

        self.action_table = None
        self.policy_reward = -float("inf")

    # ИЛИ то что ниже ИЛИ декоратор - печатать старт и конец
    # print starn
    # print end
    # @rrr
    def train(self):
        # ""
        # ,
        #           coplanar=True, collinear=True, reverse=True,
        #           maneuver_time="early",
        #           print_out=False):
        # time to maneuver - max, time, auto
        # компланарность/коллинеарность
        # пороги
        """
        if print_out:
                self.print_start()
        for i in trange(n_iterations):
                self.iteration()
                if print_out:
                        self.print_progress()
        if print_out:
                self.print_end()
        """

        """
        #@print_start
        #@print_end
        self._train_model()
        self._train():
            _train()
		"""
        pass

    # def get_reward(self, action_table=self.action_table):
    #     agent = TableAgent(action_table)
    #     return generate_session()

    # def save_action_table(self, save_path):
    #     # TODO - save reward here?
    #     header = "dVx,dVy,dVz,time to request"
    #     np.savetxt(save_path, self.action_table, delimiter=',', header=header)

    # def print_start_train(self):
    # 	pass

    # def print_end_train(self):
    # 	pass

    # def print_progress(self)"
    # pass"
