import numpy as np
from copy import copy
import pykep as pk
import pandas as pd

import sys
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from api import Environment, MAX_FUEL_CONSUMPTION
from simulator import Simulator, read_space_objects
from agent import TableAgent as Agent


np.random.seed(0)


def generate_session(protected, debris, agent, start_time, end_time, step):
    start_time_mjd2000 = pk.epoch(start_time, "mjd2000")
    env = Environment(copy(protected), copy(debris), start_time_mjd2000)
    simulator = Simulator(agent, env, update_r_p_step=None, print_out=False)
    # reward
    reward = simulator.run(end_time, step, visualize=False)
    return reward


def constrain_action(action, max_fuel_cons):
    fuel_cons = np.sum(action[:3])
    if fuel_cons > max_fuel_cons:
        action[:3] *= max_fuel_cons / fuel_cons
    action[3] = max(0.001, action[3])
    return action


def random_action_table(mu_table, sigma_table, max_fuel_cons, fuel_level):
    n_actions = mu_table.shape[0]
    action_table = np.zeros_like(mu_table)
    for i in range(n_actions - 1):
        for j in range(4):
            action_table[i, j] = np.random.normal(
                mu_table[i, j], sigma_table[i, j])
        action_table[i] = constrain_action(action_table[i], max_fuel_cons)
        fuel_level -= np.sum(action_table[i, :3])
        max_fuel_cons = min(max_fuel_cons, fuel_level)
    for j in range(3):
        action_table[-1, j] = np.random.normal(
            mu_table[-1, j], sigma_table[-1, j])
    action_table[-1] = constrain_action(action_table[-1], max_fuel_cons)
    action_table[-1, -1] = np.nan
    return action_table


class CrossEntropy:

    def __init__(self, protected, debris, start_time, end_time, step,
                 max_fuel_cons=10, fuel_level=20, n_actions=3):

        self.env = Environment(copy(protected), copy(debris), start_time)
        self.protected = protected
        self.debris = debris
        self.start_time = start_time
        self.end_time = end_time
        self.max_time = self.end_time - self.start_time
        self.step = step
        self.n_actions = n_actions
        self.fuel_level = fuel_level
        self.action_table = np.zeros((self.n_actions, 4))
        self.action_table[:, 3] = self.max_time / n_actions
        # self.mu_table[-1, -1] = np.nan
        self.sigma_table = np.ones((self.n_actions, 4))  # * 0.5
        self.sigma_table[:, 3] = 0.01
        # self.sigma_table[-1, -1] = np.nan
        self.max_fuel_cons = max_fuel_cons

    def train(self, n_iterations=10, n_sessions=20, n_best_actions=4, learning_rate=0.7, sigma_coef=1, learning_rate_coef=1):
        # TODO - percentile + perc coef
        # TODO - stop if reward change < epsilon
        # TODO - careful update
        agent = Agent(self.action_table)
        print('initial action table:', self.action_table)
        print('initial reward:', generate_session(self.protected, self.debris,
                                                  agent, self.start_time, self.end_time, self.step))
        for i in range(n_iterations):
            print("i:", i)
            rewards = []
            action_tables = []
            for j in range(n_sessions):
                # TODO - parallel this part
                print("j:", j)
                action_table = random_action_table(
                    self.action_table, self.sigma_table, self.max_fuel_cons, self.fuel_level)
                print(action_table)
                agent = Agent(action_table)
                action_tables.append(action_table)
                reward = generate_session(self.protected, self.debris,
                                          agent, self.start_time, self.end_time, self.step)
                print(reward)
                rewards.append(reward)
            best_rewards = np.argsort(rewards)[-n_best_actions:]
            best_action_tables = np.array(action_tables)[best_rewards]
            new_action_table = np.mean(best_action_tables, axis=0)
            self.action_table = (
                new_action_table * learning_rate
                + self.action_table * (1 - learning_rate)
            )
            print('best rewards', np.array(rewards)[best_rewards])
            print("new_action_table", new_action_table)
            print("action_table", self.action_table)
            print(self.get_total_reward(), '\n')
            self.sigma_table *= sigma_coef
            learning_rate *= learning_rate
        print("training completed")

    def set_action_table(self, action_table):
        self.action_table = action_table

    def get_action_table(self):
        return self.action_table

    def get_total_reward(self):
        agent = Agent(self.action_table)
        self.total_reward = generate_session(self.protected, self.debris,
                                             agent, self.start_time, self.end_time, self.step)
        return self.total_reward

    def save_action_table(self, path):
        header = "dVx,dVy,dVz,time to request"
        np.savetxt(path, self.action_table, delimiter=',', header=header)
