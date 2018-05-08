import numpy as np
import pykep as pk

from tqdm import trange

from ...api import Environment, MAX_FUEL_CONSUMPTION
from ...simulator import Simulator
from ...agent import TableAgent as Agent
from ...utils import read_space_objects
from ..train_utils import ProgressPlotter, ProgressLogger, generate_session_with_env


def random_weights(weights_shape, max_time, rand_type="uniform"):
    """ Constrain time_to_request column in action table. """
    # print("Max time: ", max_time)
    if rand_type == "uniform":
        weights = np.random.uniform(-0.1, 0.1, weights_shape)
    elif rand_type == "gauss":
        weights = np.random.randn(*weights_shape)
    time_to_request = weights[:, -1]
    weights[:, -1] = np.minimum(np.full(time_to_request.shape, max_time),
                                np.maximum(time_to_request, np.zeros_like(time_to_request)))
    return weights


class EvolutionStrategies(object):
    """ ... """

    def __init__(self, env, step, weights_shape, sigma, population_size=10, learning_rate=0.1, decay=1.0):
        self.env = env
        self.step = step

        self.n_actions, self.action_size = weights_shape

        self.pop_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.decay = decay

        # TODO: provide learning for neural network agent
        self.weights_shape = weights_shape
        self.max_time = self.env.init_params[
            'end_time'].mjd2000 - self.env.init_params['start_time'].mjd2000
        self.weights = random_weights(weights_shape, self.max_time, "gauss")

        self.reward = -float('inf')
        self.best_reward = -float('inf')
        self.best_weights = self.weights
        self.rewards_per_iter = None

        # TODO: implement logger for training
        self.logger = ProgressLogger()

    def train(self, iterations, print_out=False, print_step=1):
        # np.random.seed(0)
        if print_out:
            self.print_start_train()

        self.rewards_per_iter = np.zeros((iterations, self.pop_size))
        self.actions = np.zeros(
            (iterations, self.pop_size, self.n_actions, self.action_size))

        for iteration in range(iterations):
            rewards = np.zeros(self.pop_size)
            N = np.zeros((self.pop_size, self.n_actions, self.action_size))
            for policy in trange(self.pop_size):
                N[policy] = random_weights(self.weights_shape, self.max_time, "gauss")
                w_try = self.weights + self.sigma * N[policy]
                rewards[policy] = generate_session_with_env(w_try, self.env)
                self.actions[iteration, policy] = w_try
                # update best reward and policy
                if rewards[policy] > self.best_reward:
                    self.best_reward, self.best_weights = rewards[
                        policy], w_try

            self.rewards_per_iter[iteration] = rewards

            # calculate incremental rewards
            A = (rewards - np.mean(rewards)) / np.std(rewards)
            self.weights += self.learning_rate / \
                (self.pop_size * self.sigma) * np.dot(N.T, A).T

            self.learning_rate *= self.decay

            if print_out:
                print("Mean Reward at iter #{}: {}".format(
                    iteration, np.mean(rewards)))

        if print_out:
            self.print_end_train()

    def save(self, path):
        """ Save model to file. """
        header = "dVx,dVy,dVz,time to request"
        np.savetxt(path, self.weights, delimiter=',', header=header)

    def get_weights(self):
        return self.weights

    def get_reward(self):
        return generate_session_with_env(self.weights, self.env)

    def get_best_weights(self):
        return self.best_weights

    def get_best_reward(self):
        return self.best_reward

    def get_rewards_history(self):
        return self.rewards_per_iter

    def print_start_train(self):
        print("Start training.\nInitial action table: {}".format(self.weights))
        print("Initial reward: {}\n".format(self.get_reward()))

    def print_end_train(self):
        print("Training completed.\nTotal reward: {}".format(self.get_reward()))
        print("Action Table: {}\n".format(self.weights))
