import copy
import numpy as np
import pykep as pk
import torch

from tqdm import trange

from ...api import Environment
from ...agent import PytorchAgent
from ..train_utils import ProgressPlotter, ProgressLogger, generate_session_with_env, constrain_action


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class PytorchES(object):
    """ ... """

    def __init__(self, env, step, num_inputs, num_outputs, sigma, population_size=10, learning_rate=0.1, decay=1.0):
        self.env = env
        self.step = step

        self.pop_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.decay = decay

        self.num_inputs, self.num_outputs = num_inputs, num_outputs
        self.agent = PytorchAgent(self.num_inputs, self.num_outputs)
        self.weights = list(self.agent.parameters())

        self.reward = -float('inf')
        self.best_reward = -float('inf')
        self.best_weights = self.weights
        self.rewards_per_iter = None

        # TODO: implement logger for training
        self.logger = ProgressLogger()

    def jitter_weights(self, weights, population=[]):
        new_weights = []
        for i, param in enumerate(weights):
            jittered = torch.from_numpy(
                self.sigma * population[i]).float()
            new_weights.append(param.data + jittered)
        return new_weights

    def train(self, iterations, print_out=False, print_step=1):
        if print_out:
            self.print_start_train()

        for iteration in range(iterations):
            rewards = np.zeros(self.pop_size)

            policies = []

            for policy in trange(self.pop_size):
                population = []
                for param in self.weights:
                    population.append(np.random.randn(*param.data.size()))
                policies.append(population)

                new_weights = self.jitter_weights(
                    copy.deepcopy(self.weights), population=population)
                agent = PytorchAgent(
                    self.num_inputs, self.num_outputs, weights=new_weights)

                # generate_session_with_env(agent, self.env)
                rewards[policy] = -policy
                print(f'CURRR R: {rewards[policy]}')
                # self.actions[iteration, policy] =

                # update best reward and policy
                if rewards[policy] > self.best_reward:
                    self.best_reward, self.best_weights = rewards[
                        policy], new_weights

            if np.std(rewards) != 0:
                # calculate incremental rewards
                N = (rewards - np.mean(rewards)) / np.std(rewards)
                for i, param in enumerate(self.weights):
                    A = np.array([p[i] for p in policies])
                    param.data = param.data + self.learning_rate / \
                        (self.pop_size * self.sigma) * torch.from_numpy(
                            np.dot(A.T, N).T).float()

                self.learning_rate *= self.decay

            if print_out:
                print(f"Mean Reward at iter #{iteration}: {np.mean(rewards)}")

        if print_out:
            self.print_end_train()

    def save(self, path):
        """ Save model to file. """
        torch.save(self.agent.state_dict(), os.path.join(path, 'latest.pth'))

    def get_weights(self):
        return self.weights

    def get_reward(self):
        return generate_session_with_env(self.agent, self.env)

    def get_best_weights(self):
        return self.best_weights

    def get_best_reward(self):
        return self.best_reward

    def print_start_train(self):
        print("Start training.\n")
        print(f"Initial reward: {self.get_reward()}\n")
s
    def print_end_train(self):
        print(f"Training completed.\nTotal reward: {self.get_reward()}")
