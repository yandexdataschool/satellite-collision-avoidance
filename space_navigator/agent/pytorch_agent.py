import os
import math
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import NNAgent

# IDEAS:
# * adaptive num actions


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)


class PytorchAgent(torch.nn.Module, NNAgent):

    def __init__(self, num_inputs, num_outputs, weights=[]):
        super(PytorchAgent, self).__init__()
        """
        Args:
            action_table (np.array with shape(n_actions, 4)):
                table of actions with columns ["dVx", "dVy", "dVz", "time to request"].

        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.linear1 = nn.Linear(num_inputs, 64)
        self.linear2 = nn.Linear(64, 64)
        self.actor_linear = nn.Linear(64, num_outputs)

        if len(weights):
            for i, param in enumerate(self.parameters()):
                try:
                    param.data.copy_(weights[i])
                except:
                    param.data.copy_(weights[i].data)

    def forward(self, inputs):
        inputs = Variable(torch.FloatTensor(inputs)).view(-1, self.num_inputs)
        x = selu(self.linear1(inputs))
        x = selu(self.linear2(x))
        return self.actor_linear(x)

    def get_action(self, state):
        """ Provides action for protected object.

        Args:
            state (dict): environment state
                {'coord' (dict):
                    {'st' (np.array with shape (1, 6)): satellite r and Vx, Vy, Vz coordinates.
                     'debr' (np.array with shape (n_items, 6)): debris r and Vx, Vy, Vz coordinates.}
                'trajectory_deviation_coef' (float).
                'epoch' (pk.epoch): at which time environment state is calculated.
                'fuel' (float): current remaining fuel in protected SpaceObject. }.

        Returns:
            action (np.array([dVx, dVy, dVz, time_to_req])):
                vector of deltas for protected object (m/s),
                step in time when to request the next action (mjd2000).

        """
        # epoch = state["epoch"].mjd2000
        # state = 
        action = self.forward(state).data.numpy()[0]
        return action

    def get_params(self):
        """
        The params that should be trained by ES
        """
        return [(k, v) for k, v in zip(self.state_dict().keys(),
                                       self.state_dict().values())]
