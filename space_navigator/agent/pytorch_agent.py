import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import NNAgent


def convert_state_to_numpy(state):
    """ Provides environment state as objects positions in numpy array.

    Args:
        state (dict): Environment state as dictionary.

    Returns:
        numpy_state (np.array): object positions as numpy array.
    """

    numpy_state = np.vstack((
        state['coord']['st'][:, :3],
        state['coord']['debr'][:, :3],)
    )
    return numpy_state


class PytorchAgent(torch.nn.Module, NNAgent):

    def __init__(self, num_inputs, num_outputs, hidden_size, weights=[]):
        """
        Args:
            num_inputs (int): number of inputs.
            num_outputs (int): number of outputs.
            hidden_size (int): hidden size.
            weights ([torch.Variable, ]): list of initial weights for net.

        """
        super(PytorchAgent, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, num_outputs)
        )

        if len(weights):
            for i, param in enumerate(self.parameters()):
                try:
                    param.data.copy_(weights[i])
                except:
                    param.data.copy_(weights[i].data)

    def forward(self, inputs):
        inputs = Variable(torch.FloatTensor(inputs)).view(-1, self.num_inputs)
        return self.net(inputs)

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
        state = convert_state_to_numpy(state)
        action = self.forward(state).data.numpy()[0]
        action /= np.mean(action)
        action[-1] = np.nan
        return action
