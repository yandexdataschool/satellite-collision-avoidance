import numpy as np

from . import BaseAgent
from .agent_utils import adjust_action_table


class TableAgent(BaseAgent):

    def __init__(self, action_table=np.array([])):
        """
        Args:
            action_table (np.array with shape=(n_actions, 4) or (4) or (0)):
                table of actions with columns ["dVx", "dVy", "dVz", "time to request"].

        """
        self.action_table = adjust_action_table(action_table)
        # TODO: add index and not delete actions from table
        # self.action_idx = 0

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
        epoch = state["epoch"].mjd2000
        if not self.action_table.size:
            action = np.array([0, 0, 0, np.nan])
            return action
        # TODO - first nan can't be - fix
        action = self.action_table[0]
        self.action_table = np.delete(self.action_table, 0, axis=0)
        return action

    def get_action_table(self):
        return self.action_table
