
import pykep as pk
import numpy as np


class TableAgent:
    """ Agent implements an agent to communicate with space Environment.

        Agent can make actions to the space environment by taking it's state
        after the last action.

    """

    def __init__(self, action_table=np.array([])):
        """
        Args:
            action_table (np.array with shape(n_actions, 4)):
                table of actions with columns ["dVx", "dVy", "dVz", "time to request"].

        """
        self.action_table = action_table

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
        action = self.action_table[0]
        self.action_table = np.delete(self.action_table, 0, axis=0)
        return action
