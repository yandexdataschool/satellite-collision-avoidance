
import pykep as pk
import numpy as np
import pandas as pd


class TableAgent:
    """ Agent implements an agent to communicate with space Environment.

        Agent can make actions to the space environment by taking it's state
        after the last action.

    """

    def __init__(self, table_path="data/action_table.csv"):
        """
        Args:
            table_path (str): path to table of actions (.csv).

        """
        self.action_table = pd.read_csv(table_path, index_col=0).values

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
        time_to_req = 0.001  # mjd2000
        dVx, dVy, dVz = 0, 0, 0
        # print(epoch)
        # action = np.array([0, 0, 0, epoch, 0])  #: default action
        # if self.action_table.size:
        #     if (epoch >= self.action_table[0, 0]):
        #         action = np.hstack(
        #             [self.action_table[0, 1:], epoch, 0])
        #         print("maneuver!:", action)
        #         self.action_table = np.delete(self.action_table, 0, axis=0)
        action = np.array([dVx, dVy, dVz, time_to_req])
        return action
