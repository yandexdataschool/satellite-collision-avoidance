# Module api provides functionality for simulation of
# object movement in space , communication with the space environment
# by the Agent and state/reward exchanging.
#
# In the first implementation wi will have only one protected
# object. All other objects will be treated as space garbage.
# As a first, we will observe only ideal satellite's trajectories,
# so that we can describe any object location at time t after the
# simulation has been started.

# import PyKEP
import numpy as np


class Agent:
    """ Agent implements an agent to communicate with space Environment.
        Agent can make actions to the space environment by taking it's state
        and reward after the last action.
    """

    def __init__(self):
        """"""

    def get_action(self, s, r):
        """ Provides action  for protected object.
        Args:
            s - np.array, state of the environment as matrix.
            r - reward after last action.
        Returns:
            Vector of deltas for protected object.
        """
        action = np.zeros(3)
        return action


class Environment:
    """ Environment provides the space environment with space objects:
        satellites and garbage, in it.
    """

    def __init__(self, protected, garbage):
        """
            protected - SpaceObject, protected space object in Environment.
            grabage - [SpaceObject], list of other space objects.
        """
        self.protected = protected
        self.garbage = garbage
        self.state = EnvState()

    def get_state(self, params):
        """ Provides environment state.
            params -- dict(), which parameters to return in state.
        """
        objects = [self.protected] + self.garbage
        return self.state.get_state(params, objects)

    def get_reward(self, state, next_state, current_reward):
        """
        state, next_state: dict where keys:
            'coord': dict where:
                {'st': np.array shape (n_satellites, 6)},  satellites coordinates
                {'gb': np.array shape (n_items, 6)},  garbage coordinates
            'fuel': float
            'trajectory_deviation_coef': float
        current_reward: float
        ---
        output: float
        """
        sat_coordinates = to_xyz(state['coord']['st'])
        garb_coordinates = to_xyz(state['coord']['gb'])

        # Euclidean distance
        # distances array
        distances = np.sum((sat_coordinates - garb_coordinates) ** 2, axis=1) ** 0.5
        def distance_to_reward(dist_array):
            result = -1. / (dist_array + 0.001)
            return np.sum(result)
        collision_danger = distance_to_reward(distances)
        
        # fuel reward  
        fuel_consumption = -(state['fuel'] - next_state['fuel'])
        
        # trajectory reward
        traj_reward = -state['trajectory_deviation_coef']
        
        # whole reward    
        reward = (
            collision_danger
            + fuel_consumption
            + traj_reward
        ) 
        new_reward = current_reward + reward
        return new_reward

    def act(self, action):
        """ Change direction for protected object.
        Args:
            action -- np.array([dx, dy, dz]), vector of deltas.
        """
        self.protected.act(action)


class EnvState:
    """ Describes Environment state."""

    def __init__(self):
        self.is_end = False
        """"""

    def get_state(self, params, objects):
        """ Provides the state of the environment as matrix.
        Args:
            params - dict(), which parameteres to include into state.
            in the first implementation may consist of keys: "coord", "v".
        """
        return

    def get_coordinates(self, t, objects):
        """"""

    def get_v(self):
        """
        """


class SpaceObject:
    """ SpaceObject represents a satellite or a space garbage. """

    def __init__(self, pos, v, t):
        """
        Args:
            pos -- np.array([x, y, z]), position in space
            v -- velocity,
            t -- start timestamp.
        """
        self.pos = pos
        self.v = v
        self.t = t

    def act(self, action):
        """ Make manoeuvre for the object. """
        return
