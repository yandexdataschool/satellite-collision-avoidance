# Module api provides functionality for simulation of
# object movement in space , communication with the space environment
# by the Agent and state/reward exchanging.
#
# In the first implementation wi will have only one protected
# object. All other objects will be treated as space debris.
# As a first, we will observe only ideal satellite's trajectories,
# so that we can describe any object location at time t after the
# simulation has been started.

# import pykep
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
            Vector of ΔV for protected object.
        """
        action = np.zeros(4)
        return action


class Environment:
    """ Environment provides the space environment with space objects:
        satellites and debris, in it.
    """

    def __init__(self, protected, debris):
        """
            protected - SpaceObject, protected space object in Environment.
            grabage - [SpaceObject], list of other space objects.
        """
        self.protected = protected
        self.debris = debris
        self.state = EnvState()

    def get_state(self, params):
        """ Provides environment state.
            params -- dict(), which parameters to return in state.
        """
        objects = [self.protected] + self.debris
        return self.state.get_state(params, objects)

    def get_reward(self):
        """"""
        return 0

    def act(self, action):
        """ Change velocity for protected object.
        Args:
            action -- np.array([dVx, dVy, dVz, pk.epoch]), vector of deltas.
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
        """"""


class SpaceObject:
    """ SpaceObject represents a satellite or a space debris. """

    def __init__(self, pos, v, t, f):
        """
        Args:
            pos -- np.array([x, y, z]), position in space.
            v -- np.array([Vx, Vy, Vz]), velocity.
            t -- pykep.epoch, start time in epoch format.
            f -- float, initial fuel capacity.
        """
        self.pos = pos
        self.v = v
        self.t = t
        self.f = f

    def act(self, action):
        """ Make manoeuvre for the object. """
        return
