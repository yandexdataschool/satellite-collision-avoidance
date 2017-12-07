# Module api provides functionality for simulation of
# object movement in space , communication with the space environment
# by the Agent and state/reward exchanging.
#
# In the first implementation wi will have only one protected
# object. All other objects will be treated as space debris.
# As a first, we will observe only ideal satellite's trajectories,
# so that we can describe any object location at time t after the
# simulation has been started.

import pykep as pk
import numpy as np


def euclidean_distance(xyz_main, xyz_other, rev_sort=True):
    """ Return array of (reverse sorted) Euclidean distances between main object and other
    Args:
        xyz_main: np.array shape (1, 3) - coordinates of main object
        xyz_other: np.array shape (n_objects, 3) - coordinates of other object
    Returns:
        np.array shape (n_objects)
    """
    # distances array
    distances = np.sum(
        (xyz_main - xyz_other) ** 2,
        axis=1) ** 0.5
    if rev_sort:
        distances = np.sort(distances)[::-1]
    return distances


def fuel_consumption(dV):
    """ Provide the value of fuel consumption for given velocity delta.
    dV -- np.array([dVx, dVy, dVz]), vector of satellite velocity delta for maneuver.
    ---
    output: float.
    """
    return np.sum(dV)


class Agent:
    """ Agent implements an agent to communicate with space Environment.
        Agent can make actions to the space environment by taking it's state
        and reward after the last action.
    """

    def __init__(self):
        """"""

    def get_action(self, s):
        """ Provides action  for protected object.
        Args:
            state -- dict where keys:
                'coord' -- dict where:
                    {'st': np.array shape (1, 6)},  satellite xyz and dVx, dVy, dVz coordinates
                    {'db': np.array shape (n_items, 6)},  debris xyz and dVx, dVy, dVz coordinates
                'trajectory_deviation_coef' -- float
            r -- reward after last action.
        Returns:
            np.array([dVx, dVy, dVz, pk.epoch, time_to_req])  - vector of deltas for
            protected object, maneuver time and time to request the next action.
        """
        dVx, dVy, dVz = 0, 0, 0
        time_to_req = 1
        action = np.array([dVx, dVy, dVz, s.get("epoch").mjd2000, time_to_req])
        return action


class Environment:
    """ Environment provides the space environment with space objects:
        satellites and debris, in it.
    """

    def __init__(self, protected, debris):
        """
            protected - SpaceObject, protected space object in Environment.
            debris - [SpaceObject], list of other space objects.
        """
        self.protected = protected
        self.debris = debris
        self.reward = 0
        self.next_action = pk.epoch(0)
        self.state = dict()
        # critical convergence distance
        # TODO choose true distance
        self.crit_conv_dist = 100

    def get_reward(self, state, action):
        """
        state: dict where keys:
            'coord': dict where:
                {'st': np.array shape (1, 6)},  satellite xyz and dVx, dVy, dVz coordinates
                {'db': np.array shape (n_items, 6)},  debris xyz and dVx, dVy, dVz coordinates
            'trajectory_deviation_coef': float
        current_reward: float
        n_closest: number of nearest dabris objects to be considered
        ---
        output: float
        """
        # # min Euclidean distances
        # distances = euclidean_distance(
        #     state['coord']['st'][:, :3],
        #     state['coord']['db'][:, :3],
        # )[:n_closest]
        #
        # def distance_to_reward(dist_array):
        #     result = -1. / (dist_array + 0.001)
        #     return np.sum(result)
        # collision_danger = distance_to_reward(distances)

        # collision probabitity (for uniform distribution)
        # critical distances
        crit_dist = euclidean_distance(
            state['coord']['st'][:, :3],
            state['coord']['db'][:, :3],
            rev_sort=False
        )
        crit_dist = crit_dist[crit_dist < self.crit_conv_dist]
        r = self.crit_conv_dist
        d = crit_dist
        coll_prob = (
            (4 * r + d) * ((2 * r - d) ** 2) / (16 * r**3)
        )
        coll_prob = (1 - np.prod(1 - coll_prob))
        coll_prob_reward = -coll_prob

        fuel_consum = fuel_consumption(action[:3])

        # trajectory reward
        traj_reward = -state['trajectory_deviation_coef']

        # whole reward
        # TODO - add constants to all reward components
        # reward
        r = (
            # collision_danger
            + fuel_consum
            + traj_reward
            + coll_prob_reward
        )
        self.reward += r
        return self.reward

    def get_curr_reward(self):
        """ Provide the last calculated reward. """
        return self.reward

    def act(self, action):
        """ Change velocity for protected object.
        Args:
            action -- np.array([dVx, dVy, dVz, pk.epoch, time_to_req]), vector of deltas for
            protected object, maneuver time and time to request the next action.
        """
        # TODO(dsdubov): populate the function.
        # Learn how to make action for pykep.planet [tle or keplerian] object.
        self.next_action = pk.epoch(self.state.get(
            "epoch").mjd2000 + action[4], "mjd2000")
        self.protected.act(action)

    def get_state(self, epoch):
        """ Provides environment state as dictionary
            and is_end flag.
        Args:
            epoch -- pk.epoch, at which time to return environment state.
        ---
        output: bool, dict()
        """
        st_pos, st_v = self.protected.position(epoch)
        st = np.hstack((np.array(st_pos), np.array(st_v)))
        st = np.reshape(st, (1, -1))
        n_items = len(self.debris)
        db = np.zeros((n_items, 6))
        for i in range(n_items):
            pos, v = self.debris[i].position(epoch)
            db[i] = np.hstack((np.array(pos), np.array(v)))
        db = np.reshape(db, (1, -1))

        coord = dict(st=st, db=db)
        self.state = dict(
            coord=coord, trajectory_deviation_coef=0.0, epoch=epoch)
        self.is_end = self.check_collision()
        return self.is_end, self.state

    def check_collision(self, collision_distance=100):
        """ Return True if collision with protected object appears. """
        # distance satellite and nearest debris objuect
        min_distance = euclidean_distance(
            self.state['coord']['st'][:, :3],
            self.state['coord']['db'][:, :3],
        )[0]
        if min_distance <= collision_distance:
            return True
        return False


class SpaceObject:
    """ SpaceObject represents a satellite or a space debris. """

    def __init__(self, name, param_type, params):
        """
        Args:
            name -- str, name of satellite or a space debris.
            param_type -- str, initial parameteres type. Can be:
                    "tle" -- for TLE,
                    "eph" -- ephemerides, position and velocity state vectors.
                    "osc" -- osculating elements, 6 orbital parameteres.
            params -- dict, dictionary of space object coordinates. Keys are:
                "fuel" -- float, initial fuel capacity.

                for "tle" type:
                    "tle1" -- str, tle line1
                    "tle2" -- str, tle line2

                for "eph" type:
                    "pos" -- [x, y, z], position (cartesian).
                    "vel" -- [Vx, Vy, Vz], velocity (cartesian).
                    "epoch" -- pykep.epoch, start time in epoch format.
                    "mu_central_body" -- float, gravity parameter of the
                                                central body (SI units, i.e. m^2/s^3).
                    "mu_self" -- float, gravity parameter of the planet
                                        (SI units, i.e. m^2/s^3).
                    "radius" -- float, body radius (SI units, i.e. meters).
                    "safe_radius" -- float, mimimual radius that is safe during
                                            a fly-by of the planet (SI units, i.e. m)

                for "osc" type:
                    "elements" -- (a,e,i,W,w,M), tuple containing 6 osculating elements.
                    "epoch" -- pykep.epoch, start time in epoch format.
                    "mu_central_body", "mu_self", "radius", "safe_radius" -- same, as in "eph" type.
        """
        self.fuel = params["fuel"]

        if param_type == "tle":
            tle = pk.planet.tle(
                params["tle_line1"], params["tle_line2"])

            t0 = pk.epoch(tle.ref_mjd2000, "mjd2000")
            mu_central_body, mu_self = tle.mu_central_body, tle.mu_self
            radius, safe_radius = tle.radius, tle.safe_radius

            elements = tle.osculating_elements(t0)
            self.satellite = pk.planet.keplerian(
                t0, elements, mu_central_body, mu_self, radius, safe_radius)
        elif param_type == "eph":
            self.satellite = pk.planet.keplerian(params["epoch"],
                                                 params["pos"], params["vel"],
                                                 params["mu_central_body"],
                                                 params["mu_self"],
                                                 params["radius"],
                                                 params["safe_radius"])
        elif param_type == "osc":
            self.satellite = pk.planet.keplerian(params["epoch"],
                                                 params["elements"],
                                                 params["mu_central_body"],
                                                 params["mu_self"],
                                                 params["radius"],
                                                 params["safe_radius"])
        else:
            raise ValueError("Unknown initial parameteres type")
        self.satellite.name = name

    def act(self, action):
        """ Make manoeuvre for the object. """
        dV = action[:3]
        t_man = pk.epoch(action[3], "mjd2000")
        pos, vel = self.position(t_man)
        new_vel = list(np.array(vel) + dV)
        mu_central_body, mu_self = self.satellite.mu_central_body, self.satellite.mu_self
        radius, safe_radius = self.satellite.radius, self.satellite.safe_radius
        self.satellite = pk.planet.keplerian(t_man, list(pos), new_vel, mu_central_body,
                                             mu_self, radius, safe_radius)
        self.fuel -= fuel_consumption(dV)
        return

    def position(self, epoch):
        """ Provide SpaceObject position:
            (x, y, z) and (Vx, Vy, Vz), at given epoch.
        """
        pos, vel = self.satellite.eph(epoch)
        return pos, vel

    def get_name(self):
        return self.satellite.name
