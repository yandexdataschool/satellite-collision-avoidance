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
from scipy.stats import norm

PROPAGATION_STEP = 0.00001  # about 1 second


def euclidean_distance(xyz_main, xyz_other, rev_sort=True):
    """ Returns array of (reverse sorted) Euclidean distances between main object and other.
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
    return np.linalg.norm(dV)


def sum_coll_prob(p):
    return (1 - np.prod(1 - np.array(p)))


def collision_prob_normal(xyz_0, xyz_1, sigma):
    """ Returns probability of collision between two objects
        which coordinates are distributed normally
    Args:
        xyz_1, xyz_2 -- np.array([x, y, z]), objects coordinates
        sigma -- float, standard deviation
    ---
    output: float, probability

    """
    # TODO - truncated normal distribution?
    # TODO - multivariate normal distribution?
    probability = 1
    for c0, c1 in zip(xyz_0, xyz_1):
        av = (c0 + c1) / 2.
        integtal = norm.cdf(av, loc=min(c0, c1), scale=sigma)
        probability *= (1 - integtal) / integtal
    return probability


def danger_debr_and_collision_prob(st, debr, threshold, sigma):
    """ Returns danger debris indices and collision probability
    Args:
        st -- np.array shape(1, 3), satellite coordinates
        debr -- np.array shape(n_denris, 3), debris coordinates
        threshold -- float, danger distance
        sigma -- float, standard deviation
    ---
    output: dict {danger_debris: coll_prob}
    """
    # getting danger debris
    crit_dist = euclidean_distance(st, debr, rev_sort=False)
    danger_debris = np.where(crit_dist < threshold)[0]
    # collision probability for any danger debris
    coll_prob = dict()

    for d in danger_debris:
        coll_prob[d] = collision_prob_normal(st[0], debr[d], sigma)

    return coll_prob


class Agent:
    """ Agent implements an agent to communicate with space Environment.
        Agent can make actions to the space environment by taking it's state
        after the last action.
    """

    def __init__(self):
        """"""

    def get_action(self, s):
        """ Provides action  for protected object.
        Args:
            state -- dict where keys:
                'coord' -- dict where:
                    {'st': np.array shape (1, 6)},  satellite xyz and dVx, dVy, dVz coordinates.
                    {'debr': np.array shape (n_items, 6)},  debris xyz and dVx, dVy, dVz coordinates.
                'trajectory_deviation_coef' -- float.
                'epoch' -- pk.epoch, at which time environment state is calculated.
        Returns:
            np.array([dVx, dVy, dVz, pk.epoch, time_to_req])  - vector of deltas for
                protected object, maneuver time and time to request the next action.
        """
        dVx, dVy, dVz = 0, 0, 0
        epoch = s.get("epoch").mjd2000
        time_to_req = 1
        action = np.array([dVx, dVy, dVz, epoch, time_to_req])
        return action


class Environment:
    """ Environment provides the space environment with space objects:
        satellites and debris, in it.
    """

    def __init__(self, protected, debris):
        """
            protected -- SpaceObject, protected space object in Environment.
            debris -- [SpaceObject], list of other space objects.
        """
        self.protected = protected
        self.debris = debris
        self.next_action = pk.epoch(0)
        self.state = dict()
        # critical convergence distance
        # TODO choose true distance
        self.crit_conv_dist = 10000
        n_debris = len(debris)
        # coll_prob = collision probability
        self.coll_prob = dict(
            zip(range(n_debris), np.zeros(n_debris))
        )
        self.buffer_coll_prob = dict()
        self.whole_coll_prob = 0
        self.reward = 0
        self.sigma = 100

    def propagate_forward(self, start, end, prop_step=PROPAGATION_STEP):
        """
        Args:
            start, end -- float, start and end time for propagation as mjd2000.
        """
        for t in np.arange(start, end + prop_step, prop_step):
            epoch = pk.epoch(t, "mjd2000")
            st_pos, st_v = self.protected.position(epoch)
            st = np.hstack((np.array(st_pos), np.array(st_v)))
            st = np.reshape(st, (1, -1))
            n_items = len(self.debris)
            debr = np.zeros((n_items, 6))
            for i in range(n_items):
                pos, v = self.debris[i].position(epoch)
                debr[i] = np.hstack((np.array(pos), np.array(v)))
            debr = np.reshape(debr, (1, -1))

            coord = dict(st=st, debr=debr)
            self.state = dict(
                coord=coord, trajectory_deviation_coef=0.0, epoch=epoch)
            # TODO - check reward update and add ++reward?
            p = self.update_collision_probability()
            self.reward += self.get_reward(p)
            print(self.reward)
            print(self.whole_coll_prob)

        return

    def get_reward(self, collision_probabilities):
        """ Provide total reward from the environment state.
        collision_probabilities: list of probabilities
        ---
        output: float
        """
        # trajectory reward
        traj_reward = -self.state['trajectory_deviation_coef']

        # whole reward
        # TODO - add constants to all reward components
        r = (
            + self.protected.get_fuel()
            + traj_reward
            + np.sum(collision_probabilities))
        return r

    def update_collision_probability(self):
        """ Update the probability of collision on the propagation step. 
        ---
        output -- np.array(current collision probabilities)
        """
        # TODO - log probability
        current_coll_prob = danger_debr_and_collision_prob(
            self.state['coord']['st'][:, :3],
            self.state['coord']['debr'][:, :3],
            self.crit_conv_dist, self.sigma)
        for d in range(len(self.debris)):
            if (d in current_coll_prob.keys()) and (d in self.buffer_coll_prob.keys()):
                # convergence continues
                # update probability buffer
                self.buffer_coll_prob[d] = max(
                    current_coll_prob[d], self.buffer_coll_prob[d])
            elif (d in current_coll_prob.keys()):
                # convergence begins
                # add probability to buffer
                self.buffer_coll_prob[d] = current_coll_prob[d]
            elif (d in self.buffer_coll_prob.keys()):
                # convergence discontinued
                # update environment coll_prob and whole_coll_prob
                # via buffer
                p = [self.coll_prob[d], self.buffer_coll_prob[d]]
                self.coll_prob[d] = sum_coll_prob(p)
                self.whole_coll_prob = sum_coll_prob(coll_prob.values())
        return list(current_coll_prob.values())

    def act(self, action):
        """ Change velocity for protected object.
        Args:
            action -- np.array([dVx, dVy, dVz, pk.epoch, time_to_req]), vector of deltas for
            protected object, maneuver time and time to request the next action.
        """
        self.next_action = pk.epoch(self.state.get(
            "epoch").mjd2000 + action[4], "mjd2000")
        self.protected.maneuver(action[:4])
        return

    def get_next_action(self):
        return self.next_action

    def get_state(self):
        """ Provides environment state. """
        return self.state


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
                t0, elements, mu_central_body, mu_self, radius, safe_radius, name)
        elif param_type == "eph":
            self.satellite = pk.planet.keplerian(params["epoch"],
                                                 params["pos"], params["vel"],
                                                 params["mu_central_body"],
                                                 params["mu_self"],
                                                 params["radius"],
                                                 params["safe_radius"],
                                                 name)
        elif param_type == "osc":
            self.satellite = pk.planet.keplerian(params["epoch"],
                                                 params["elements"],
                                                 params["mu_central_body"],
                                                 params["mu_self"],
                                                 params["radius"],
                                                 params["safe_radius"],
                                                 name)
        else:
            raise ValueError("Unknown initial parameteres type")

    def maneuver(self, action):
        """ Make manoeuvre for the object.
        Args:
            action -- np.array([dVx, dVy, dVz, pk.epoch]), vector
                of deltas for protected object and maneuver time.
         """
        dV = action[:3]
        t_man = pk.epoch(action[3], "mjd2000")
        pos, vel = self.position(t_man)
        new_vel = list(np.array(vel) + dV)

        mu_central_body, mu_self = self.satellite.mu_central_body, self.satellite.mu_self
        radius, safe_radius = self.satellite.radius, self.satellite.safe_radius
        name = self.get_name()

        self.satellite = pk.planet.keplerian(t_man, list(pos), new_vel, mu_central_body,
                                             mu_self, radius, safe_radius, name)
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

    def get_fuel(self):
        return self.fuel
