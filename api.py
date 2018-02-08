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
import pandas as pd
from scipy.stats import norm

MAX_PROPAGATION_STEP = 0.000001  # equal to 0.0864 sc.
MAX_FUEL_CONSUMPTION = 10


def fuel_consumption(dV):
    """ Provide the value of fuel consumption for given velocity delta.

    Args:
        dV (np.array ([dVx, dVy, dVz]) ): vector of satellite velocity delta for maneuver (m/s).

    Returns:
        float: fuel consumption.
    """
    return np.linalg.norm(dV)


def sum_coll_prob(p, axis=0):
    """Summation of probabilities.

    Agrs:
        p (np.array): probabilities.

    Returns:
        result (float or np.array): probabilities sum.

    """
    result = (1 - np.prod(1 - p, axis=axis))
    return result


def rV2ocs(r0, r1, V0, V1):
    """ Convert cartesian coordinate system to orbital coordinate system.

    Args:
        r0, r1 (np.array([x,y,z])): coordinates (meters).
        V0, V1 (np.array([Vx,Vy,Vz])): velocities (m/s).

    Returns:
        floats: dr, dn, db

    """
    dr_vec = r0 - r1
    # orbital coordinate system
    i_r = dr_vec / np.linalg.norm(dr_vec)
    i_b = dr_vec * V0
    i_b = -i_b / np.linalg.norm(i_b)
    i_n = i_r * i_b
    i_n = -i_n / np.linalg.norm(i_n)

    dr = np.dot(dr_vec, i_r)
    dn = np.dot(dr_vec, i_n)
    db = np.dot(dr_vec, i_b)

    return dr, dn, db


def TestProbability(p):
    # Value error
    if ((not isinstance(p, float)) & (not isinstance(p, int))):
        raise TypeError("incorrect probability type: " +
                        str(p) + " " + str(type(p)))
    if ((p > 1) | (p < 0)):
        raise ValueError("incorrect probability value: " + str(p))
    return


def coll_prob_estimation(rV0, rV1, d0=1., d1=1.,
                         sigma=1., approach="normal"):
    """ Returns probability of collision between two objects.

    Args:
        rV0, rV1 (np.array([x, y, z, Vx, Vy, Vz])): objects coordinates (meteres) and velocities(m/s).
        d0, d1 (float, float): objects size (meters).
        sigma (float): standard deviation.
        approach (str): name of approach.
            "Hutor" - Hutorovski approach suited for near circular orbits and
                for convergence at a large angle.
            "normal" - assumption of coordinates are distributed normally
                common empirical approach.

    Returns:
        float: probability.

    Raises:
        ValueError: If any probability has incorrect value.
        TypeError: If any probability has incorrect type.
        ValueError: If approach="Hutor" and V0=np.zeros(3), V1=np.zeros(3).

    """
    probability = 1.
    # TODO - remove

    if approach == "Hutor":
        # # V test
        # if np.array_equal(V0, np.zeros(3)) | np.array_equal(V1, np.zeros(3)):
        #     raise Exception("velocities are required for Hutorovski approach")
        # # cosine of angle between velocities
        # cos_vel_angle = (np.dot(V0, V1)
        #                  / (np.linalg.norm(V0) * np.linalg.norm(V1)))
        # dr, dn, db = rV2ocs(r0, r1, V0, V1)
        # A = (
        #     dr**2 / (4 * sigma**2)
        #     + dn**2 / (2 * sigma**2)
        #     + db**2 / (2 * sigma**2)
        # )
        # k = (
        #     (d0 + d1)**2
        #     / (sigma * (11 + 5 * sigma * cos_vel_angle**2))
        # )
        # probability = k * np.exp(-A)
        pass
    elif approach == "normal":
        # TODO - truncated normal distribution?
        # TODO - multivariate normal distribution?
        for c0, c1 in zip(rV0[:3], rV1[:3]):
            av = (c0 + c1) / 2.
            integtal = norm.cdf(av, loc=min(c0, c1), scale=sigma)
            probability *= (1 - integtal) / integtal
    else:
        raise Exception("unknown probability estimation approach:" + approach)
    TestProbability(probability)
    return probability


def get_conjunction(st_r, debr_r, threshold_d=10000):
    """ Finding potentially dangerous debris, comparing the distance to them with the threshold.

    Args:
        st_r (np.array with shape(1, 3)): satellite position (meters).
        debr_r (np.array with shape(n_denris, 3)): debris positions (meters).
        threshold_d (float): distance danger threshold (meters).

    Returns:
        distances (np.array): Euclidean distances for the each danger debris (meters).
        danger_debris (np.array): danger debris indicies.

    TODO:
        * add distance units and true threshold_d.
    """
    distances = np.linalg.norm(debr_r - st_r, axis=1)
    danger_debris = np.where(distances <= threshold_d)[0]
    distances = distances[danger_debris]
    return distances, danger_debris


class Agent:

    """ Agent implements an agent to communicate with space Environment.

    Agent can make actions to the space environment by taking it's state
    after the last action.

    """

    def __init__(self):
        """"""

    def get_action(self, state):
        """ Provides action for protected object.

        Args:
            state (dict): environment state
                {'coord' (dict):
                    {'st' (np.array with shape (1, 6)): satellite r and Vx, Vy, Vz coordinates (meters).
                     'debr' (np.array with shape (n_items, 6)): debris r and Vx, Vy, Vz coordinates (meters).}
                'trajectory_deviation_coef' (float).
                'epoch' (pk.epoch): at which time environment state is calculated.
                'fuel' (float): current remaining fuel in protected SpaceObject. }
        Returns:
            action (np.array([dVx, dVy, dVz, pk.epoch, time_to_req])):
                vector of deltas for protected object (m/s),
                maneuver time (mjd2000) and step in time
                when to request the next action (mjd2000).
        """
        dVx, dVy, dVz = 0, 0, 0  # m/s
        epoch = state["epoch"].mjd2000
        time_to_req = 0.001  # mjd2000
        action = np.array([dVx, dVy, dVz, epoch, time_to_req])

        return action


class Environment:
    """ Environment provides the space environment with space objects: satellites and debris, in it."""

    def __init__(self, protected, debris, start_time):
        """
        Args:
            protected (SpaceObject): protected space object in Environment.
            debris ([SpaceObject, ]): list of other space objects.
            start (pk.epoch): initial time of the environment.

        """
        self.protected = protected
        self.debris = debris
        self.next_action = pk.epoch(0, "mjd2000")
        self.state = dict(epoch=start_time, fuel=self.protected.get_fuel())
        self.n_debris = len(debris)
        self.st_d = 1.  #: Satellite size (meters)
        self.debr_d = np.ones(self.n_debris)  #: Debris sizes (meters)
        self.crit_distance = 50  #: Critical convergence distance (meters)
        # TODO choose true sigma
        self.sigma = 50000  #: Coordinates uncertainly (meters)

        self.min_distances_in_current_conjunction = np.full(
            (self.n_debris), np.nan)  # np.nan if not in conjunction.
        self.state_for_min_distances_in_current_conjunction = dict()
        self.danger_debris_in_current_conjunction = np.array([])

        self.collision_probability_prior_to_current_conjunction = np.zeros(
            self.n_debris)
        self.total_collision_probability_array = np.zeros(self.n_debris)
        self.total_collision_probability = 0.

        self.whole_trajectory_deviation = 0.
        self.reward = 0.

    def propagate_forward(self, end_time, auto_update_reward_probability=False):
        """ Forward step.

        Args:
            end_time (float): end time for propagation as mjd2000.
            auto_update_reward (bool):
                True if reward and total probability are updated at the each step,
                False if reward and total probability are updated only from outside.

        Raises:
            ValueError: if end_time is less then current time of the environment.
            Exception: if step in propagation_grid is less then MAX_PROPAGATION_STEP.

        """
        curr_time = self.state["epoch"].mjd2000
        if end_time == curr_time:
            return
        elif end_time < curr_time:
            raise ValueError(
                "end_time should be greater or equal to current time")

        # Choose number of steps in linspace, s.t.
        # restep is less then MAX_PROPAGATION_STEP.
        number_of_time_steps_plus_one = int(np.ceil(
            (end_time - curr_time) / MAX_PROPAGATION_STEP) + 1)

        propagation_grid, retstep = np.linspace(
            curr_time, end_time, number_of_time_steps_plus_one, retstep=True)

        if retstep > MAX_PROPAGATION_STEP:
            raise Exception(
                "Step in propagation grid should be <= MAX_PROPAGATION_STEP")

        for t in propagation_grid:
            epoch = pk.epoch(t, "mjd2000")
            st_pos, st_v = self.protected.position(epoch)
            st = np.hstack((np.array(st_pos), np.array(st_v)))[np.newaxis, ...]
            n_items = len(self.debris)
            debr = np.zeros((n_items, 6))
            for i in range(n_items):
                pos, v = self.debris[i].position(epoch)
                debr[i] = np.array(pos + v)

            coord = dict(st=st, debr=debr)
            trajectory_deviation_coef = 0.0
            self.whole_trajectory_deviation += trajectory_deviation_coef
            self.state = dict(
                coord=coord, trajectory_deviation_coef=trajectory_deviation_coef,
                epoch=epoch, fuel=self.protected.get_fuel()
            )
            self.update_distances_and_probabilities_prior_to_current_conjunction()
            if auto_update_reward_probability:
                self.update_collision_probability()
                self.update_reward()

        return

    def update_distances_and_probabilities_prior_to_current_conjunction(self):
        """ Update the distances and collision probabilities prior to current conjunction.

        """
        new_distances_in_current_conjunction, new_danger_debris = get_conjunction(
            self.state['coord']['st'][:, :3],
            self.state['coord']['debr'][:, :3],
            self.crit_distance
        )
        end_cojunction_debris = np.setdiff1d(
            self.danger_debris_in_current_conjunction,
            new_danger_debris
        )  # the debris for which the conjunction has now ceased.
        new_cojunction_debris = np.setdiff1d(
            new_danger_debris,
            self.danger_debris_in_current_conjunction
        )
        self.danger_debris_in_current_conjunction = new_danger_debris

        for_update_debris = new_danger_debris[np.logical_not(
            new_distances_in_current_conjunction >
            self.min_distances_in_current_conjunction[new_danger_debris]
        )]
        """
        np.array: Debris (indices) which:
            have been in current conjunction and the distance to which has decreased,
            have not been not in conjunction, but now are taken into account.
        """

        # Update min distances and states for danger debris
        # in current conjunction.
        if for_update_debris.size:
            self.min_distances_in_current_conjunction[
                for_update_debris] = new_distances_in_current_conjunction[for_update_debris]
            for d in for_update_debris:
                self.state_for_min_distances_in_current_conjunction[d] = np.vstack((
                    self.state['coord']['st'][0, :],
                    self.state['coord']['debr'][d, :]
                ))

        # Update collision probability prior to current conjunction.
        if end_cojunction_debris.size:
            coll_prob = np.array([
                coll_prob_estimation(
                    self.state_for_min_distances_in_current_conjunction[d][0],
                    self.state_for_min_distances_in_current_conjunction[d][1],
                    self.st_d, self.debr_d[d], self.sigma
                )
                for d in end_cojunction_debris
            ])
            self.collision_probability_prior_to_current_conjunction[
                end_cojunction_debris]
            self.collision_probability_prior_to_current_conjunction[end_cojunction_debris] = sum_coll_prob(
                np.vstack([
                    self.collision_probability_prior_to_current_conjunction[
                        end_cojunction_debris],
                    coll_prob
                ])
            )
            self.min_distances_in_current_conjunction[
                end_cojunction_debris] = np.nan
            for d in end_cojunction_debris:
                del self.state_for_min_distances_in_current_conjunction[d]

    def update_collision_probability(self):
        """ Update the total collision probability.

        """
        if self.danger_debris_in_current_conjunction.size:
            collision_probability_in_current_conjunction = np.array([
                coll_prob_estimation(
                    self.state_for_min_distances_in_current_conjunction[d][0],
                    self.state_for_min_distances_in_current_conjunction[d][1],
                    self.st_d, self.debr_d[d], self.sigma
                )
                for d in self.danger_debris_in_current_conjunction
            ])
            self.total_collision_probability_array[self.danger_debris_in_current_conjunction] = sum_coll_prob(
                np.vstack([
                    self.collision_probability_prior_to_current_conjunction[
                        self.danger_debris_in_current_conjunction],
                    collision_probability_in_current_conjunction
                ])
            )
            self.total_collision_probability = sum_coll_prob(
                self.total_collision_probability_array
            )
        return

    def update_reward(self, coll_prob_C=10000., traj_C=1., fuel_C=1.,
                      danger_prob=10e-4):
        """ Update total reward from the environment state.

        Args:
            coll_prob_C, traj_C, fuel_C (float): constants for the singnificance regulation of reward components.
            danger_prob (float): the threshold below which the probability is negligible.

        """
        # reward components
        coll_prob = self.total_collision_probability
        ELU = lambda x: x if (x >= 0) else (1 * (np.exp(x) - 1))
        # collision probability reward - some kind of ELU function
        # of collision probability
        coll_prob_r = -(ELU((coll_prob - danger_prob) * coll_prob_C) + 1)
        traj_r = - traj_C * self.whole_trajectory_deviation
        fuel_r = fuel_C * self.protected.get_fuel()

        # whole reward
        # TODO - add weights to all reward components
        self.reward = (coll_prob_r + traj_r + fuel_r)
        return

    def act(self, action):
        """ Change velocity for protected object.
        Args:
            action (np.array([dVx, dVy, dVz, pk.epoch, time_to_req])):
                vector of velocity deltas for protected object (m/s),
                maneuver time (mjd2000) and step in time
                when to request the next action (mjd2000).
        """
        self.next_action = pk.epoch(
            self.state["epoch"].mjd2000 + float(action[4]), "mjd2000")
        error, fuel_cons = self.protected.maneuver(action[:4])
        if not error:
            self.state["fuel"] -= fuel_cons
        return error

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
            name (str): name of satellite or a space debris.
            param_type (str): initial parameteres type. Could be:
                    "tle": for TLE object,
                    "eph": ephemerides, initialize with position and velocity state vectors,
                    "osc": osculating elements, initialize with 6 orbital parameteres.
            params (dict): dictionary of space object coordinates.
                "fuel" (float): initial fuel capacity.

                for "tle" type:
                    "tle1" (str): tle line1.
                     "tle2" (str): tle line2.

                for "eph" type:
                    "pos" ([x, y, z]): position (cartesian, meters).
                    "vel" ([Vx, Vy, Vz]): velocity (cartesian, m/s).
                    "epoch" (pykep.epoch): start time (mjd2000).
                    "mu_central_body" (float): gravity parameter of the
                        central body (m^2/s^3).
                    "mu_self"(float): gravity parameter of the planet (m^2/s^3).
                    "radius" (float): body radius (meters).
                    "safe_radius" (float): mimimual radius that is safe during
                        a fly-by of the planet (meters).

                for "osc" type:
                    "elements" (tuple): containing 6 orbital osculating elements.
                    "epoch" (pykep.epoch): start time.
                    "mu_central_body", "mu_self", "radius", "safe_radius" (float): same, as in "eph" type.
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
            action (np.array([dVx, dVy, dVz, pk.epoch])): vector of velocity
                deltas for protected object and maneuver time (m/s).
        Returns:
            (string): empty string if action is successfully made by satellite,
                error message otherwise.
            fuel_cons (float): fuel consumption of the provided action.
         """
        dV = action[:3]
        fuel_cons = fuel_consumption(dV)
        if fuel_cons > MAX_FUEL_CONSUMPTION:
            return "requested action exceeds the fuel consumption limit.", 0
        elif fuel_cons > self.fuel:
            return "requested action exceeds fuel amount in the satellite.", 0

        t_man = pk.epoch(float(action[3]), "mjd2000")
        pos, vel = self.position(t_man)
        new_vel = list(np.array(vel) + dV)

        mu_central_body, mu_self = self.satellite.mu_central_body, self.satellite.mu_self
        radius, safe_radius = self.satellite.radius, self.satellite.safe_radius
        name = self.get_name()

        self.satellite = pk.planet.keplerian(t_man, list(pos), new_vel, mu_central_body,
                                             mu_self, radius, safe_radius, name)
        self.fuel -= fuel_cons
        return "", fuel_cons

    def position(self, epoch):
        """ Provide SpaceObject position at given epoch:
        Args:
            epoch (pk.epoch): at what time to calculate position.
        Returns
            pos (tuple): position x, y, z (meters).
            vel (tuple): velocity Vx, Vy, Vz (m/s).
        """
        pos, vel = self.satellite.eph(epoch)
        return pos, vel

    def get_name(self):
        return self.satellite.name

    def get_fuel(self):
        return self.fuel
