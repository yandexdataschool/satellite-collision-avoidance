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


class CollProbEstimation:

    """ Estimate probability of collision between two objects. """

    def __init__(self):
        """"""

    def ChenBai_approach(self, rV1, rV2,
                         cs_r1=100, cs_r2=0.1,
                         sigma_1N=50, sigma_1T=50, sigma_1W=50,
                         sigma_2N=300, sigma_2T=300, sigma_2W=300):
        """ Returns probability of collision between two objects.

        Lei Chen, Xian-Zong Bai, Yan-Gang Liang, Ke-Bo Li
        closest approach between objects from "Orbital Data Applications for Space Objects".

        Args:
            rV1, rV2 (np.array([x, y, z, Vx, Vy, Vz])): objects coordinates (meters) and velocities (m/s).
            cs_r1, cs_r2 (float): objects cross-section radii (meters).
            sigma_N, sigma_T, sigma_W (float): objects positional error standard deviations
                in normal, tangential, cross-track direction (meters).

        Note:
            The default args imply the first object is protected satellite
            and other is debris.

        Returns:
            float: probability.

        Raises:
            ValueError: If any probability has incorrect value.
            TypeError: If any probability has incorrect type.

        """

        r1_vec = rV1[:3]
        r2_vec = rV2[:3]
        v1_vec = rV1[3:]
        v2_vec = rV2[3:]

        r1 = np.linalg.norm(r1_vec)
        r2 = np.linalg.norm(r2_vec)
        v1 = np.linalg.norm(v1_vec)
        v2 = np.linalg.norm(v2_vec)
        dr0_vec = r2_vec - r1_vec
        dr0 = np.linalg.norm(dr0_vec)

        rA = cs_r1 + cs_r2  #: Combined collision cross-section radii
        miss_distance = dr0
        nu = v2 / v1
        #: The angle w between two velocities
        if np.array_equal(v1_vec, v2_vec):
            psi = 0
        else:
            psi = np.arccos(np.dot(v1_vec, v2_vec) / (v1 * v2))
        epsilon = 10e-7
        temp = v1 * v2 * np.sin(psi)**2
        if temp == 0:
            temp += epsilon
        t1_min = (nu * np.dot(dr0_vec, v1_vec) - np.cos(psi) *
                  np.dot(dr0_vec, v2_vec)) / temp
        t2_min = (np.cos(psi) * np.dot(dr0_vec, v1_vec) -
                  np.dot(dr0_vec, v2_vec) / nu) / temp
        #: Crossing time difference of the common perpendicular line (sec)
        dt = abs(t2_min - t1_min)
        #: The relative position vector at closest approach between orbits
        dr_min_vec = dr0_vec + v2_vec * t2_min - v1_vec * t1_min
        #: The closest approach distance between two straight line trajectories
        dr_min = np.linalg.norm(dr_min_vec)
        # Probability.
        temp = 1 + nu**2 - 2 * nu * np.cos(psi)
        if temp == 0:
            temp += epsilon
        mu_x = dr_min
        mu_y = v2 * np.sin(psi) * dt / temp**0.5
        sigma_x_square = sigma_1N**2 + sigma_2N**2
        sigma_y_square = ((sigma_1T * nu * np.sin(psi))**2
                          + ((1 - nu * np.cos(psi)) * sigma_1W)**2
                          + (sigma_2T * np.sin(psi))**2
                          + ((nu - np.cos(psi)) * sigma_2W)**2
                          ) / temp
        if sigma_x_square == 0:
            sigma_x_square += epsilon
        if sigma_y_square == 0:
            sigma_y_square += epsilon
        probability = np.exp(
            -0.5 * (mu_x**2 / sigma_x_square + mu_y**2 / sigma_y_square)
        ) * (1 - np.exp(-rA**2 / (2 * (sigma_x_square * sigma_y_square)**0.5)))
        return probability

    def norm_approach(self, rV1, rV2, sigma=50):
        """ Returns probability of collision between two objects.

        Args:
            rV1, rV2 (np.array([x, y, z, Vx, Vy, Vz])): objects coordinates (meteres) and velocities (m/s).
            d1, d2 (float, float): objects size (meters).
            sigma (float): standard deviation.

        Returns:
            float: probability.

        Raises:
            ValueError: If any probability has incorrect value.
            TypeError: If any probability has incorrect type.

        """
        probability = 1.
        for c1, c2 in zip(rV1[:3], rV2[:3]):
            av = (c1 + c2) / 2.
            integtal = norm.cdf(av, loc=min(c1, c2), scale=sigma)
            probability *= (1 - integtal) / integtal

        return probability


def get_dangerous_debris(st_r, debr_r, crit_distance):
    """ Finding potentially dangerous debris, comparing the distance to them with the threshold.

    Args:
        st_r (np.array with shape(1, 3)): satellite position (meters).
        debr_r (np.array with shape(n_denris, 3)): debris positions (meters).
        crit_distance (float): dangerous distance threshold (meters).

    Returns:
        dangerous_debris (np.array): dangerous debris indicies.
        distances (np.array): Euclidean distances for the each dangerous debris (meters).

    TODO:
        * add distance units and true crit_distance.
    """
    distances = np.linalg.norm(debr_r - st_r, axis=1)
    dangerous_debris = np.where(distances <= crit_distance)[0]
    distances = distances[dangerous_debris]
    return dangerous_debris, distances


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
        self.protected_r = protected.get_radius()
        self.init_fuel = self.protected.get_fuel()
        self.debris_r = np.array([d.get_radius() for d in debris])
        self.next_action = pk.epoch(0, "mjd2000")
        self.state = dict(epoch=start_time, fuel=self.protected.get_fuel())
        self.n_debris = len(debris)
        self.crit_distance = 10000  #: Critical convergence distance (meters)
        self.collision_probability_estimator = CollProbEstimation()

        self.min_distances_in_current_conjunction = np.full(
            (self.n_debris), np.nan)  # np.nan if not in conjunction.
        self.state_for_min_distances_in_current_conjunction = dict()
        self.dangerous_debris_in_current_conjunction = np.array([])

        self.collision_probability_prior_to_current_conjunction = np.zeros(
            self.n_debris)
        self.total_collision_probability_array = np.zeros(self.n_debris)
        self.total_collision_probability = 0.

        self.whole_trajectory_deviation = 0.
        self.reward = 0.
        # : epoch: Last reward and collision probability update
        self.last_r_p_update = None
        # : int: number of propagate forward iterations
        # since last update collision probability and reward.
        self.pf_iterations_since_update = 0

    def propagate_forward(self, end_time, update_r_p_step=20):
        """ Forward step.

        Args:
            end_time (float): end time for propagation as mjd2000.
            update_r_p_step (int): update reward and probability step.

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

        self.pf_iterations_since_update += 1

        if self.pf_iterations_since_update == update_r_p_step:
            self.get_reward()

    def update_distances_and_probabilities_prior_to_current_conjunction(self):
        """ Update the distances and collision probabilities prior to current conjunction.

        """
        new_curr_dangerous_debris, new_curr_dangerous_distances = get_dangerous_debris(
            self.state['coord']['st'][:, :3],
            self.state['coord']['debr'][:, :3],
            self.crit_distance
        )
        end_cojunction_debris = np.setdiff1d(
            self.dangerous_debris_in_current_conjunction,
            new_curr_dangerous_debris
        )  # the debris with which the conjunction has now ceased.
        begin_cojunction_debris = np.setdiff1d(
            new_curr_dangerous_debris,
            self.dangerous_debris_in_current_conjunction
        )  # the debris with which the conjunction begins.

        self.dangerous_debris_in_current_conjunction = new_curr_dangerous_debris

        for_update_debris = new_curr_dangerous_debris[np.logical_not(
            new_curr_dangerous_distances >
            self.min_distances_in_current_conjunction[
                new_curr_dangerous_debris]
        )]
        """
        np.array: Debris (indices) which:
            have been in current conjunction and the distance to which has decreased,
            have not been not in conjunction, but now are taken into account.
        """

        # Update min distances and states for dangerous debris
        # in current conjunction.
        if for_update_debris.size:
            self.min_distances_in_current_conjunction[
                for_update_debris] = new_curr_dangerous_distances[for_update_debris]
            for d in for_update_debris:
                self.state_for_min_distances_in_current_conjunction[d] = np.vstack((
                    self.state['coord']['st'][0, :],
                    self.state['coord']['debr'][d, :]
                ))

        # Update collision probability prior to current conjunction.
        if end_cojunction_debris.size:
            coll_prob = np.array([
                self.collision_probability_estimator.ChenBai_approach(
                    self.state_for_min_distances_in_current_conjunction[d][0],
                    self.state_for_min_distances_in_current_conjunction[d][1],
                    self.protected_r, self.debris_r[d]
                )
                for d in end_cojunction_debris
            ])
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

    def get_collision_probability(self):
        """ Update and return total collision probability."""
        if self.dangerous_debris_in_current_conjunction.size:
            collision_probability_in_current_conjunction = np.array([
                self.collision_probability_estimator.ChenBai_approach(
                    self.state_for_min_distances_in_current_conjunction[
                        d][0],
                    self.state_for_min_distances_in_current_conjunction[
                        d][1],
                    self.protected_r, self.debris_r[d]
                )
                for d in self.dangerous_debris_in_current_conjunction
            ])
            self.total_collision_probability_array[self.dangerous_debris_in_current_conjunction] = sum_coll_prob(
                np.vstack([
                    self.collision_probability_prior_to_current_conjunction[
                        self.dangerous_debris_in_current_conjunction],
                    collision_probability_in_current_conjunction
                ])
            )
        else:
            self.total_collision_probability_array = self.collision_probability_prior_to_current_conjunction

        self.total_collision_probability = sum_coll_prob(
            self.total_collision_probability_array
        )
        return self.total_collision_probability

    def get_reward(self, coll_prob_C=10000., traj_C=1., fuel_C=1.,
                   dangerous_prob=10e-4):
        """ Update and return total reward from the environment state.

        Args:
            coll_prob_C, traj_C, fuel_C (float): constants for the singnificance regulation of reward components.
            dangerous_prob (float): the threshold below which the probability is negligible.

        """
        # reward components
        coll_prob = self.get_collision_probability()
        ELU = lambda x: x if (x >= 0) else (1 * (np.exp(x) - 1))
        # collision probability reward - some kind of ELU function
        # of collision probability
        coll_prob_r = -(ELU((coll_prob - dangerous_prob) * coll_prob_C) + 1)
        traj_r = - traj_C * self.whole_trajectory_deviation
        fuel_r = - fuel_C * (self.init_fuel - self.protected.get_fuel())

        # total reward
        # TODO - add weights to all reward components
        self.reward = (coll_prob_r + traj_r + fuel_r)
        self.last_r_p_update = self.state["epoch"]
        self.pf_iterations_since_update = 0
        return self.reward

    def act(self, action):
        """ Change velocity for protected object.
        Args:
            action (np.array([dVx, dVy, dVz, time_to_req])):
                vector of velocity deltas for protected object (m/s),
                step in time when to request the next action (mjd2000).
        """
        self.next_action = pk.epoch(
            self.state["epoch"].mjd2000 + float(action[3]), "mjd2000")
        error, fuel_cons = self.protected.maneuver(
            action[:3], self.state["epoch"])
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

    def maneuver(self, action, t_man):
        """ Make manoeuvre for the object.
        Args:
            action (np.array([dVx, dVy, dVz])): vector of velocity
                deltas for protected object (m/s).
            t_man (pk.epoch): maneuver time.
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

    def get_radius(self):
        return self.satellite.radius
