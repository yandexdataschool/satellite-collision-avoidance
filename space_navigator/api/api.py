# Module api provides functionality for simulation of
# object movement in space , communication with the space environment
# by the Agent and state/reward exchanging.
#
# In the first implementation wi will have only one protected
# object. All other objects will be treated as space debris.
# As a first, we will observe only ideal satellite's trajectories,
# so that we can describe any object location at time t after the
# simulation has been started.

import numpy as np
import pykep as pk
from copy import copy

from .api_utils import fuel_consumption, sum_coll_prob, get_dangerous_debris
from ..collision import CollProbEstimator

MAX_PROPAGATION_STEP = 0.000001  # equal to 0.0864 sc.
MAX_FUEL_CONSUMPTION = 10


class Environment:
    """ Environment provides the space environment with space objects: satellites and debris, in it."""

    def __init__(self, protected, debris, start_time, end_time):
        """
        Args:
            protected (SpaceObject): protected space object in Environment.
            debris ([SpaceObject, ]): list of other space objects.
            start (pk.epoch): initial time of the environment.

        """
        self.init_params = dict(protected=copy(protected), debris=copy(
            debris), start_time=start_time, end_time=end_time)

        self.protected = protected
        self.debris = debris

        self.protected_r = self.protected.get_radius()
        self.init_fuel = self.protected.get_fuel()
        self.init_orbital_elements = self.protected.get_orbital_elements()
        self.n_debris = len(debris)
        self.debris_r = np.array([d.get_radius() for d in debris])

        self.next_action = pk.epoch(0, "mjd2000")
        self.state = dict(epoch=start_time, fuel=self.protected.get_fuel())

        self.crit_distance = 10000  #: Critical convergence distance (meters)
        self.min_distances_in_current_conjunction = np.full(
            (self.n_debris), np.nan)  # np.nan if not in conjunction.
        self.state_for_min_distances_in_current_conjunction = dict()
        self.dangerous_debris_in_current_conjunction = np.array([])

        self.collision_probability_estimator = CollProbEstimator()
        self.collision_probability_prior_to_current_conjunction = np.zeros(
            self.n_debris)
        self.total_collision_probability_array = np.zeros(self.n_debris)
        self.total_collision_probability = 0

        self.reward_components = (0, 0, 0)
        self.reward = 0
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
            self.state = dict(
                coord=coord, epoch=epoch, fuel=self.protected.get_fuel()
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

    def get_trajectory_deviation(self, singnificance=(0.01, 1, 1, 1, 1, 0)):
        """Returns trajectory deviation from init the trajectory.

        Note:
            six osculating keplerian elements (a,e,i,W,w,M):
                a (semi-major axis): meters,
                e (eccentricity): greater than 0,
                i (inclination), W (Longitude of the ascending node): radians,
                w (Argument of periapsis), M (mean anomaly): radians.

        Args:
            singnificance (tuple): array of multipliers for orbital parameter differences.

        Return:
            deviation (float): trajectory deviation from init the trajectory.
        """
        diff = np.abs(
            np.array(self.protected.get_orbital_elements()) - np.array(self.init_orbital_elements))
        deviation = np.sum(diff * np.array(singnificance))
        return deviation

    def get_reward(self, coll_prob_C=10000, traj_C=1, fuel_C=1,
                   dangerous_prob=10e-4, get_components=False):
        """ Update and return total reward from the environment state.

        Args:
            coll_prob_C, traj_C, fuel_C (float): constants for the singnificance regulation of reward components.
            dangerous_prob (float): the threshold below which the probability is negligible.

        """
        # reward components
        coll_prob = self.get_collision_probability()
        trajectory_deviation = self.get_trajectory_deviation()
        ELU = lambda x: x if (x >= 0) else (1 * (np.exp(x) - 1))
        # collision probability reward - some kind of ELU function
        # of collision probability
        coll_prob_r = -(ELU((coll_prob - dangerous_prob) * coll_prob_C) + 1)
        fuel_r = - fuel_C * (self.init_fuel - self.protected.get_fuel())
        traj_r = - traj_C * trajectory_deviation

        # total reward
        # TODO - add weights to all reward components
        self.reward_components = (coll_prob_r, fuel_r, traj_r)
        self.reward = (coll_prob_r + fuel_r + traj_r)
        self.last_r_p_update = self.state["epoch"]
        self.pf_iterations_since_update = 0
        if get_components:
            return self.reward, self.reward_components
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

    def get_fuel_consumption(self):
        return self.init_fuel - self.protected.get_fuel()

    def reset(self):
        """ Return Environment to initial state. """
        self.__init__(self.init_params['protected'], self.init_params['debris'],
                      self.init_params['start_time'], self.init_params['end_time'])
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

    def get_orbital_elements(self):
        return self.satellite.orbital_elements
