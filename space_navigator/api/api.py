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

from .api_utils import (
    fuel_consumption, sum_coll_prob, reward_func,
    lower_estimate_of_time_to_conjunction, correct_angular_deviations,
)
from ..collision import CollProbEstimator

MAX_FUEL_CONSUMPTION = 10


class Environment:
    """ Environment provides the space environment with space objects: satellites and debris, in it."""

    def __init__(self, protected, debris, start_time, end_time,
                 coll_prob_thr=1e-4, fuel_cons_thr=10,
                 traj_dev_thr=(100, 0.01, 0.01, 0.01, 0.01, None),
                 target_osculating_elements=None):
        """
        Args:
            protected (SpaceObject): protected space object in Environment.
            debris ([SpaceObject, ]): list of other space objects.
            start_time (pk.epoch): initial time of the environment.
            end_time (pk.epoch): end time of the environment.
            coll_prob_thr (float): collision propalility threshold.
            fuel_cons_thr (float): fuel consumption threshold.
            traj_dev_thr (tuple): trajectory deviation thresholds,
                represents the thresholds of differencies of
                six osculating keplerian elements (a,e,i,W,w,M):
                    a (semi-major axis): meters,
                    e (eccentricity): 0 < e < 1,
                    i (inclination), W (Longitude of the ascending node): radians,
                    w (Argument of periapsis), M (mean anomaly): radians.
            target_osculating_elements (tuple):
                six osculating keplerian elements (a,e,i,W,w,M) of target orbit of protected object,
                is equal to protected object osculating elements if None.

        Note:
            Reward component is not taken into account, if threshold is None.

        Raises:
            ValueError: If eccentricity (traj_dev_thr[1]) is not in interval (0, 1).

        """
        if not (0 < traj_dev_thr[1] < 1):
            raise ValueError(f"eccentricity: {traj_dev_thr[1]}, should be in interval (0, 1).")

        self.init_params = dict(
            protected=copy(protected), debris=copy(debris), start_time=start_time,
            end_time=end_time, coll_prob_thr=coll_prob_thr, fuel_cons_thr=fuel_cons_thr,
            traj_dev_thr=traj_dev_thr, target_osculating_elements=target_osculating_elements,
        )

        self.protected = protected
        self.debris = debris

        self.protected_r = protected.get_radius()
        self.init_fuel = protected.get_fuel()
        if target_osculating_elements is None:
            self.init_osculating_elements = np.array(
                self.protected.osculating_elements(self.init_params["start_time"]))
        else:
            self.init_osculating_elements = target_osculating_elements
        self.trajectory_deviation = None
        self.n_debris = len(debris)
        self.debris_r = np.array([d.get_radius() for d in debris])

        self.next_action = pk.epoch(0, "mjd2000")

        # TODO - calculate crit_distance using probability
        self.crit_distance = 2000  #: Critical convergence distance (meters)
        self.min_distances_in_current_conjunction = np.full(
            (self.n_debris), np.nan)  # np.nan if not in conjunction.
        # np.array, each row contains: (st coord, debr coord, epoch mjd2000)
        self.state_for_min_distances_in_current_conjunction = np.full(
            (self.n_debris, 13), np.nan)
        self.dangerous_debris_in_current_conjunction = np.array([])

        self.collision_probability_estimator = CollProbEstimator.ChenBai_approach
        self.collision_probability_prior_to_current_conjunction = np.zeros(
            self.n_debris)
        self.total_collision_probability_arr = np.zeros(self.n_debris)
        self.total_collision_probability = None

        self.coll_prob_thr = coll_prob_thr
        self.fuel_cons_thr = fuel_cons_thr
        self.traj_dev_thr = traj_dev_thr
        self._reward_thr = np.concatenate(
            ([coll_prob_thr], [fuel_cons_thr], traj_dev_thr)
        ).astype(np.float)

        self.reward_components = None
        self.reward = None

        # initiate state with initial positions
        st, debr = self.get_coords_by_epoch(start_time)
        coord = dict(st=st, debr=debr)
        self.state = dict(
            coord=coord, epoch=start_time, fuel=self.protected.get_fuel())

        self.collision_data = []

        self._update_all_reward_components(zero_update=True)

    def propagate_forward(self, end_time, step=10e-6, each_step_propagation=False):
        """ Forward propagation.

        Args:
            end_time (float): end time for propagation as mjd2000.
            step (float): propagation time step.
                By default is equal to 0.000001 (0.0864 sc.).
            each_step_propagation (bool): whether propagate for each step
                or skip the steps using a lower estimation of the time to conjunction.

        Raises:
            ValueError: if end_time is less then current time of the environment.
            Exception: if step in propagation_grid is less then step.

        """
        curr_time = self.state["epoch"].mjd2000
        if end_time == curr_time:
            return
        elif end_time < curr_time:
            raise ValueError(
                "end_time should be greater or equal to current time")

        # Choose number of steps in linspace, s.t.
        # restep is less then step.
        n_time_steps_plus_one = int(np.ceil(
            (end_time - curr_time) / step) + 1)

        propagation_grid, retstep = np.linspace(
            curr_time, end_time, n_time_steps_plus_one, retstep=True)

        if retstep > step:
            raise Exception(
                "Step in propagation grid should be <= step")

        s = 0
        while s < n_time_steps_plus_one:
            t = propagation_grid[s]
            epoch = pk.epoch(t, "mjd2000")
            st, debr = self.get_coords_by_epoch(epoch)
            coord = dict(st=st, debr=debr)
            self.state = dict(
                coord=coord, epoch=epoch, fuel=self.protected.get_fuel()
            )
            # dangerous debris, distances
            # and estimation of time to collision with closest debris
            debr, dist, time_to_collision = lower_estimate_of_time_to_conjunction(
                self.state['coord']['st'], self.state['coord']['debr'], self.crit_distance)
            time_to_collision_is_finite = np.isfinite(time_to_collision)
            if time_to_collision_is_finite:
                self._update_distances_and_probabilities_prior_to_current_conjunction(
                    debr, dist)
            # calculation of the number of steps forward
            if each_step_propagation:
                s += 1
            else:
                if time_to_collision_is_finite:
                    n_steps = max(1, int(time_to_collision / retstep))
                    n_steps = min(n_time_steps_plus_one - s, n_steps)
                    s += n_steps
                else:
                    s = n_time_steps_plus_one - 1 if s < n_time_steps_plus_one - 1 else s + 1

        self._update_all_reward_components()

    def _update_distances_and_probabilities_prior_to_current_conjunction(self, debr, dist):
        """ Update the distances and collision probabilities prior to current conjunction."""
        new_curr_dangerous_debris = np.copy(debr)
        new_curr_dangerous_distances = np.copy(dist)
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
                for_update_debris] = new_curr_dangerous_distances
            self.state_for_min_distances_in_current_conjunction[
                for_update_debris, : 6] = self.state['coord']['st'][0, :]
            self.state_for_min_distances_in_current_conjunction[
                for_update_debris, 6: 12] = self.state['coord']['debr'][for_update_debris, :]
            self.state_for_min_distances_in_current_conjunction[
                for_update_debris, 12] = self.state["epoch"].mjd2000

        # Update collision probability prior to current conjunction.
        if end_cojunction_debris.size:
            coll_prob = []
            for d in end_cojunction_debris:
                p = self.collision_probability_estimator(
                    self.state_for_min_distances_in_current_conjunction[
                        d, :6],
                    self.state_for_min_distances_in_current_conjunction[
                        d, 6:12],
                    self.protected_r, self.debris_r[d]
                )
                self.collision_data.append({
                    "probability": p,
                    "distance": self.min_distances_in_current_conjunction[d],
                    "epoch": self.state_for_min_distances_in_current_conjunction[d, 12],
                    "debris_name": self.debris[d].get_name(),
                    "debris_id": d,
                })
                coll_prob.append(p)
            self.collision_probability_prior_to_current_conjunction[end_cojunction_debris] = sum_coll_prob(
                np.vstack([
                    self.collision_probability_prior_to_current_conjunction[
                        end_cojunction_debris],
                    np.array(coll_prob)
                ])
            )
            self.min_distances_in_current_conjunction[
                end_cojunction_debris] = np.nan
            self.state_for_min_distances_in_current_conjunction[
                end_cojunction_debris] = np.nan

    def _update_total_collision_probability(self):
        """ Update total collision probability."""
        if self.dangerous_debris_in_current_conjunction.size:
            collision_probability_in_current_conjunction = np.array([
                self.collision_probability_estimator(
                    self.state_for_min_distances_in_current_conjunction[
                        d, :6],
                    self.state_for_min_distances_in_current_conjunction[
                        d, 6:12],
                    self.protected_r, self.debris_r[d]
                )
                for d in self.dangerous_debris_in_current_conjunction
            ])
            self.total_collision_probability_arr[self.dangerous_debris_in_current_conjunction] = sum_coll_prob(
                np.vstack([
                    self.collision_probability_prior_to_current_conjunction[
                        self.dangerous_debris_in_current_conjunction],
                    collision_probability_in_current_conjunction
                ])
            )
        else:
            self.total_collision_probability_arr = self.collision_probability_prior_to_current_conjunction

        self.total_collision_probability = sum_coll_prob(
            self.total_collision_probability_arr
        )

    def _update_trajectory_deviation(self, zero_update=False):
        """Update trajectory deviation from init the trajectory.

        Args:
            zero_update (bool): no deviation at zero update.

        """
        if zero_update:
            deviation = np.zeros((6))
        else:
            initial = self.init_osculating_elements
            current = np.array(self.protected.osculating_elements(self.init_params[
                "start_time"]))
            deviation = current - initial
            correct_angular_deviations(deviation[2:6])
            assert np.all(np.abs(deviation[2:6]) <= np.pi), f"bad deviation {deviation}"

        self.trajectory_deviation = np.round_(deviation, 6)

    def _update_reward(self):
        """Update reward and reward components."""
        values = np.concatenate(
            (
                [self.get_total_collision_probability()],
                [self.get_fuel_consumption()],
                np.abs(self.get_trajectory_deviation()),
            )
        ).astype(np.float)
        reward_arr = reward_func(values, self._reward_thr)
        coll_prob_r = reward_arr[0]
        fuel_r = reward_arr[1]
        traj_dev_r = reward_arr[2:]
        # reward components
        self.reward_components = {
            "coll_prob": coll_prob_r, "fuel": fuel_r, "traj_dev": tuple(traj_dev_r)
        }
        # total reward
        self.reward = np.sum(reward_arr)
        assert self.reward <= 0, f"reward: {self.reward} > 0\ncomponents: {self.reward_components}"

    def _update_all_reward_components(self, zero_update=False):
        """Update total collision probability, trajectory deviation, reward components and reward."""
        self._update_total_collision_probability()
        self._update_trajectory_deviation(zero_update=zero_update)
        self._update_reward()

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

    def get_total_collision_probability(self):
        return self.total_collision_probability

    def get_trajectory_deviation(self):
        return self.trajectory_deviation

    def get_reward_components(self):
        """Provides reward components.

        Returns:
            selt.reward_components (dict): dict with keys:
                "coll_prob", "fuel", "traj_dev".
        """
        return self.reward_components

    def get_reward(self):
        return self.reward

    def get_next_action(self):
        return self.next_action

    def get_state(self):
        """ Provides environment state. """
        return self.state

    def get_fuel_consumption(self):
        return float(self.init_fuel - self.protected.get_fuel())

    def get_coords_by_epoch(self, epoch):
        st_pos, st_v = self.protected.position(epoch)
        st = np.hstack((np.array(st_pos), np.array(st_v)))[np.newaxis, ...]
        n_items = len(self.debris)
        debr = np.zeros((n_items, 6))
        for i in range(n_items):
            pos, v = self.debris[i].position(epoch)
            debr[i] = np.array(pos + v)

        return st, debr

    def get_collision_data(self):
        return self.collision_data

    def get_start_time(self):
        return self.init_params["start_time"]

    def get_end_time(self):
        return self.init_params["end_time"]

    def reset(self):
        """ Return Environment to initial state. """
        self.__init__(
            self.init_params['protected'], self.init_params['debris'],
            self.init_params['start_time'], self.init_params['end_time'],
            self.init_params['coll_prob_thr'],
            self.init_params['fuel_cons_thr'],
            self.init_params['traj_dev_thr'],
            self.init_params['target_osculating_elements'],
        )
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
        Returns:
            pos (tuple): position x, y, z (meters).
            vel (tuple): velocity Vx, Vy, Vz (m/s).
        """
        pos, vel = self.satellite.eph(epoch)
        return pos, vel

    def osculating_elements(self, epoch):
        """ Provide SpaceObject position at given epoch:
        Args:
            epoch (pk.epoch): at what time to calculate osculating elements.
        Returns:
            elements (tuple): six osculating keplerian elements (a,e,i,W,w,M).
        """
        elements = self.satellite.osculating_elements(epoch)
        return elements

    def get_name(self):
        return self.satellite.name

    def get_fuel(self):
        return self.fuel

    def get_radius(self):
        return self.satellite.radius

    def get_orbital_elements(self):
        return self.satellite.orbital_elements

    def get_mu_central_body(self):
        return self.satellite.mu_central_body

    def get_mu_self(self):
        return self.satellite.mu_self

    def get_safe_radius(self):
        return self.satellite.safe_radius

    def get_orbital_period(self):
        a = self.get_orbital_elements()[0]  # meters.
        mu = pk.MU_EARTH  # meters^3 / sc^2.
        T = 2 * np.pi * (a**3 / mu)**0.5 * pk.SEC2DAY  # mjd2000 (or days).
        return T
