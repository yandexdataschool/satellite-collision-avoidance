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

MAX_PROPAGATION_STEP = 0.000001  # is equal to 0.0864 sc.


def euclidean_distance(r_main, r_other, rev_sort=True):
    """Euclidean distances calculation.

    Args:
        r_main (np.array with shape=(1, 3)): xyz coordinates of main object.
        r_other (np.array with shape=(n_objects, 3)): xyz coordinates of other objects.
        rev_sort (bool): True for reverse sorted distances array, False for not sorted one.

    Returns:
        distances (np.array with shape=(n_objects)): Euclidean distances between main object and the others.

    """
    distances = np.sum(
        (r_main - r_other) ** 2,
        axis=1) ** 0.5
    if rev_sort:
        distances = np.sort(distances)[::-1]
    return distances


def fuel_consumption(dV):
    """ Provide the value of fuel consumption for given velocity delta.

    Args:
        dV (np.array ([dVx, dVy, dVz]) ): vector of satellite velocity delta for maneuver.

    Returns:
        float: fuel consumption.
    """
    return np.linalg.norm(dV)


def sum_coll_prob(p):
    """Summation of probabilities.

    Agrs:
        p (list or np.array or tuple or set): probabilities.

    Returns:
        result (float): probabilities sum.

    Raises:
        ValueError: If any probability has incorrect value.
        TypeError: If any probability has incorrect type.

    """
    result = (1 - np.prod(1 - np.array(p)))
    TestProbability(result)
    return result


def rV2ocs(r0, r1, V0, V1):
    """ Convert cartesian coordinate system to orbital coordinate system.

    Args:
        r0, r1 (np.array([x,y,z])): coordinates.
        V0, V1 (np.array([Vx,Vy,Vz])): velocities.

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


def coll_prob_estimation(r0, r1, V0=np.zeros(3), V1=np.zeros(3), d0=1, d1=1,
                         sigma=1., approach="normal"):
    """ Returns probability of collision between two objects
    Args:
        r0, r1 (np.array([x, y, z])): objects coordinates.
        V0, V1 (np.array([Vx,Vy,Vz])): velocities.
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

    if approach == "Hutor":
        # V test
        if np.array_equal(V0, np.zeros(3)) | np.array_equal(V1, np.zeros(3)):
            raise Exception("velocities are required for Hutorovski approach")
        # cosine of angle between velocities
        cos_vel_angle = (np.dot(V0, V1)
                         / (np.linalg.norm(V0) * np.linalg.norm(V1)))
        dr, dn, db = rV2ocs(r0, r1, V0, V1)
        A = (dr**2 / (4 * sigma**2)
             + dn**2 / (2 * sigma**2)
             + db**2 / (2 * sigma**2))
        k = ((d0 + d1)**2
             / (sigma * (11 + 5 * sigma * cos_vel_angle**2)))
        probability = k * np.exp(-A)
    elif approach == "normal":
        # TODO - truncated normal distribution?
        # TODO - multivariate normal distribution?
        for c0, c1 in zip(r0, r1):
            av = (c0 + c1) / 2.
            integtal = norm.cdf(av, loc=min(c0, c1), scale=sigma)
            probability *= (1 - integtal) / integtal
    else:
        raise Exception("unknown probability estimation approach:" + approach)
    TestProbability(probability)
    return probability


def danger_debr_and_collision_prob(st_rV, debr_rV, st_d, debr_d, sigma, threshold_p):
    """Probability of collision with danger debris.

    Args:
        st_r (np.array with shape(1, 6)): satellite position and velocity.
        debr_r (np.array with shape(n_denris, 6)): debris positions and velocities.
        st_d (float): satellite size.
        debr_d (np.array with shape(n_denris, 6)): debris sizes
        threshold_p (float): danger probability.
        sigma (float): standard deviation.

    Returns:
        coll_prob (dict {danger_debris: coll_prob}): danger debris indices and collision probability.

    Raises:
        ValueError: If any probability has incorrect value.
        TypeError: If any probability has incorrect type.
        ValueError: If sigma <= 0.

    """
    TestProbability(threshold_p)
    if (sigma <= 0):
        raise ValueError("sigma should be more than 0")

    coll_prob = dict()
    for d in range(debr_rV.shape[0]):
        p = coll_prob_estimation(
            st_rV[0, :3], debr_rV[d, :3], st_rV[0, 3:], debr_rV[d, 3:], st_d, debr_d[d], sigma)
        if (p >= threshold_p):
            coll_prob[d] = p

    return coll_prob


class Agent:
    """ Agent implements an agent to communicate with space Environment.
        Agent can make actions to the space environment by taking it's state
        after the last action.
    """

    def __init__(self):
        """"""

    def get_action(self, state):
        """ Provides action  for protected object.
        Args:
            state (dict): environment state. 
                {'coord' (dict):
                    {'st' (np.array with shape (1, 6)): satellite r and Vx, Vy, Vz coordinates.
                     'debr' (np.array with shape (n_items, 6)): debris r and Vx, Vy, Vz coordinates.}
                'trajectory_deviation_coef' (float).
                'epoch' (pk.epoch): at which time environment state is calculated. }
        Returns:
            np.array([dVx, dVy, dVz, pk.epoch, time_to_req]): vector of deltas for
                protected object, maneuver time and time to request the next action.
        """
        dVx, dVy, dVz = 0, 0, 0
        epoch = state["epoch"].mjd2000
        time_to_req = 1
        action = np.array([dVx, dVy, dVz, epoch, time_to_req])
        return action


class Environment:
    """ Environment provides the space environment with space objects:
        satellites and debris, in it.
    """

    def __init__(self, protected, debris, start_time):
        """
        Args:
            protected (SpaceObject): protected space object in Environment.
            debris ([SpaceObject, ]): list of other space objects.
            start (pk.epoch): initial time of the environment.
        """
        self.protected = protected
        self.debris = debris
        self.next_action = pk.epoch(0)
        self.state = dict(epoch=start_time, fuel=self.protected.get_fuel())
        n_debris = len(debris)
        # satellite size
        self.st_d = 1.
        # debris sizes
        self.debr_d = np.ones(n_debris)
        # critical convergence distance
        # TODO choose true distance
        self.crit_prob = 10e-5
        self.sigma = 2000000
        self.collision_probability_in_current_conjunction = dict()
        self.collision_probability_prior_to_current_conjunction_dict = dict(
            zip(range(n_debris), np.zeros(n_debris)))

        self.total_collision_probability_dict = dict(
            zip(range(n_debris), np.zeros(n_debris)))
        self.total_collision_probability = 0
        self.whole_trajectory_deviation = 0
        self.reward = 0

    def propagate_forward(self, end_time):
        """ 
        Args:
            end_time (float): end time for propagation as mjd2000.

        Raises:
            ValueError: if end_time is less then current time of the environment.
            Exception: if step in propagation_grid is less then MAX_PROPAGATION_STEP.
        """
        curr_time = self.state["epoch"].mjd2000
        if end_time <= curr_time:
            return
        elif end_time < curr_time:
            raise ValueError(
                "end_time should be greater or equal to current time")

        # Choose number of steps in linspace, s.t.
        # restep is less then MAX_PROPAGATION_STEP.
        number_of_time_steps_plus_one = np.ceil(
            (end_time - curr_time) / MAX_PROPAGATION_STEP) + 1

        propagation_grid, retstep = np.linspace(
            curr_time, end_time, number_of_time_steps_plus_one, retstep=True)

        if retstep > MAX_PROPAGATION_STEP:
            raise Exception(
                "Step in propagation grid should be <= MAX_PROPAGATION_STEP")

        for t in propagation_grid:
            epoch = pk.epoch(t, "mjd2000")
            st_pos, st_v = self.protected.position(epoch)
            st = np.hstack((np.array(st_pos), np.array(st_v)))
            st = np.reshape(st, (-1, 6))
            n_items = len(self.debris)
            debr = np.zeros((n_items, 6))
            for i in range(n_items):
                pos, v = self.debris[i].position(epoch)
                debr[i] = np.hstack((np.array(pos), np.array(v)))
            debr = np.reshape(debr, (-1, 6))

            coord = dict(st=st, debr=debr)
            trajectory_deviation_coef = 0.0
            self.whole_trajectory_deviation += trajectory_deviation_coef
            self.state = dict(
                coord=coord, trajectory_deviation_coef=trajectory_deviation_coef,
                epoch=epoch, fuel=self.protected.get_fuel())
            self.update_collision_probability()
            self.reward = self.get_reward()

        return

    def get_reward(self, coll_prob_w=0.6, traj_w=0.2, fuel_w=0.2):
        """ Provide total reward from the environment state.
        Args:
            coll_prob_w, traj_w, fuel_w (float): weights of reward components.

        Returns:
            float: total reward.
        """
        # reward components
        coll_prob = self.total_collision_probability
        ELU = lambda x: x if (x >= 0) else (1 * (np.exp(x) - 1))
        # collision probability reward - some kind of ELU function
        # of collision probability
        coll_prob_r = -(ELU((coll_prob - 10e-4) * 10000) + 1)
        traj_r = -self.whole_trajectory_deviation
        fuel_r = self.protected.get_fuel()

        # whole reward
        # TODO - add weights to all reward components
        r = (coll_prob_w * coll_prob_r
             + traj_w * traj_r
             + fuel_w * fuel_r)
        return r

    def update_collision_probability(self):
        """ Update the probability of collision on the propagation step.

        Returns"
            [float, ]: list of current collision probabilities.
        """
        # TODO - log probability?
        new_collision_probability_in_current_conjunction = danger_debr_and_collision_prob(
            self.state['coord']['st'],
            self.state['coord']['debr'],
            self.st_d, self.debr_d,
            self.sigma, self.crit_prob)

        for d in range(len(self.debris)):
            if d in new_collision_probability_in_current_conjunction:
                if d in self.collision_probability_in_current_conjunction:
                    # convergence continues
                    # update collision_probability_in_current_conjunction
                    self.collision_probability_in_current_conjunction[d] = max(
                        new_collision_probability_in_current_conjunction[d], self.collision_probability_in_current_conjunction[d])
                else:
                    # convergence begins
                    # add probability to buffer
                    self.collision_probability_in_current_conjunction[
                        d] = new_collision_probability_in_current_conjunction[d]

                self.total_collision_probability_dict[d] = sum_coll_prob(
                    [self.collision_probability_in_current_conjunction[d],
                     self.collision_probability_prior_to_current_conjunction_dict[d]])

            elif d in self.collision_probability_in_current_conjunction:
                # convergence discontinued
                # clean buffer
                self.collision_probability_prior_to_current_conjunction_dict[
                    d] = self.total_collision_probability_dict[d]
                del self.collision_probability_in_current_conjunction[d]

        self.total_collision_probability = sum_coll_prob(
            list(self.total_collision_probability_dict.values()))

        return

    def act(self, action):
        """ Change velocity for protected object.
        Args:
            action (np.array([dVx, dVy, dVz, pk.epoch, time_to_req])): vector of deltas for
                protected object, maneuver time and time to request the next action.
        """
        self.next_action = pk.epoch(
            self.state["epoch"].mjd2000 + action[4], "mjd2000")
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
                    "pos" ([x, y, z]): position (cartesian).
                    "vel" ([Vx, Vy, Vz]): velocity (cartesian).
                    "epoch" (pykep.epoch): start time.
                    "mu_central_body" (float): gravity parameter of the
                        central body (SI units, i.e. m^2/s^3).
                    "mu_self"(float): gravity parameter of the planet
                        (SI units, i.e. m^2/s^3).
                    "radius" (float): body radius (SI units, i.e. meters).
                    "safe_radius" (float): mimimual radius that is safe during
                        a fly-by of the planet (SI units, i.e. m).

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
            action (np.array([dVx, dVy, dVz, pk.epoch])): vector
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
        """ Provide SpaceObject position at given epoch:
        Args:
            epoch (pk.epoch): at what time to calculate position.
        Returns
            pos (tuple): position x, y, z.
            vel (tuple): velocity Vx, Vy, Vz.
        """
        pos, vel = self.satellite.eph(epoch)
        return pos, vel

    def get_name(self):
        return self.satellite.name

    def get_fuel(self):
        return self.fuel
