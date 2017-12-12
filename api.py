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


def sum_collision_probability(p):
    return 1 - np.prod(1 - p)


def danger_db_and_collision_prob(st, db, treshould):
    """ Returns danger debris indices and collision probability
    Args:
        st -- np.array shape(1, 3), satellite coordinates
        db -- np.array shape(n_denris, 3), debris coordinates
        treshould -- float, danger distance
    ---
    output: dict {danger_debris: collision_probability}
    """
    # getting danger debris
    # TODO - normal distribution
    crit_dist = euclidean_distance(st, db, rev_sort=False)
    danger_debris = np.where(crit_dist < treshould)

    collision_prob = dict()
    r = treshould
    for debris in danger_debris:
        r = self.crit_conv_dist
        d = crit_dist[debris]
        coll_prob = (
            (4 * r + d) * ((2 * r - d) ** 2) / (16 * r**3)
        )
        coll_prob = sum_collision_probability(coll_prob)
        collision_prob[debris] = coll_prob

    return collision_prob


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
                    {'db': np.array shape (n_items, 6)},  debris xyz and dVx, dVy, dVz coordinates.
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
        self.crit_conv_dist = 100
        n_debris = debris.shape[0]
        self.collision_probability = dict(zip(range(n_debris, [0] * n_debris)))
        self.whole_collision_probability = 0
        self.collision_risk_reward = 0

    def propagate_forward(self, start, end, prop_step=PROPAGATION_STEP):
        """
        Args:
            start, end -- float, start and end time for propagation as mjd2000.
        """
        for t in range(start, end + prop_step, prop_step):
            epoch = pk.epoch(t, "mjd2000")
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
            # TODO - check reward update and add ++reward?
            self.update_total_collision_risk_for_iteration()

        return

    def get_reward(self):
        """ Provide total reward from the environment state.
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
            # collision_danger
            + self.collision_risk_reward
        )
        return r

    def update_total_collision_risk_for_iteration(self):
        """ Update the risk of collision on the iteration. """
        current_collision_probability = danger_db_and_collision_prob(
            st[:, :3], db[:, :3], self.crit_conv_dist)
        for d in current_collision_probability.keys():
            self.collision_probability[d] = max(
                self.collision_probability[d], current_collision_probability[d])
        self.whole_collision_probability = sum_collision_probability(
            self.collision_probability.values())
        self.collision_risk_reward = self.whole_collision_probability
        self.is_end = self.check_collision()
        return self.is_end

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

    def check_collision(self, collision_distance=50):
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
