import numpy as np
import pykep as pk

SEC_IN_DAY = 86400  # number of seconds in one day


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


def lower_estimate_of_time_to_conjunction(prot_rV, debr_rV, crit_distance):
    """The lower estimate of the time to the nearest conjunction.

    It is assumed that the objects move directly to each other.

    Args:
        prot_rV (np.array with shape(1, 6)): vector of coordinates (meters)
            and velocities (m/s) for protected object. Vector format: (x,y,z,Vx,Vy,Vz).
        debr_rV (np.array with shape(n_denris, 6)): vectors of coordinates (meters)
            and velocities (m/s) for each debris. Vectors format: (x,y,z,Vx,Vy,Vz).
        crit_distance (float): dangerous distance threshold (meters).

    Returns:
        dangerous_debris (np.array): dangerous debris indicies.
        distances (np.array): Euclidean distances for the each dangerous debris (meters).
        time_to_conjunction (float): estimation of the time to the nearest conjunction (mjd2000).

    TODO:
        The assumption depends on eccentricity and orbit type
            because velocity could change dramatically
            (examples: https://en.wikipedia.org/wiki/Elliptic_orbit).
            Add an account of eccentricity and orbit type.
    """
    if not debr_rV.size:
        return np.array([]), np.array([]), float("inf")

    distances = np.linalg.norm(debr_rV[:, :3] - prot_rV[:, :3], axis=1)
    closest_debr = np.argmin(distances)
    min_dist = distances[closest_debr]
    dangerous_debris = np.where(distances <= crit_distance)[0]
    if min_dist <= crit_distance:
        time_to_conjunction = 0
    else:
        V1 = np.linalg.norm(prot_rV[:, 3:])
        V2 = np.linalg.norm(debr_rV[closest_debr, 3:])
        sec_to_collision = (min_dist - crit_distance) / (V1 + V2)
        time_to_conjunction = sec_to_collision / SEC_IN_DAY

    return dangerous_debris, distances[dangerous_debris], time_to_conjunction


# TODO - explore reward functions

def reward_func_0(value, thr, r_thr=-1,
                  thr_times_exceeded=2, r_thr_times_exceeded=-10):
    """Reward function.

    Piecewise linear function with increased penalty for exceeding the threshold.

    Args:
        value (float): value of the rewarded parameter.
        thr (float): threshold of the rewarded parameter. 
        r_thr (float): reward in case the parameter value is equal to the threshold.
        thr_times_exceeded (float): how many times should the parameter value exceed
            the threshold to be rewarded as r_thr_times_exceeded.
        r_thr_times_exceeded (float): reward in case the parameter value exceeds
            the threshold thr_times_exceeded times.

    Returns:
        reward (float): result reward.
    """
    if value <= thr:
        reward = value * r_thr / thr
    else:
        reward = (
            (-r_thr_times_exceeded + r_thr)
            * (1 - value / thr) / (thr_times_exceeded - 1)
            + r_thr
        )
    return reward


def reward_func(values, thr, reward_func=reward_func_0, *args, **kwargs):
    """Returns reward values for np.array input.

    Args:
        values (np.array): array of values of the rewarded parameters.
        thr (np.array): array of thresholds of the rewarded parameter.
            if the threshold is np.nan, then the reward is 0
            (there is no penalty for the parameter).
        reward_func (function): reward function.
        *args, **kwargs: additional arguments of reward_func.

    Returns: 
        reward (np.array): reward array.
    """

    def reward_thr(values, thr):
        return reward_func(values, thr, *args, **kwargs)
    reward_thr_v = np.vectorize(reward_thr)

    reward = np.zeros_like(values)
    id_nan = np.isnan(thr)
    id_not_nan = np.logical_not(id_nan)

    reward[id_nan] = 0.
    if np.count_nonzero(id_not_nan) != 0:
        reward[id_not_nan] = reward_thr_v(values[id_not_nan], thr[id_not_nan])
    return reward


def correct_angular_deviations(angular_deviations):
    # check over pi angular deviations
    over_pi_angular_deviations_1 = angular_deviations > np.pi
    over_pi_angular_deviations_2 = angular_deviations < -np.pi
    angular_deviations[over_pi_angular_deviations_1] = (
        angular_deviations[over_pi_angular_deviations_1] - 2 * np.pi)
    angular_deviations[over_pi_angular_deviations_2] = (
        angular_deviations[over_pi_angular_deviations_2] + 2 * np.pi)
    return angular_deviations
