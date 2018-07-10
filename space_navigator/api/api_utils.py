import numpy as np
import pykep as pk
import datetime

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

    """
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


def reward_threshold(x, thr, mult=2, y_t=1, y_t_mult=10):
    if x <= thr:
        y = - x * y_t / thr
    else:
        y = (y_t_mult - y_t) * (1 - x / thr) / (mult - 1) - y_t
    return y
