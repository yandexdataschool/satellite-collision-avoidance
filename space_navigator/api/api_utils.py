import numpy as np
import pykep as pk
import datetime


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


def get_lower_estimate_of_time_to_conjunction(protected, debris, crit_distance):
    """The lower estimate of the time to the nearest conjunction.

    It is assumed that the objects move directly to each other.

    Args:
        protected (SpaceObject): protected space object in Environment.
        debris ([SpaceObject, ]): list of other space objects.
        crit_distance (float): dangerous distance threshold (meters).

    Returns:
        time_to_conjunction (float): estimation of the time to the nearest conjunction (mjd2000).

    """
    distances = np.linalg.norm(debris[:, :3] - protected[:, :3], axis=1)
    closest_debr = np.argmin(distances)
    min_dist = distances[closest_debr]
    if min_dist <= crit_distance:
        time_to_conjunction = 0
    else:
        V1 = np.linalg.norm(protected[:, 3:])
        V2 = np.linalg.norm(debris[closest_debr, 3:])

        sec_to_collision = (min_dist - crit_distance) / (V1 + V2)
        time_to_conjunction = sec2mjd2000(sec_to_collision)
    return time_to_conjunction


def sec2mjd2000(sec):
    """Converts time in seconds to time in mjd2000."""
    assert (sec < 86400), "the number of seconds {} is greater than 86400".format(sec)
    s = "2000-01-01 " + str(datetime.timedelta(seconds=sec))
    time_mjd2000 = pk.epoch_from_string(s).mjd2000

    return time_mjd2000
