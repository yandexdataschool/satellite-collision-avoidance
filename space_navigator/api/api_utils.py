import numpy as np


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
