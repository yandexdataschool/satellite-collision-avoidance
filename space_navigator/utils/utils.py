import numpy as np
import pykep as pk

import torch

from ..api import SpaceObject
from ..api import Environment
from ..agent import TableAgent, PytorchAgent


def read_space_objects(file, param_type):
    """ Create SpaceObjects from a text file.
    Args:
        file (str): path to file with objects
        param_type (str): parameter types for initializing a SpaceObject.
            Could be "tle", "oph" or "osc".
    Returns:
        ([SpaceObject]): list of space objects.
    """
    with open(file, 'r') as satellites:
        lines = satellites.readlines()
    return read_space_objects_from_list(lines, param_type)


def read_space_objects_from_list(lines, param_type):
    """ Create SpaceObjects from a lisst.
    Args:
        lines (list): list of lines with objects.
        param_type (str): parameter types for initializing a SpaceObject.
            Could be "tle", "oph" or "osc".
    Returns:
        ([SpaceObject]): list of space objects.
    """
    space_objects = []
    iterator = iter(lines)
    while True:
        try:
            name = next(iterator).strip()
        except StopIteration:
            break
        if param_type == "tle":
            tle_line1 = next(iterator).strip()
            tle_line2 = next(iterator).strip()
            params = dict(
                tle_line1=tle_line1,
                tle_line2=tle_line2,
                fuel=1,
            )
        elif param_type == "eph":
            epoch = pk.epoch(
                float(next(iterator).strip()), "mjd2000")
            # pos ([x, y, z]): position towards earth center (meters).
            pos = [float(x)
                   for x in next(iterator).strip().split(",")]
            # vel ([Vx, Vy, Vz]): velocity (m/s).
            vel = [float(x)
                   for x in next(iterator).strip().split(",")]
            mu_central_body, mu_self, radius, safe_radius = [
                float(x) for x in next(iterator).strip().split(",")]
            fuel = float(next(iterator).strip())
            params = dict(
                pos=pos, vel=vel, epoch=epoch,
                mu_central_body=mu_central_body,
                mu_self=mu_self,
                radius=radius,
                safe_radius=safe_radius,
                fuel=fuel,
            )

        elif param_type == "osc":
            epoch = pk.epoch(
                float(next(iterator).strip()), "mjd2000")
            # six osculating keplerian elements (a,e,i,W,w,M) at the reference epoch:
            # a (semi-major axis): meters,
            # e (eccentricity): greater than 0,
            # i (inclination), W (Longitude of the ascending node): radians,
            # w (Argument of periapsis), M (mean anomaly): radians.
            elements = tuple(
                [float(x) for x in next(iterator).strip().split(",")])
            mu_central_body, mu_self, radius, safe_radius = [
                float(x) for x in next(iterator).strip().split(",")]
            fuel = float(next(iterator).strip())
            params = dict(
                elements=elements, epoch=epoch,
                mu_central_body=mu_central_body,
                mu_self=mu_self,
                radius=radius,
                safe_radius=safe_radius,
                fuel=fuel,
            )

        satellite = SpaceObject(name, param_type, params)
        space_objects.append(satellite)

    return space_objects


def read_environment(path):
    """ Create SpaceObjects from a text file.
    Args:
        path (str): parameter types for initializing a SpaceObject.
            Could be "tle", "oph" or "osc".
    Returns:
        (Environment): environment with given in file parameteres.
    """
    with open(path, 'r') as satellites:
        lines = satellites.readlines()

    start_time, end_time = [float(x) for x in lines[0].strip().split(",")]
    start_time = pk.epoch(start_time, "mjd2000")
    end_time = pk.epoch(end_time, "mjd2000")

    param_type = lines[1].strip()

    objects = read_space_objects_from_list(lines[2:], param_type)
    protected, debris = objects[0], objects[1:]

    return Environment(protected, debris, start_time, end_time)


def get_agent(agent_type, model_path='', num_inputs=6, num_outputs=4, hidden_size=64):
    """ ... """
    if agent_type == 'table':
        if model_path:
            action_table = np.loadtxt(model_path, delimiter=',')
            agent = TableAgent(action_table)
        else:
            agent = TableAgent()
    elif agent_type == 'pytorch':
        agent = PytorchAgent(num_inputs, num_outputs, hidden_size)
        if model_path:
            agent.load_state_dict(torch.load(model_path))
    else:
        raise ValueError("Invalid agent type")
    return agent


def is_action_table_empty(action_table):
    return action_table.size == 0 or np.count_nonzero(action_table[:, :3]) == 0


def action_table2maneuver_table(action_table, start_time):
    """"""
    if is_action_table_empty(action_table):
        return np.zeros((0, 4))
    maneuvers = action_table
    maneuvers[1:, 3] = maneuvers[:-1, 3]
    maneuvers[0, 3] = 0
    maneuvers[:, 3] = np.cumsum(maneuvers[:, 3]) + start_time.mjd2000
    if np.count_nonzero(maneuvers[0, :3]) == 0:
        maneuvers = maneuvers[1:]
    return maneuvers
