import numpy as np
import pykep as pk

import torch

from copy import copy

from ..api import SpaceObject
from ..api import Environment
from ..agent import TableAgent, PytorchAgent, adjust_action_table


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


def space_object_to_str(space_object, epoch):
    # TODO - different formats?
    name = space_object.get_name()
    time = epoch.mjd2000
    elements = space_object.get_orbital_elements()

    # pykep debug
    if np.isnan(elements[5]):
        elements = list(elements)
        elements[5] = -np.pi
        elements = tuple(elements)

    mu_central_body = space_object.get_mu_central_body()
    mu_self = space_object.get_mu_self()
    radius = space_object.get_radius()
    safe_radius = space_object.get_safe_radius()

    fuel = space_object.get_fuel()

    s = name + "\n"
    s += f"{time}\n"
    s += str(elements).strip('()') + "\n"
    s += f"{mu_central_body}, {mu_self}, {radius}, {safe_radius}\n"
    s += f"{fuel}\n"

    return s


def read_environment(path):
    """ Read Environment from a text file.

    Args:
        path (str): path to file with objects

    Returns:
        env (Environment): environment with given in file parameteres.

    TODO:
        param_type (str): parameter types for initializing a SpaceObject.
            Could be "tle", "oph" or "osc".
    """
    with open(path, 'r') as satellites:
        lines = satellites.readlines()

    start_time, end_time = [float(x) for x in lines[0].strip().split(",")]
    start_time = pk.epoch(start_time, "mjd2000")
    end_time = pk.epoch(end_time, "mjd2000")

    param_type = lines[1].strip()

    objects = read_space_objects_from_list(lines[2:], param_type)
    protected, debris = objects[0], objects[1:]

    env = Environment(protected, debris, start_time, end_time)

    return env


def write_environment(env, path):
    """ Write Environment to a text file.

    Args:
        env (Environment): environment with given parameteres.
        path (str): path to the file for writing.

    TODO:
        param_type (str): parameter types for initializing a SpaceObject.
            Could be "tle", "oph" or "osc".
    """
    env = copy(env)
    env.reset()
    start_time = env.get_start_time()
    end_time = env.get_end_time()
    with open(path, 'w') as f:
        f.write(f'{start_time.mjd2000}, {end_time.mjd2000}\n')
        f.write('osc\n')
        f.write(space_object_to_str(env.protected, start_time))
        for debr in env.debris:
            f.write(space_object_to_str(debr, start_time))


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


def actions_to_maneuvers(action_table, start_time):
    """
    Args:
        action_table (np.array with shape=(n_actions, 4) or (4) or (0)):
            table of actions with columns ["dVx", "dVy", "dVz", "time to request"].
        start_time (pk.epoch): initial time of the environment.

    Returns:
        maneuvers (list of dicts): list of maneuvers.

    """
    maneuvers = []
    maneuvers_table = adjust_action_table(action_table)
    if maneuvers_table.size:
        maneuvers_table[-1, -1] = 0
        maneuvers_table[:, 3] = np.cumsum(
            maneuvers_table[:, 3]) + start_time.mjd2000
        if np.count_nonzero(maneuvers_table[0, :3]) == 0:
            maneuvers_table = np.delete(maneuvers_table, 0, axis=0)
        for man in maneuvers_table:
            maneuvers.append(
                {"action": man[:3], "t_man": pk.epoch(man[3], "mjd2000")})
    return maneuvers
