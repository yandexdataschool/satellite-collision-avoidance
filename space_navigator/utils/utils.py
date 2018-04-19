import pykep as pk
from ..api import SpaceObject

def read_space_objects(file, param_type):
    """ Create SpaceObjects from a text file.
    Args:
        param_type (str): parameter types for initializing a SpaceObject.
            Could be "tle", "oph" or "osc".
    """
    space_objects = []
    with open(file, 'r') as satellites:
        while True:
            name = satellites.readline().strip()
            if not name:
                break
            if param_type == "tle":
                tle_line1 = satellites.readline().strip()
                tle_line2 = satellites.readline().strip()
                params = dict(
                    tle_line1=tle_line1,
                    tle_line2=tle_line2,
                    fuel=1,
                )
            elif param_type == "eph":
                epoch = pk.epoch(
                    float(satellites.readline().strip()), "mjd2000")
                # pos ([x, y, z]): position towards earth center (meters).
                pos = [float(x)
                       for x in satellites.readline().strip().split(",")]
                # vel ([Vx, Vy, Vz]): velocity (m/s).
                vel = [float(x)
                       for x in satellites.readline().strip().split(",")]
                mu_central_body, mu_self, radius, safe_radius = [
                    float(x) for x in satellites.readline().strip().split(",")]
                fuel = float(satellites.readline().strip())
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
                    float(satellites.readline().strip()), "mjd2000")
                # six osculating keplerian elements (a,e,i,W,w,M) at the reference epoch:
                # a (semi-major axis): meters,
                # e (eccentricity): greater than 0,
                # i (inclination), W (Longitude of the ascending node): radians,
                # w (Argument of periapsis), M (mean anomaly): radians.
                elements = tuple(
                    [float(x) for x in satellites.readline().strip().split(",")])
                mu_central_body, mu_self, radius, safe_radius = [
                    float(x) for x in satellites.readline().strip().split(",")]
                fuel = float(satellites.readline().strip())
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
