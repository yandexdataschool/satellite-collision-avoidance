# Module simulator provides simulator of space environment
# and learning proccess of the agent.

import logging

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import pykep as pk
from pykep.orbit_plots import plot_planet

from api import SpaceObject

logging.basicConfig(filename="simulator.log", level=logging.DEBUG,
                    filemode='w', format='%(name)s:%(levelname)s\n%(message)s\n')

PAUSE_TIME = 0.0001
EARTH_RADIUS = 6.3781e6  # meters


def draw_sphere(axis, centre, radius, wireframe_params={}):
    """
    Draws a wireframe sphere.
    Args:
       axis (matplotlib.axes._subplots.Axes3DSubplot): axis to plot on
       centre (list-like): sphere centre. Must support [] operator
       radius (float): sphere radius
       wireframe_params (dict): arguments to pass to plot_wireframe
    Returns:
       mpl_toolkits.mplot3d.art3d.Line3DCollection
    """
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + centre[0]
    y = radius * np.sin(u) * np.sin(v) + centre[1]
    z = radius * np.cos(v) + centre[2]
    return axis.plot_wireframe(x, y, z, **wireframe_params)


def strf_position(satellite, epoch):
    """ Print SpaceObject position at epoch. """
    pos, vel = satellite.position(epoch)
    return "{} position: x - {:0.5f}, y - {:0.5f}, z - {:0.5f}.\
      \n{} velocity: Vx - {:0.5f}, Vy - {:0.5f}, Vz - {:0.5f}\
      ".format(satellite.get_name(), pos[0], pos[1], pos[2],
               satellite.get_name(), vel[0], vel[1], vel[2])


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


class Visualizer:
    """ Visualizer allows to plot satellite movement simulation
        in real time.
    """

    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_aspect("equal")

    def run(self):
        plt.ion()

    def plot_planet(self, satellite, t, size, color):
        """ Plot a pykep.planet object. """
        plot_planet(satellite, ax=self.ax,
                    t0=t, s=size, legend=True, color=color)

    def plot_earth(self):
        """ Add earth to the plot and legend. """
        draw_sphere(self.ax, (0, 0, 0), EARTH_RADIUS, {
            "color": "b", "lw": 0.5, "alpha": 0.2})
        plt.legend()

    def pause_and_clear(self):
        """ Pause the frame to watch it. Clear axis for next frame. """
        plt.legend()
        plt.pause(PAUSE_TIME)
        plt.cla()

    def plot_iteration(self, epoch, last_update, reward, collision_prob):
        s = 'Epoch: {}     Last Update: {}     R: {:.7}     Coll Prob: {:.5}'.format(
            epoch, last_update, reward, collision_prob)
        self.ax.text2D(-0.2, 1.1, s, transform=self.ax.transAxes)


class Simulator:
    """ Simulator allows to start the simulation of provided environment,
    and starts agent-environment collaboration.
    """

    def __init__(self, agent, environment, print_out=False):
        """
        Args:
            agent (api.Agent, agent, to do actions in environment.
            environment (api.Environment): the initial space environment.
            start_time (pk.epoch): start epoch of simulation.
            print_out (bool): print out some results for each step.
        """

        self.agent = agent
        self.env = environment
        self.start_time = self.env.state["epoch"]
        self.curr_time = self.start_time

        self.vis = Visualizer()
        self.logger = logging.getLogger('simulator.Simulator')
        self.print_out = print_out

    def run(self, end_time, step=0.001, visualize=True, reward_probability_update_step=10):
        """
        Args:
            end_time (float): end time of simulation provided as mjd2000.
            step (float): time step in simulation.
            visualize (bool): whether show the simulation or not.
        """
        iteration = 0
        if visualize:
            self.vis.run()

        if self.print_out:
            print("Simulation started.\n\nStart time: {} \t End time: {} \t Simulation step:{}\n".format(
                self.start_time.mjd2000, end_time, step))
            print("Protected SpaceObject:\n{}".format(
                self.env.protected.satellite))
            print("Debris objects:\n")
            for spaceObject in self.env.debris:
                print(spaceObject.satellite)

        while self.curr_time.mjd2000 <= end_time:
            self.env.propagate_forward(self.curr_time.mjd2000)
            if iteration % reward_probability_update_step == 0:
                self.env.get_reward()

            if self.curr_time.mjd2000 >= self.env.get_next_action().mjd2000:
                s = self.env.get_state()
                action = self.agent.get_action(s)
                r = self.env.get_reward()
                err = self.env.act(action)
                if err:
                    self.log_bad_action(err, action)

                self.log_reward_action(iteration, r, action)

            self.log_iteration(iteration)
            self.log_protected_position()
            self.log_debris_positions()

            if visualize:
                self.plot_protected()
                self.plot_debris()
                self.vis.plot_earth()
                self.vis.pause_and_clear()
                # self.env.reward - reward without update
                self.vis.plot_iteration(
                    self.curr_time, self.env.last_r_p_update, self.env.reward, self.env.total_collision_probability)

            self.curr_time = pk.epoch(
                self.curr_time.mjd2000 + step, "mjd2000")

            if self.print_out:

                print("\niteration:", iteration)
                print("crit distance:", self.env.crit_distance)
                print("min_distances_in_current_conjunction:",
                      self.env.min_distances_in_current_conjunction)
                print("collision_probability_prior_to_current_conjunction:",
                      self.env.collision_probability_prior_to_current_conjunction)
                print("danger debris in curr conj:",
                      self.env.dangerous_debris_in_current_conjunction)

                print("total coll prob array:",
                      self.env.total_collision_probability_array)
                print("total coll prob:",
                      self.env.total_collision_probability)
                print("traj dev:", self.env.whole_trajectory_deviation)
                # self.env.reward - reward without update
                print("reward:", self.env.reward)
            iteration += 1

        self.log_protected_position()

        print("Simulation ended.\nCollision probability: {}.\nReward: {}.".format(
            self.env.get_collision_probability(), self.env.get_reward()))

    def log_protected_position(self):
        self.logger.info(strf_position(self.env.protected, self.curr_time))

    def log_debris_positions(self):
        for obj in self.env.debris:
            self.logger.info(strf_position(obj, self.curr_time))

    def log_iteration(self, iteration):
        self.logger.debug("Iter #{} \tEpoch: {} \tCollision Probability: {}".format(
            iteration,  self.curr_time, self.env.total_collision_probability))

    def log_reward_action(self, iteration, reward, action):
        self.logger.info("Iter: {} \tReward: {} \taction: (dVx:{}, dVy: {}, dVz: {}, epoch: {}, time_to_request: {})".format(
            iteration, reward, *action))

    def log_bad_action(self, message, action):
        self.logger.warning(
            "Unable to make action (dVx:{}, dVy:{}, dVz:{}): {}".format(action[0], action[1], action[2], message))

    def plot_protected(self):
        """ Plot Protected SpaceObject. """
        self.vis.plot_planet(self.env.protected.satellite,
                             t=self.curr_time, size=100, color="black")

    def plot_debris(self):
        """ Plot space debris. """
        cmap = plt.get_cmap('gist_rainbow')
        n_items = len(self.env.debris)
        colors = [cmap(i) for i in np.linspace(0, 1, n_items)]
        for i in range(n_items):
            self.vis.plot_planet(
                self.env.debris[i].satellite, t=self.curr_time,
                size=25, color=colors[i])
