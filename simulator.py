# Module simulator provides simulator of space environment
# and learning proccess of the agent.

import time
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


def strf_position(satellite, epoch):
    """ Print SpaceObject position. """
    pos, vel = satellite.position(epoch)
    return "{} position: x - {:0.2f}, y - {:0.2f}, z - {:0.2f}.\
      \n{} velocity: Vx - {:0.2f}, Vy - {:0.2f}, Vz - {:0.2f}\
      ".format(satellite.get_name(), pos[0], pos[1], pos[2],
               satellite.get_name(), vel[0], vel[1], vel[2])


def read_space_objects(file, param_type):
    """ Create SpaceObjects from a text file.
        param_type -- str, "tle", "oph" or "osc". Different parameter
                      types for initializing a SpaceObject.
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
                params = dict(tle_line1=tle_line1,
                              tle_line2=tle_line2,
                              fuel=1)
            elif param_type == "eph":
                epoch = pk.epoch(
                    float(satellites.readline().strip()), "mjd2000")
                pos = [float(x)
                       for x in satellites.readline().strip().split(",")]
                vel = [float(x)
                       for x in satellites.readline().strip().split(",")]
                mu_central_body, mu_self, radius, safe_radius = [
                    float(x) for x in satellites.readline().strip().split(",")]
                fuel = float(satellites.readline().strip())
                params = dict(pos=pos, vel=vel, epoch=epoch,
                              mu_central_body=mu_central_body,
                              mu_self=mu_self,
                              radius=radius,
                              safe_radius=safe_radius,
                              fuel=fuel)

            elif param_type == "osc":
                epoch = pk.epoch(
                    float(satellites.readline().strip()), "mjd2000")
                elements = tuple(
                    [float(x) for x in satellites.readline().strip().split(",")])
                mu_central_body, mu_self, radius, safe_radius = [
                    float(x) for x in satellites.readline().strip().split(",")]
                fuel = float(satellites.readline().strip())
                params = dict(elements=elements, epoch=epoch,
                              mu_central_body=mu_central_body,
                              mu_self=mu_self,
                              radius=radius,
                              safe_radius=safe_radius,
                              fuel=fuel)

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

    def run(self):
        plt.ion()
        plt.show()

    def plot_planet(self, satellite, t, size, color):
        """ Plot a pykep.planet object. """
        plot_planet(satellite, ax=self.ax,
                    t0=t, s=size, legend=True, color=color)

    def plot_earth(self):
        """ Add earth to the plot and legend. """
        self.ax.scatter(0, 0, 0, color='green', label="Earth")
        plt.legend()

    def pause_and_clear(self):
        """ Pause the frame to watch it. Clear axis for next frame. """
        plt.legend()
        plt.pause(PAUSE_TIME)
        plt.cla()


class Simulator:
    """ Simulator allows to start the simulation of provided environment,
    and starts agent-environment collaboration.
    """

    def __init__(self, agent, environment, start_time=None):
        """
            agent -- Agent(), agent, to do actions in environment.
            environment -- Environment(), the initial space environment.
            start_time -- pk.epoch, start epoch of simulation.
        """
        self.is_end = False

        self.agent = agent
        self.env = environment

        if not start_time:
            self.start_time = pk.epoch_from_string(
                time.strftime("%Y-%m-%d %T"))
        else:
            self.start_time = start_time
        self.curr_time = self.start_time

        self.vis = Visualizer()
        self.logger = logging.getLogger('simulator.Simulator')

    def run(self, visualize=True, num_iter=None, step=0.001):
        iteration = 0

        if visualize:
            self.vis.run()

        while iteration != num_iter:
            self.env.propagate_forward(
                self.curr_time.mjd2000 - step, self.curr_time.mjd2000)
            self.is_end = self.env.update_total_collision_risk()

            if self.is_end:
                break

            if self.curr_time.mjd2000 >= self.env.get_next_action().mjd2000:
                s = self.env.get_state()
                action = self.agent.get_action(s)
                # r = self.env.get_reward()
                r = self.env.reward
                self.env.act(action)

                self.log_ra(iteration, r, action)

            self.log_iteration(iteration)
            self.log_protected_position()
            self.log_debris_positions()

            if visualize:
                self.plot_protected()
                self.plot_debris()
                self.vis.plot_earth()
                self.vis.pause_and_clear()

            self.curr_time = pk.epoch(
                self.curr_time.mjd2000 + step, "mjd2000")

            iteration += 1
        # TODO - whole reward
        # TODO - probability of collision
        print("Simulation ended. Collision: {}".format(self.is_end))

    def log_protected_position(self):
        self.logger.info(strf_position(self.env.protected, self.curr_time))

    def log_debris_positions(self):
        for obj in self.env.debris:
            self.logger.info(strf_position(obj, self.curr_time))

    def log_iteration(self, iteration):
        self.logger.debug("Iter #{} \tEpoch: {}\tCollision: {}\t Collision Probability: {}".format(
            iteration,  self.curr_time, self.is_end, self.env.whole_collision_probability))

    def log_ra(self, iteration, reward, action):
        self.logger.info("Iter: {}\tReward: {}\t action: {}".format(
            iteration, reward, action))

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
