# Module simulator provides simulator of space environment
# and learning proccess of the agent.

import logging

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
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
        self.fig = plt.figure(figsize=[14, 10])
        self.gs = gridspec.GridSpec(3, 2)
        self.subplot_3d = self.fig.add_subplot(self.gs[:, 0], projection='3d')
        self.subplot_3d.set_aspect("equal")
        self.subplot_p = self.fig.add_subplot(self.gs[0, 1])
        self.subplot_f = self.fig.add_subplot(self.gs[1, 1])
        self.subplot_r = self.fig.add_subplot(self.gs[2, 1])

    def run(self):
        plt.ion()

    def plot_planet(self, satellite, t, size, color):
        """ Plot a pykep.planet object. """
        plot_planet(satellite, ax=self.subplot_3d,
                    t0=t, s=size, legend=True, color=color)

    def plot_earth(self):
        """ Add earth to the plot and legend. """
        draw_sphere(self.subplot_3d, (0, 0, 0), EARTH_RADIUS, {
            "color": "b", "lw": 0.5, "alpha": 0.2})
        plt.legend()

    def pause_and_clear(self):
        """ Pause the frame to watch it. Clear axis for next frame. """
        plt.legend()
        plt.pause(PAUSE_TIME)
        self.subplot_3d.cla()
        self.subplot_p.cla()
        self.subplot_f.cla()
        self.subplot_r.cla()

    def plot_iteration(self, epoch, last_update, collision_prob, fuel_cons, reward):
        s = '  Epoch: {}\nUpdate: {}\nColl Prob: {:.7}     Fuel Cons: {:.5}     Reward: {:.5}'.format(
            epoch, last_update, collision_prob, fuel_cons, reward)
        self.subplot_3d.text2D(-0.3, 1.05, s,
                               transform=self.subplot_3d.transAxes)

    def plot_prob_fuel(self, time_arr, prob_arr, fuel_cons_arr, reward_arr):
        self.subplot_p.step(time_arr, prob_arr)
        self.subplot_p.set_title('Total collision probability')
        self.subplot_p.grid(True)
        self.subplot_p.set_ylabel('prob')

        self.subplot_f.step(time_arr, fuel_cons_arr)
        self.subplot_f.set_title('Total fuel consumption')
        self.subplot_f.grid(True)
        self.subplot_f.set_ylabel('fuel (dV)')

        self.subplot_r.step(time_arr, reward_arr)
        self.subplot_r.set_title('Total reward')
        self.subplot_r.grid(True)
        self.subplot_f.set_ylabel('reward')
        self.subplot_r.set_xlabel('time (mjd2000)')


class Simulator:
    """ Simulator allows to start the simulation of provided environment,
    and starts agent-environment collaboration.
    """

    def __init__(self, agent, environment, update_r_p_step=None, print_out=True):
        """
        Args:
            agent (api.Agent, agent, to do actions in environment.
            environment (api.Environment): the initial space environment.
            start_time (pk.epoch): start epoch of simulation.
            update_r_p_step (int): update_r_p_step (int): update reward and probability step;
                if update_r_p_step == None, reward and probability are updated only by agent).
            print_out (bool): print out some parameters and results (reward and probability).
        """

        self.agent = agent
        self.env = environment
        self.start_time = self.env.state["epoch"]
        self.curr_time = self.start_time

        self.logger = logging.getLogger('simulator.Simulator')
        self.print_out = print_out
        self.update_r_p_step = update_r_p_step

        self.vis = None
        self.time_arr = None
        self.prob_arr = None
        self.fuel_cons_arr = None
        self.reward_arr = None

    def run(self, end_time, step=0.001, visualize=True):
        """
        Args:
            end_time (float): end time of simulation provided as mjd2000.
            step (float): time step in simulation.
            visualize (bool): whether show the simulation or not.

        Returns:
            reward (self.env.get_reward()): reward of session.

        """
        iteration = 0
        if visualize:
            self.vis = Visualizer()
            self.vis.run()
            self.time_arr = [self.curr_time.mjd2000]
            self.prob_arr = [self.env.total_collision_probability]
            self.fuel_cons_arr = [self.env.get_fuel_consumption()]
            self.reward_arr = [self.env.reward]

        if self.print_out:
            print("Simulation started.\n\nStart time: {} \t End time: {} \t Simulation step:{}\n".format(
                self.start_time.mjd2000, end_time, step))
            print("Protected SpaceObject:\n{}".format(
                self.env.protected.satellite))
            print("Debris objects:\n")
            for spaceObject in self.env.debris:
                print(spaceObject.satellite)

        while self.curr_time.mjd2000 <= end_time:
            self.env.propagate_forward(
                self.curr_time.mjd2000, self.update_r_p_step)

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
                p = self.env.total_collision_probability
                f = self.env.get_fuel_consumption()
                r = self.env.reward
                if iteration % self.update_r_p_step == 0:
                    self.time_arr.append(self.curr_time.mjd2000)
                    self.prob_arr.append(p)
                    self.fuel_cons_arr.append(f)
                    self.reward_arr.append(r)
                self.plot_prob_fuel()
                self.vis.pause_and_clear()
                # self.env.reward and self.env.total_collision_probability -
                # without update.

                self.vis.plot_iteration(
                    self.curr_time, self.env.last_r_p_update, p, f, r)

            self.curr_time = pk.epoch(
                self.curr_time.mjd2000 + step, "mjd2000")

            iteration += 1

        self.log_protected_position()

        if self.print_out:
            print("Simulation ended.\nCollision probability: {}.\nReward: {}.\nFuel consumption: {}.".format(
                self.env.get_collision_probability(), self.env.get_reward(), self.env.get_fuel_consumption()))

        return self.env.get_reward()

    def log_protected_position(self):
        self.logger.info(strf_position(self.env.protected, self.curr_time))

    def log_debris_positions(self):
        for obj in self.env.debris:
            self.logger.info(strf_position(obj, self.curr_time))

    def log_iteration(self, iteration):
        self.logger.debug("Iter #{} \tEpoch: {} \tCollision Probability: {}".format(
            iteration,  self.curr_time, self.env.total_collision_probability))

    def log_reward_action(self, iteration, reward, action):
        self.logger.info("Iter: {} \tReward: {} \taction: (dVx:{}, dVy: {}, dVz: {}, time_to_request: {})".format(
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

    def plot_prob_fuel(self):
        self.vis.plot_prob_fuel(
            self.time_arr, self.prob_arr, self.fuel_cons_arr, self.reward_arr)
