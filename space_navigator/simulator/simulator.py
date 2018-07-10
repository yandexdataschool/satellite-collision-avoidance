# Module simulator provides simulator of space environment
# and learning proccess of the agent.

import logging
import time

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import pykep as pk
from pykep.orbit_plots import plot_planet


logging.basicConfig(filename="simulator.log", level=logging.DEBUG,
                    filemode='w', format='%(name)s:%(levelname)s\n%(message)s\n')

PAUSE_TIME = 0.0001  # sc
PAUSE_ACTION_TIME = 2  # sc
ARROW_LENGTH = 4e6  # meters
EARTH_RADIUS = 6.3781e6  # meters


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


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


def draw_action(axis, pos, dV):
    """ Draws action.
    Args:
        axis (matplotlib.axes._subplots.Axes3DSubplot): axis to plot on
        pos ([x, y, z]): position of protecred object
        dV ([dVx, dVy, dVz]): action
    """
    x, y, z = pos
    dVx, dVy, dVz = dV

    return axis.quiver(x, y, z, dVx, dVy, dVz, length=ARROW_LENGTH, normalize=True)


def strf_position(satellite, epoch):
    """ Print SpaceObject position at epoch. """
    pos, vel = satellite.position(epoch)
    return "{} position: x - {:0.5f}, y - {:0.5f}, z - {:0.5f}.\
      \n{} velocity: Vx - {:0.5f}, Vy - {:0.5f}, Vz - {:0.5f}\
      ".format(satellite.get_name(), pos[0], pos[1], pos[2],
               satellite.get_name(), vel[0], vel[1], vel[2])


class Visualizer:
    """ Visualizer allows to plot satellite movement simulation
        in real time.
    """

    def __init__(self, curr_time, prob, fuel_cons, traj_dev, foo, reward_components, reward):
        self.fig = plt.figure(figsize=[14, 12])
        self.gs = gridspec.GridSpec(15, 2)
        self.subplot_3d = self.fig.add_subplot(self.gs[:, 0], projection='3d')
        self.subplot_3d.set_aspect("equal")
        self.subplot_p = self.fig.add_subplot(self.gs[:3, 1])
        self.subplot_f = self.fig.add_subplot(self.gs[4:7, 1])
        self.subplot_foo = self.fig.add_subplot(self.gs[8:11, 1])
        self.subplot_r = self.fig.add_subplot(self.gs[12:, 1])
        # initialize data for plots
        self.time_arr = [0]
        self.prob_arr = [prob]
        self.fuel_cons_arr = [fuel_cons]
        self.traj_dev = traj_dev
        self.foo_arr = [foo]
        self.reward_components = reward_components
        self.reward_arr = [reward]
        # initialize zero action
        self.dV_plot = np.zeros(3)

    def run(self):
        plt.ion()

    def update_data(self, curr_time, prob, fuel_cons, traj_dev, foo, reward_components, reward):
        self.time_arr.append(curr_time)
        self.prob_arr.append(prob)
        self.fuel_cons_arr.append(fuel_cons)
        self.traj_dev = traj_dev
        self.foo_arr.append(foo)
        self.reward_components = reward_components
        self.reward_arr.append(reward)

    def plot_planet(self, satellite, t, size, color):
        """ Plot a pykep.planet object. """
        plot_planet(satellite, ax=self.subplot_3d,
                    t0=t, s=size, legend=True, color=color)

    def plot_earth(self):
        """ Add earth to the plot and legend. """
        draw_sphere(self.subplot_3d, (0, 0, 0), EARTH_RADIUS, {
            "color": "b", "lw": 0.5, "alpha": 0.2})
        plt.legend()

    def pause(self, pause_time):
        """ Pause the frame to watch it. """
        plt.legend()
        plt.pause(pause_time)

    def clear(self):
        """ Clear axis for next frame. """
        self.subplot_3d.cla()
        self.subplot_p.cla()
        self.subplot_f.cla()
        self.subplot_foo.cla()
        self.subplot_r.cla()

    def plot_iteration(self, epoch):

        r_coll_prob = self.reward_components["coll_prob"]
        r_fuel = self.reward_components["fuel"]
        r_traj_dev = sum(self.reward_components["traj_dev"])
        s = f"""
Epoch: {epoch}

Collision Probability: {self.prob_arr[-1]:.5}.
Fuel Consumption: {self.fuel_cons_arr[-1]:.5} (|dV|).
Trajectory Deviation:
    a: {self.traj_dev[0]:.5} (m);
    e: {self.traj_dev[1]:.5};
    i: {self.traj_dev[2]:.5} (rad);
    W: {self.traj_dev[3]:.5} (rad);
    w: {self.traj_dev[4]:.5} (rad);
    M: {self.traj_dev[5]:.5} (rad).

Reward Components:
    R Collision Probability: {r_coll_prob:.5};
    R Fuel Consumption: {r_fuel:.5};
    R Trajectory Deviation: {r_traj_dev:.5}.

Total Reward: {self.reward_arr[-1]:.5}
"""
        self.subplot_3d.text2D(-0.3, 0.7, s,
                               transform=self.subplot_3d.transAxes)

    def plot_graphics(self):
        self.make_step_on_graph(self.subplot_p, self.time_arr, self.prob_arr,
                                title='Total collision probability', ylabel='prob')
        self.make_step_on_graph(self.subplot_f, self.time_arr, self.fuel_cons_arr,
                                title='Total fuel consumption', ylabel='fuel (dV)')
        self.make_step_on_graph(self.subplot_foo, self.time_arr, self.foo_arr,
                                title='-----------', ylabel='----------')
        self.make_step_on_graph(self.subplot_r, self.time_arr, self.reward_arr,
                                title='Total reward', ylabel='reward', xlabel='time (mjd2000)')

    def make_step_on_graph(self, ax, time, data, title, ylabel, xlabel=None):
        ax.step(time, data)
        ax.set_title(title)
        ax.grid(True)
        ax.set_ylabel(ylabel)
        if xlabel is not None:
            ax.set_xlabel(xlabel)

    def plot_action(self, pos, time):
        draw_action(self.subplot_3d, pos, self.dV_plot)

        # create bbox for 3d subplot only
        extent = full_extent(self.subplot_3d).transformed(
            self.fig.dpi_scale_trans.inverted())

        self.fig.savefig(f'action_{self.dV_plot}_{time}.png', bbox_inches=extent)
        # set plotted action to zero
        self.dV_plot = np.zeros(3)

    def save_graphics(self):
        fig = plt.figure(figsize=[7, 12])
        gs = gridspec.GridSpec(15, 1)
        subplot_p = fig.add_subplot(gs[:3, 0])
        subplot_f = fig.add_subplot(gs[4:7, 0])
        subplot_foo = fig.add_subplot(gs[8:11, 0])
        subplot_r = fig.add_subplot(gs[12:, 0])

        self.make_step_on_graph(subplot_p, self.time_arr, self.prob_arr,
                                title='Total collision probability', ylabel='prob')
        self.make_step_on_graph(subplot_f, self.time_arr, self.fuel_cons_arr,
                                title='Total fuel consumption', ylabel='fuel (dV)')
        self.make_step_on_graph(subplot_foo, self.time_arr, self.foo_arr,
                                title='----------', ylabel='----------')
        self.make_step_on_graph(subplot_r, self.time_arr, self.reward_arr,
                                title='Total reward', ylabel='reward', xlabel='time since simulation starts (mjd2000)')

        fig.savefig("simulation_graphics.png")


class Simulator:
    """ Simulator allows to start the simulation of provided environment,
        and starts agent-environment collaboration.
    """

    def __init__(self, agent, environment, step=10e-6):
        """
        Args:
            agent (api.Agent): agent, to do actions in environment.
            environment (api.Environment): the initial space environment.
            step (float): propagation time step.
                By default is equal to 0.000001 (0.0864 sc.).

        """

        self.agent = agent
        self.env = environment
        self.start_time = self.env.init_params["start_time"]
        self.end_time = self.env.init_params["end_time"]
        self.curr_time = self.start_time

        self.step = step

        self.vis = None
        self.logger = None

    def run(self, visualize=False, n_steps_vis=1000, log=True, each_step_propagation=False, print_out=False):
        """
        Args:
            visualize (bool): whether show the simulation or not.
            n_steps_vis (int): number of propagation steps in one step of visualization.
            log (bool): whether log the simulation or not.
            each_step_propagation (bool): whether propagate for each step
                or skip the steps using a lower estimation of the time to conjunction.
            print_out (bool): whether show the print out or not.

        Returns:
            reward (float): reward of the session.

        """

        if visualize:
            self.vis = Visualizer(self.curr_time.mjd2000, self.env.get_total_collision_probability(),
                                  self.env.get_fuel_consumption(), self.env.get_trajectory_deviation(),
                                  -999, self.env.get_reward_components(), self.env.get_reward())
            self.vis.run()
            action = np.zeros(4)
            n_steps_since_vis = 1

        if log:
            iteration = 0
            self.logger = logging.getLogger('simulator.Simulator')

        if print_out:
            self.print_start()
            simulation_start_time = time.time()

        while True:
            self.env.propagate_forward(
                self.curr_time.mjd2000, self.step, each_step_propagation)

            if self.curr_time.mjd2000 >= self.env.get_next_action().mjd2000:
                s = self.env.get_state()
                action = self.agent.get_action(s)
                err = self.env.act(action)

                if log:
                    r = self.env.get_reward()
                    if err:
                        self.log_bad_action(err, action)
                    self.log_reward_action(iteration, r, action)

                if visualize and not err:
                    self.vis.dV_plot = action[:3]

            if log:
                self.log_iteration(iteration)
                self.log_protected_position()
                self.log_debris_positions()
                iteration += 1

            if visualize:
                self.plot_protected()
                self.plot_debris()
                self.vis.plot_earth()
                if n_steps_since_vis % n_steps_vis == 0:
                    self.update_vis_data()
                    n_steps_since_vis = 1

                if np.not_equal(self.vis.dV_plot, np.zeros(3)).all():
                    self.vis.plot_action(
                        self.env.protected.position(self.curr_time)[0], self.curr_time)
                    self.vis.pause(PAUSE_ACTION_TIME)

                self.vis.plot_graphics()
                self.vis.pause(PAUSE_TIME)
                self.vis.clear()
                self.vis.plot_iteration(self.curr_time)

            if self.curr_time.mjd2000 >= self.end_time.mjd2000:
                break

            next_action_time = self.env.get_next_action().mjd2000

            if np.isnan(next_action_time) or next_action_time > self.end_time.mjd2000:
                next_time = self.end_time
            else:
                next_time = pk.epoch(next_action_time, "mjd2000")

            if visualize:
                n_steps_to_next_time = int(next_time.mjd2000 / self.step)
                n_steps_to_next_vis = n_steps_vis - n_steps_since_vis
                if n_steps_to_next_time > n_steps_to_next_vis:
                    next_time = pk.epoch(
                        self.curr_time.mjd2000 + n_steps_to_next_vis * self.step, "mjd2000")
                    n_steps_since_vis = n_steps_vis
                else:
                    n_steps_since_vis += n_steps_to_next_time

            self.curr_time = next_time

        if log:
            self.log_protected_position()

        if visualize:
            self.update_vis_data()
            self.vis.save_graphics()

        if print_out:
            simulation_time = time.time() - simulation_start_time
            self.print_end(simulation_time)

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

    def update_vis_data(self):
        self.vis.update_data(
            self.curr_time.mjd2000 - self.start_time.mjd2000,
            self.env.get_total_collision_probability(),
            self.env.get_fuel_consumption(),
            self.env.get_trajectory_deviation(),
            -999,
            self.env.get_reward_components(),
            self.env.get_reward())

    def print_start(self):
        print("Simulation started.\n\nStart time: {} \t End time: {} \t Simulation step:{}\n".format(
            self.start_time.mjd2000, self.end_time.mjd2000, self.step))
        print("Protected SpaceObject:\n{}".format(
            self.env.protected.satellite))
        print("Debris objects:\n")
        for spaceObject in self.env.debris:
            print(spaceObject.satellite)

    def print_end(self, simulation_time):
        traj_dev = self.env.get_trajectory_deviation()
        reward_components = self.env.get_reward_components()
        coll_prob_r = reward_components["coll_prob"]
        fuel_r = reward_components["fuel"]
        traj_dev_r = reward_components["traj_dev"]
        collision_data = self.env.get_collision_data()

        print(f"Simulation ended in {simulation_time:.5} sec.")
        n = len(collision_data)
        if n == 0:
            print("No collisions.")
        else:
            print(f"\n{n} collisions:")
            for i, c in enumerate(collision_data):
                print(f"    #{i+1}: at {c['epoch']} with {c['debris name']};")
                print(f"    distance: {c['distance']:.5}; probability: {c['probability']:.5}.")
        print(f"""
Collision probability: {self.env.get_total_collision_probability():.5}
Fuel consumption: {self.env.get_fuel_consumption():.5}
Trajectory deviation:
    a: {traj_dev[0]:.5};
    e: {traj_dev[1]:.5};
    i: {traj_dev[2]:.5};
    W: {traj_dev[3]:.5};
    w: {traj_dev[4]:.5};
    M: {traj_dev[5]:.5}.

Reward components:
    Collision probability: {coll_prob_r:0.5};
    Fuel consumption: {fuel_r:0.5};
    Trajectory deviation:
        a: {traj_dev_r[0]:.5};
        e: {traj_dev_r[1]:.5};
        i: {traj_dev_r[2]:.5};
        W: {traj_dev_r[3]:.5};
        w: {traj_dev_r[4]:.5};
        M: {traj_dev_r[5]:.5};
        Total: {sum(traj_dev_r):.5}.

Total Reward: {self.env.get_reward():0.5}.
""")
