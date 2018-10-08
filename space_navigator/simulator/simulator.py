# Module simulator provides simulator of space environment
# and learning proccess of the agent.

import logging
import time
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import pykep as pk
from pykep.orbit_plots import plot_planet

from ..agent import TableAgent
from ..utils import is_action_table_empty, action_table2maneuver_table


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
    # items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
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

    def __init__(self, curr_time, prob, fuel_cons, traj_dev, reward_components, reward, curr_alert_info):
        self.fig = plt.figure(figsize=[14, 12])
        self.gs = gridspec.GridSpec(15, 2)
        self.subplot_3d = self.fig.add_subplot(self.gs[:, 0], projection='3d')
        self.subplot_3d.set_aspect("equal")
        self.subplot_p = self.fig.add_subplot(self.gs[:3, 1])
        self.subplot_f = self.fig.add_subplot(self.gs[4:7, 1])
        self.subplot_r_t = self.fig.add_subplot(self.gs[8:11, 1])
        self.subplot_r = self.fig.add_subplot(self.gs[12:, 1])
        # initialize data for plots
        self.time_arr = [0]
        self.prob_arr = [prob]
        self.fuel_cons_arr = [fuel_cons]
        self.traj_dev = traj_dev
        self.reward_components = reward_components
        self.r_traj_dev_arr = [sum(traj_dev)]
        self.reward_arr = [reward]
        self.curr_alert_info = curr_alert_info
        # initialize zero action
        self.dV_plot = np.zeros(3)

    def run(self):
        plt.ion()

    def update_data(self, curr_time, prob, fuel_cons, traj_dev, reward_components, reward, curr_alert_info):
        self.time_arr.append(curr_time)
        self.prob_arr.append(prob)
        self.fuel_cons_arr.append(fuel_cons)
        self.traj_dev = traj_dev
        self.reward_components = reward_components
        self.r_traj_dev_arr.append(sum(reward_components["traj_dev"]))
        self.reward_arr.append(reward)
        self.curr_alert_info = curr_alert_info

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
        self.subplot_r_t.cla()
        self.subplot_r.cla()

    def plot_iteration(self, epoch):

        r_coll_prob = self.reward_components["coll_prob"]
        r_fuel = self.reward_components["fuel"]
        r_traj_dev = sum(self.reward_components["traj_dev"])
        s = f"""Epoch: {epoch}\n
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
        if self.curr_alert_info:
            s_alert = f"""Danger of collision!\n
Object:                   {self.curr_alert_info["debris_name"]};
Probability:              {self.curr_alert_info["probability"]};
Miss distance:            {self.curr_alert_info["distance"]};
Epoch:                    {self.curr_alert_info["epoch"]};
Seconds before collision: {self.curr_alert_info["sec_before_collision"]}.
"""
        else:
            s_alert = "No danger."
        self.subplot_3d.text2D(-0.3, 0.7, s,
                               transform=self.subplot_3d.transAxes)
        self.subplot_3d.text2D(0.4, 1.07, s_alert,
                               transform=self.subplot_3d.transAxes)

    def plot_graphics(self):
        self.make_step_on_graph(self.subplot_p, self.time_arr, self.prob_arr,
                                title='Total collision probability', ylabel='prob')
        self.make_step_on_graph(self.subplot_f, self.time_arr, self.fuel_cons_arr,
                                title='Total fuel consumption', ylabel='fuel (dV)')
        self.make_step_on_graph(self.subplot_r_t, self.time_arr, self.r_traj_dev_arr,
                                title='R Trajectory Deviation', ylabel='reward')
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

        # self.fig.savefig(f'action_{self.dV_plot}_{time}.png', bbox_inches=extent)
        # set plotted action to zero
        self.dV_plot = np.zeros(3)

    def save_graphics(self):
        fig = plt.figure(figsize=[7, 12])
        gs = gridspec.GridSpec(15, 1)
        subplot_p = fig.add_subplot(gs[:3, 0])
        subplot_f = fig.add_subplot(gs[4:7, 0])
        subplot_r_t = fig.add_subplot(gs[8:11, 0])
        subplot_r = fig.add_subplot(gs[12:, 0])

        self.make_step_on_graph(subplot_p, self.time_arr, self.prob_arr,
                                title='Total collision probability', ylabel='prob')
        self.make_step_on_graph(subplot_f, self.time_arr, self.fuel_cons_arr,
                                title='Total fuel consumption', ylabel='fuel (dV)')
        self.make_step_on_graph(subplot_r_t, self.time_arr, self.r_traj_dev_arr,
                                title='R Trajectory Deviation', ylabel='reward')
        self.make_step_on_graph(subplot_r, self.time_arr, self.reward_arr,
                                title='Total reward', ylabel='reward', xlabel='time since simulation starts (mjd2000)')

        fig.savefig("simulation_graphics.png")


class Simulator:
    """ Simulator allows to start the simulation of provided environment,
        and starts agent-environment collaboration.
    """

    def __init__(self, agent, environment, step=1e-6):
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
        self.alerts = None
        self.curr_alert_id = None
        self.curr_alert = None

    def run(self, visualize=False, n_steps_vis=1000, log=True, each_step_propagation=False,
            print_out=False, json_log=False, n_orbits_alert=1.):
        """
        Args:
            visualize (bool): whether show the simulation or not.
            n_steps_vis (int): number of propagation steps in one step of visualization.
            log (bool): whether log the simulation or not.
            each_step_propagation (bool): whether propagate for each step
                or skip the steps using a lower estimation of the time to conjunction.
            print_out (bool): whether show the print out or not.
            json_log (bool): whether log the simulation to file or not.
            n_orbits_alert (float or None): number of orbits before collision emergency alert,
                or None for not emergency alert; used for visualisation, json_log, log;
                (should correlate with agent training methods).

        Returns:
            reward (float): reward of the session.

        """
        if not (visualize or log or json_log) or n_orbits_alert == 0:
            n_orbits_alert = None

        if n_orbits_alert is not None:
            # TODO: move this part to utils
            if print_out:
                print("\nPreprocessing started.")
            env_temp = self.env.copy()
            orbit_time = env_temp.protected.get_orbital_period()
            warning_time = orbit_time * n_orbits_alert
            end_time_shifted = pk.epoch(
                self.end_time.mjd2000 + warning_time, "mjd2000")
            env_temp.set_end_time(end_time_shifted)
            assert self.env.get_end_time().mjd2000 == self.end_time.mjd2000

            # collision data without maneuvers
            env_temp.reset()
            agent = TableAgent()
            simulator_wo_man = Simulator(agent, env_temp, self.step)
            simulator_wo_man.run(visualize=False, log=False, each_step_propagation=False,
                                 print_out=False, json_log=False, n_orbits_alert=None)
            collision_data_wo_man = env_temp.collision_data()

            # collision data with maneuvers
            env_temp.reset()
            agent = self.agent
            simulator_with_man = Simulator(agent, env_temp, self.step)
            simulator_with_man.run(visualize=False, log=False, each_step_propagation=False,
                                   print_out=False, json_log=False, n_orbits_alert=None)
            collision_data_with_man = env_temp.collision_data()

            # alert data
            maneuvers = action_table2maneuver_table(
                agent.get_action_table(), self.start_time)
            maneuvers_epochs = maneuvers[:, 3]
            are_maneuvers = maneuvers.size != 0
            # TODO: expand to many maneuvers
            self.alerts = []
            for coll in collision_data_wo_man:
                if are_maneuvers and coll["epoch"] > maneuvers_epochs[0] + warning_time:
                    break
                start_alert_epoch = max(
                    coll["epoch"] - warning_time, 0)
                end_alert_epoch = coll["epoch"]
                if are_maneuvers:
                    end_alert_epoch = min(end_alert_epoch, maneuvers_epochs[0])
                coll["start_alert_epoch"] = start_alert_epoch
                coll["end_alert_epoch"] = end_alert_epoch
                assert end_alert_epoch >= start_alert_epoch
                self.alerts.append(coll)
            if not are_maneuvers:
                assert len(self.alerts) == len(collision_data_wo_man)
            if are_maneuvers:
                for coll in collision_data_wo_man:
                    if coll["epoch"] >= maneuvers_epochs[0] + warning_time:
                        start_alert_epoch = max(
                            coll["epoch"] - warning_time, 0)
                        # TODO: min(coll epoch, next man)
                        end_alert_epoch = coll["epoch"]
                        coll["start_alert_epoch"] = start_alert_epoch
                        coll["end_alert_epoch"] = end_alert_epoch
                        self.alerts.append(coll)

            assert len(self.alerts) >= max(
                len(collision_data_wo_man), len(collision_data_with_man))

            self.curr_alert_id = 0
            self.curr_alert = self.curr_alert_info()

            if print_out:
                print("Preprocessing ended.\n")

        if visualize:

            self.vis = Visualizer(self.curr_time.mjd2000, self.env.get_total_collision_probability(),
                                  self.env.get_fuel_consumption(), self.env.get_trajectory_deviation(),
                                  self.env.get_reward_components(), self.env.get_reward(), self.curr_alert)
            self.vis.run()
            action = np.zeros(4)
            n_steps_since_vis = 1

        if log:
            iteration = 0
            self.logger = logging.getLogger('simulator.Simulator')

        if json_log:
            json_log_iter = 0
            self.log_json(json_log_iter, start=True)

        if print_out:
            self.print_start()
            simulation_start_time = time.time()

        while True:
            self.env.propagate_forward(
                self.curr_time.mjd2000, self.step, each_step_propagation)

            if self.curr_time.mjd2000 >= self.env.get_next_action().mjd2000:
                s = self.env.get_state()
                action = self.agent.get_action(s)
                # TODO: assert: no actions without alert
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

                self.vis.plot_iteration(self.curr_time)
                self.vis.plot_graphics()
                if np.not_equal(self.vis.dV_plot, np.zeros(3)).all():
                    self.vis.plot_action(
                        self.env.protected.position(self.curr_time)[0], self.curr_time)
                    self.vis.pause(PAUSE_ACTION_TIME)
                else:
                    self.vis.pause(PAUSE_TIME)
                self.vis.clear()

            if json_log:
                self.log_json(json_log_iter)
                json_log_iter += 1

            if self.curr_time.mjd2000 >= self.end_time.mjd2000:
                break

            next_action_time = self.env.get_next_action().mjd2000

            if json_log:
                next_time = pk.epoch(
                    self.curr_time.mjd2000 + self.step, "mjd2000")
            elif np.isnan(next_action_time) or next_action_time > self.end_time.mjd2000:
                next_time = self.end_time
            else:
                next_time = pk.epoch(next_action_time, "mjd2000")

            if visualize and not json_log:
                n_steps_to_next_time = int(next_time.mjd2000 / self.step)
                n_steps_to_next_vis = n_steps_vis - n_steps_since_vis
                if n_steps_to_next_time > n_steps_to_next_vis:
                    next_time = pk.epoch(
                        self.curr_time.mjd2000 + n_steps_to_next_vis * self.step, "mjd2000")
                    n_steps_since_vis = n_steps_vis
                else:
                    n_steps_since_vis += n_steps_to_next_time

            if n_orbits_alert is not None:
                self.curr_alert = self.curr_alert_info()

            self.curr_time = next_time

        if log:
            self.log_protected_position()

        if visualize:
            self.update_vis_data()
            self.vis.save_graphics()

        if print_out:
            simulation_time = time.time() - simulation_start_time
            self.print_end(simulation_time)

        if json_log:
            self.log_json(json_log_iter, end=True)

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

    def log_json(self, id, start=False, end=False):
        json_log_path = "vr/json_log.json"
        if start:
            with open(json_log_path, "w") as f:
                f.write("{")
        else:
            point = {
                "time_mjd2000": self.curr_time.mjd2000,
                "epoch": str(self.curr_time),
                "protected_pos": list(self.env.protected.position(self.curr_time)[0]),
            }
            debris_pos = []
            for d in self.env.debris:
                debris_pos.append(list(d.position(self.curr_time)[0]))
            point["debris_pos"] = debris_pos
            if self.alerts is not None:
                point["alert"] = {
                    "is_alert": len(self.curr_alert) != 0,
                    "info": self.curr_alert,
                }
            with open(json_log_path, "a") as f:
                f.write(f"\"{id}\": ")
                json.dump(point, f)
        with open(json_log_path, "a") as f:
            if end:
                f.write("}")
            elif not start:
                f.write(", ")

    def curr_alert_info(self):
        curr_epoch = self.curr_time.mjd2000
        # TODO: all alerts info, not just about closest one
        while self.curr_alert_id < len(self.alerts):
            if self.alerts[self.curr_alert_id]["start_alert_epoch"] <= curr_epoch:
                if self.alerts[self.curr_alert_id]["end_alert_epoch"] < curr_epoch:
                    self.curr_alert_id += 1
                else:
                    info = self.alerts[self.curr_alert_id].copy()
                    info.pop("start_alert_epoch")
                    info.pop("end_alert_epoch")
                    info["probability"] = round(info["probability"], 8)
                    info["distance"] = round(info["distance"], 3)
                    info["sec_before_collision"] = round(
                        86400 * (info["epoch"] - self.curr_time.mjd2000), 1)
                    info["epoch"] = round(info["epoch"], 5)
                    info["debris_id"] = str(info["debris_id"])
                    return info
            else:
                break
        return {}

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
            self.env.get_reward_components(),
            self.env.get_reward(),
            self.curr_alert)

    def print_start(self):
        print("Simulation started.\n\nStart time: {} \t End time: {} \t Simulation step:{}\n".format(
            self.start_time.mjd2000, self.end_time.mjd2000, self.step))
        print("Protected SpaceObject:\n{}".format(
            self.env.protected.satellite))
        print("Debris objects:\n")
        for spaceObject in self.env.debris:
            print(spaceObject.satellite)

    def print_end(self, simulation_time):
        # TODO: if alert, do not copy the code
        # data
        coll_prob_thr = self.env.coll_prob_thr
        fuel_cons_thr = self.env.fuel_cons_thr
        traj_dev_thr = self.env.traj_dev_thr
        action_table = self.agent.action_table
        action_table_not_empty = not is_action_table_empty(action_table)
        crit_distance = self.env.crit_distance

        coll_prob = self.env.get_total_collision_probability()
        fuel_cons = self.env.get_fuel_consumption()
        traj_dev = self.env.get_trajectory_deviation()
        total_reward = self.env.get_reward()
        reward_components = self.env.get_reward_components()
        coll_prob_r = reward_components["coll_prob"]
        fuel_r = reward_components["fuel"]
        traj_dev_r = reward_components["traj_dev"]
        collision_data = self.env.collision_data()

        # w/o maneuvers
        if action_table_not_empty:
            agent = TableAgent()
            env_wo = self.env.copy()
            env_wo.reset()
            simulator_wo = Simulator(agent, env_wo, self.step)
            simulator_wo.run(visualize=False, n_steps_vis=1000,
                             log=False, each_step_propagation=False, print_out=False)

            coll_prob_wo = env_wo.get_total_collision_probability()
            fuel_cons_wo = env_wo.get_fuel_consumption()
            traj_dev_wo = env_wo.get_trajectory_deviation()
            total_reward_wo = env_wo.get_reward()
            reward_components_wo = env_wo.get_reward_components()
            coll_prob_r_wo = reward_components_wo["coll_prob"]
            fuel_r_wo = reward_components_wo["fuel"]
            traj_dev_r_wo = reward_components_wo["traj_dev"]
            collision_data_wo = env_wo.collision_data()
        else:
            coll_prob_wo = coll_prob
            fuel_cons_wo = fuel_cons
            traj_dev_wo = traj_dev
            total_reward_wo = total_reward
            reward_components_wo = reward_components
            coll_prob_r_wo = coll_prob_r
            fuel_r_wo = fuel_r
            traj_dev_r_wo = traj_dev_r
            collision_data_wo = collision_data

        # simulation time
        print(f"Simulation ended in {simulation_time:.5} sec.")

        # maneuvers
        print("\nManeuvers table:")
        if action_table_not_empty:
            maneuvers = action_table2maneuver_table(
                action_table, self.start_time)
            columns = ["dVx (m^2/s)", "dVy (m^2/s)",
                       "dVz (m^2/s)", "time (mjd2000)"]
            df = pd.DataFrame(maneuvers,
                              index=range(1, maneuvers.shape[0] + 1),
                              columns=columns)
            print(df)
        else:
            print("no maneuvers.")

        # collisions
        print(f"\nCollisions (distance <= {crit_distance} meters):")
        n = len(collision_data_wo)
        if n > 0:
            print(f"    without maneuvers (total number: {n}):")
            for i, c in enumerate(collision_data_wo):
                print(f"        #{i+1}: at {c['epoch']} with {c['debris_name']};")
                print(f"        distance: {c['distance']:.5}; probability: {c['probability']:.5}.")
        else:
            print("    no collisions without maneuvers.")
        if action_table_not_empty:
            n = len(collision_data)
            if n > 0:
                print(f"    with maneuvers (total number: {n}):")
                for i, c in enumerate(collision_data):
                    print(f"        #{i+1}: at {c['epoch']} with {c['debris_name']};")
                    print(f"        distance: {c['distance']:.5}; probability: {c['probability']:.5}.")
            else:
                print("    no collisions with maneuvers.")

        # total reward
        print("\nTotal Reward:")
        print(f"    without maneuvers: {total_reward_wo}.")
        if action_table_not_empty:
            print(f"    with maneuvers: {total_reward}.")

        # table of significant parameters
        print("\nParameters table:")
        columns = ["threshold", "value w/o man", "reward w/o man"]
        if action_table_not_empty:
            columns += ["value with man", "reward with man"]
        index = [
            "coll prob", "fuel (|dV|)",
            "dev a (m)", "dev e", "dev i (rad)",
            "dev W (rad)", "dev w (rad)", "dev M (rad)",
        ]
        df = pd.DataFrame(index=index, columns=columns)
        df["threshold"] = [coll_prob_thr, fuel_cons_thr] + list(traj_dev_thr)
        df["threshold"].fillna(value="not taken", inplace=True)
        df["value w/o man"] = [coll_prob_wo,
                               fuel_cons_wo] + list(traj_dev_wo)
        df["reward w/o man"] = [coll_prob_r_wo,
                                fuel_r_wo] + list(traj_dev_r_wo)
        if action_table_not_empty:
            df["value with man"] = [coll_prob, fuel_cons] + list(traj_dev)
            df["reward with man"] = [coll_prob_r,
                                     fuel_r] + list(traj_dev_r)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
