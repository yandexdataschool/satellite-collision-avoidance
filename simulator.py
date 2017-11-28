# Module simulator provides simulator of space environment
# and learning proccess of the agent.

import time
import argparse
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

import pykep as pk
from pykep.orbit_plots import plot_planet

from api import Agent, Environment, SpaceObject

logging.basicConfig(filename="simulator.log", level=logging.DEBUG,
                    filemode='w', format='%(name)s:%(levelname)s\n%(message)s\n')

DEBRIS_NUM = 5


def strf_position(satellite, epoch):
    """ Print SpaceObject position. """
    pos, v = satellite.position(epoch)
    return "{} position: x - {:0.2f}, y - {:0.2f}, z - {:0.2f}.\
      \n{} velocity: Vx - {:0.2f}, Vy - {:0.2f}, Vz - {:0.2f}\
      ".format(satellite.get_name(), pos[0], pos[1], pos[2],
               satellite.get_name(), v[0], v[1], v[2])


def read_tle_satellites(f):
    """Create SpaceObjects from a text file f."""
    space_objects = []
    with open(f, 'r') as satellites:
        while True:
            name = satellites.readline().strip()
            if not name:
                break
            tle_line1 = satellites.readline().strip()
            tle_line2 = satellites.readline().strip()
            satellite = SpaceObject(name, True, dict(tle_line1=tle_line1,
                                                     tle_line2=tle_line2,
                                                     fuel=1))
            space_objects.append(satellite)
    return space_objects


class Vizualizer:
    """ Vizualizer allows to plot satellite movement simulation
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

    def pause_and_clear(self):
        """ Pause the frame to watch it. Clear axis for next frame. """
        plt.pause(0.005)
        plt.cla()


class Simulator:
    """ Simulator allows to start the simulation of provided environment,
    and starts agent-environment collaboration.
    """

    def __init__(self, agent, environment, start_time=None):
        """
            agent -- Agent(), agent, to do actions in environment.
            environment -- Environment(), the initial space environment.
            step -- float, step in julian date.
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

        self.viz = Vizualizer()
        self.logger = logging.getLogger('simulator.Simulator')

    def run(self, vizualize=True, N=None, step=1):
        iteration = 0

        if vizualize:
            self.viz.run()

        while iteration != N and not self.is_end:
            self.is_end, s = self.env.get_state(self.curr_time)
            r = self.env.get_reward(s, self.env.get_curr_reward())
            action = self.agent.get_action(s, r)
            self.env.act(action)

            self.log_iteration(iteration)
            self.log_protected_position()
            self.log_debris_positions()

            if vizualize:
                self.plot_protected()
                self.plot_debris()
                self.viz.pause_and_clear()

            self.curr_time = pk.epoch(
                self.curr_time.mjd2000 + step, "mjd2000")

            iteration += 1

        print("Simulation ended. Collision: {}".format(self.is_end))

    def log_protected_position(self):
        self.logger.info(strf_position(self.env.protected, self.curr_time))

    def log_debris_positions(self):
        for obj in self.env.debris:
            self.logger.info(strf_position(obj, self.curr_time))

    def log_iteration(self, iteration):
        self.logger.debug("Iter #{} \tEpoch: {}\tCollision: {}\t Reward: {}".format(
            iteration,  self.curr_time, self.is_end, self.env.get_curr_reward()))

    def plot_protected(self):
        """ Plot Protected SpaceObject. """
        self.viz.plot_planet(self.env.protected.satellite,
                             t=self.curr_time, size=100, color="black")

    def plot_debris(self):
        """ Plot space debris. """
        cmap = plt.get_cmap('gist_rainbow')
        N = len(self.env.debris)
        colors = [cmap(i) for i in np.linspace(0, 1, N)]
        for i in range(N):
            self.viz.plot_planet(
                self.env.debris[i].satellite, t=self.curr_time,
                size=25, color=colors[i])


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vizualize", type=str,
                        default="True", required=False)
    parser.add_argument("-N", "--N_iteration", type=int,
                        default=None, required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=1, required=False)
    args = parser.parse_args(args)

    vizualize = args.vizualize.lower() == "true"
    N, step = args.N_iteration, args.step

    sattelites = read_tle_satellites("stations.txt")
    # ISS - first row in the file, our protected object. Other satellites -
    # space debris.
    ISS, debris = sattelites[0], sattelites[1:DEBRIS_NUM]

    # Example of SpaceObject with initial parameters: pos, v, epoch.
    pos, v = [2315921.25, 3814078.37, 5096751.46], [4363.18, 1981.83, 5982.45]
    epoch = pk.epoch_from_string("2017-Nov-27 15:16:20")
    mu, fuel = 398600800000000, 1.0
    d1 = SpaceObject("Debris 1", False, dict(
        pos=pos, v=v, epoch=epoch, mu=mu, fuel=fuel))
    debris.append(d1)

    agent = Agent()
    env = Environment(ISS, debris)

    simulator = Simulator(agent, env)
    simulator.run(vizualize=vizualize, N=N, step=step)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
