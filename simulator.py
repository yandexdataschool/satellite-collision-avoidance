# Module simulator provides simulator of space environment
# and learning proccess of the agent.

import numpy as np
import pykep as pk
from pykep.orbit_plots import plot_planet
import time
from api import Agent, Environment, SpaceObject
import matplotlib.pyplot as plt

PARAMS = dict(coord=True, v=True)


def print_position(satellite, epoch):
    """ Print SpaceObject position. """
    pos, v = satellite.position(epoch)
    print("{} position: x - {:0.2f}, y - {:0.2f}, z - {:0.2f}.\
      \n{} velocity: Vx - {:0.2f}, Vy - {:0.2f}, Vz - {:0.2f}\
      \n".format(satellite.get_name(), pos[0], pos[1], pos[2],
                 satellite.get_name(), v[0], v[1], v[2]))


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
                                                     f=0))
            space_objects.append(satellite)
    return space_objects


class Simulator:
    """ Simulator allows to start the simulation of provided environment,
    and starts agent-environment collaboration.
    """

    def __init__(self, agent, environment, step=0.001):
        """
            agent -- Agent(), agent, to do actions in environment.
            environment -- Environment(), the initial space environment.
            step -- float, step in julian date.
        """
        self.is_end = False
        self.agent = agent
        self.env = environment
        self.step = step
        self.start_time = pk.epoch_from_string(time.strftime("%Y-%m-%d %T"))
        self.curr_time = self.start_time

    def run(self):
        iteration = 0
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.ion()
        plt.show()

        while not self.is_end:
            iteration += 1

            r = self.env.get_reward()
            s = self.env.get_state(PARAMS)
            action = self.agent.get_action(s, r)
            self.env.act(action)

            self.is_end = self.env.state.is_end

            print("Iter #{} \tEpoch: {}\tCollision: {}".format(
                iteration,  self.curr_time, self.is_end))

            print_position(self.env.protected, self.curr_time)
            for obj in self.env.debris:
                print_position(obj, self.curr_time)

            self.curr_time = pk.epoch(
                self.curr_time.mjd2000 + self.step, "mjd2000")

            self.plot_protected(ax)
            self.plot_debris(ax)
            # pause, to see the figure.
            plt.pause(0.05)
            plt.cla()

    def plot_protected(self, ax):
        """ Plot Protected SpaceObject. """
        plot_planet(self.env.protected.satellite,
                    ax=ax, t0=self.curr_time, s=100, legend=True)

    def plot_debris(self, ax):
        """ Plot space debris. """
        cmap = plt.get_cmap('gnuplot')
        N = len(self.env.debris)
        colors = [cmap(i) for i in np.linspace(0, 1, N)]
        for i in range(N):
            plot_planet(self.env.debris[i].satellite, ax=ax,
                        t0=self.curr_time, s=25, legend=True, color=colors[i])


def main():
    sattelites = read_tle_satellites("stations.txt")
    # ISS - first row in the file, our protected object. Other satellites -
    # space debris.
    ISS, debris = sattelites[0], sattelites[1:4]

    # Example of SpaceObject with initial parameters: pos, v, epoch.
    pos, v = [2315921.25, 3814078.37, 5096751.46], [4363.18, 1981.83, 5982.45]
    epoch = pk.epoch_from_string("2017-Nov-27 15:16:20")
    mu, f = 398600800000000, 1.0
    d1 = SpaceObject("Debris 1", False, dict(
        pos=pos, v=v, epoch=epoch, mu=mu, f=f))
    debris.append(d1)

    agent = Agent()
    env = Environment(ISS, debris)

    simulator = Simulator(agent, env)
    simulator.run()


if __name__ == "__main__":
    main()
