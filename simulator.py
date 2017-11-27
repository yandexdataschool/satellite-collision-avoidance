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

    def __init__(self, agent, environment, step=0.01):
        """
            agent -- Agent(), agent, to do actions in environment.
            environment -- Environment(), the initial space environment.
            step -- float, step in julian date.
        """
        self.is_end = False
        self.agent = agent
        self.environment = environment
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

            r = self.environment.get_reward()
            s = self.environment.get_state(PARAMS)
            action = self.agent.get_action(s, r)
            self.environment.act(action)

            self.is_end = self.environment.state.is_end

            print("Iter #{} \tEpoch: {}\tCollision: {}".format(
                iteration,  self.curr_time, self.is_end))

            print_position(self.environment.protected, self.curr_time)
            for obj in self.environment.debris:
                print_position(obj, self.curr_time)

            self.curr_time = pk.epoch(
                self.curr_time.mjd2000 + self.step, "mjd2000")

            # Plot Protected SpaceObject
            plot_planet(self.environment.protected.satellite,
                        ax=ax, t0=self.curr_time, s=100, legend=True)
            
            # Plot space debris
            cmap = plt.get_cmap('gnuplot')
            N = len(self.environment.debris)
            colors = [cmap(i) for i in np.linspace(0, 1, N)]
            for i in range(N):
                plot_planet(self.environment.debris[i].satellite, ax=ax,
                            t0=self.curr_time, s=10, legend=True, color=colors[i])
            # pause, to see the figure.
            plt.pause(0.5)
            plt.cla()


def main():
    sattelites = read_tle_satellites("stations.txt")
    print(len(sattelites))
    # ISS - first row in the file, our protected object. Other satellites - space debris.
    ISS, debris = sattelites[0], sattelites[1:5]

    # Example of SpaceObject with initial parameters: pos, v, epoch.
    pos, v = [1, 0, 1], [1, 0, 1]
    t = time.strftime("%Y-%m-%d %T")
    epoch = pk.epoch_from_string(t)
    mu, f = 1.0, 1.0
    d1 = SpaceObject("Debris 1", False, dict(
        pos=pos, v=v, epoch=epoch, mu=mu, f=f))
    debris.append(d1)

    agent = Agent()
    env = Environment(ISS, debris)

    simulator = Simulator(agent, env)
    simulator.run()


if __name__ == "__main__":
    main()
