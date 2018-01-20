# Example 1 runs simulator with provided parameteres.
# We use data from https://www.celestrak.com/NORAD/elements/stations.txt
# to provide ISS - protected object and other stellites as debris.

import argparse
import sys
import time

from simulator import Simulator, read_space_objects
from api import Agent, Environment

import pykep as pk

# Number of TLE satellites to read from file.
DEBRIS_NUM = 3


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", type=str,
                        default="True", required=False)
    parser.add_argument("-t", "--end_time", type=float,
                        default=6000.01, required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=0.001, required=False)
    args = parser.parse_args(args)

    visualize = args.visualize.lower() == "true"
    end_time, step = args.end_time, args.step

    # SpaceObjects with TLE initial parameters.
    satellites = read_space_objects("data/stations.tle", "tle")
    # ISS - first row in the file, our protected object. Other satellites -
    # space debris.
    iss, debris = satellites[0], satellites[1: 1 + DEBRIS_NUM]

    # SpaceObjects with "eph" initial parameters: pos, v, epoch.
    eph = read_space_objects("data/space_objects.eph", "eph")
    for obj in eph:
        debris.append(obj)

    # SpaceObjects with "osc" initial parameteres: 6 orbital
    # elements and epoch.
    osc = read_space_objects("data/space_objects.osc", "osc")
    for obj in osc:
        debris.append(obj)
    agent = Agent()
    start_time = pk.epoch(6000)
    env = Environment(iss, debris, start_time)

    simulator = Simulator(agent, env, print_out=False)
    simulator.run(end_time=end_time, step=step, visualize=visualize)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
