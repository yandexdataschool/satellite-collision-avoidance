# Example 1 runs simulator with provided parameteres.
# We use data from https://www.celestrak.com/NORAD/elements/stations.txt
# to provide ISS - protected object and other stellites as debris.

import argparse
import sys

import pykep as pk


from space_navigator.simulator import Simulator
from space_navigator.api import Environment
from space_navigator.agent import TableAgent
from space_navigator.utils import read_space_objects

START_TIME = 6000
SIMULATION_STEP = 0.000001
END_TIME = 6000.01
# Number of TLE satellites to read from file.
DEBRIS_NUM = 3


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", type=str,
                        default="True", required=False)
    parser.add_argument("-start", "--start_time", type=float,
                        default=START_TIME, required=False)
    parser.add_argument("-end", "--end_time", type=float,
                        default=END_TIME, required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=SIMULATION_STEP, required=False)
    parser.add_argument("-n_v", "--n_steps_vis", type=int,
                        default=1000, required=False,
                        help="the number of propagation steps in one step of visualization")
    parser.add_argument("-p", "--print_out", type=str,
                        default="False", required=False)

    args = parser.parse_args(args)

    visualize = args.visualize.lower() == "true"
    print_out = args.print_out.lower() == "true"
    start_time, end_time = args.start_time, args.end_time
    step, n_steps_vis = args.step, args.n_steps_vis

    # SpaceObjects with TLE initial parameters.
    satellites = read_space_objects(
        "data/environments/stations.tle", "tle")
    # ISS - first row in the file, our protected object. Other satellites -
    # space debris.
    iss, debris = satellites[0], satellites[1: 1 + DEBRIS_NUM]

    # SpaceObjects with "eph" initial parameters: pos, v, epoch.
    eph = read_space_objects("data/environments/space_objects.eph", "eph")
    for obj in eph:
        debris.append(obj)

    # SpaceObjects with "osc" initial parameteres: 6 orbital
    # elements and epoch.
    osc = read_space_objects("data/environments/space_objects.osc", "osc")
    for obj in osc:
        debris.append(obj)

    agent = TableAgent()

    start_time = pk.epoch(start_time, "mjd2000")
    end_time = pk.epoch(end_time, "mjd2000")
    env = Environment(iss, debris, start_time, end_time)

    simulator = Simulator(agent, env, step)
    simulator.run(visualize=visualize,
                  n_steps_vis=n_steps_vis, print_out=print_out)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
