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
SIMULATION_STEP = 0.0001
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
    parser.add_argument("-u", "--update_r_p_step", type=int,
                        default=20, required=False)
    parser.add_argument("-p", "--print_out", type=str,
                        default="False", required=False)

    args = parser.parse_args(args)

    visualize = args.visualize.lower() == "true"
    print_out = args.print_out.lower() == "true"
    start_time, end_time = args.start_time, args.end_time
    step, update_r_p_step = args.step, args.update_r_p_step

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

    simulator = Simulator(
        agent, env, update_r_p_step=update_r_p_step, print_out=print_out)
    simulator.run(step=step, visualize=visualize)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
