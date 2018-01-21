# Collision runs simulator with two objects, having collision
# at time 6600.

import argparse
import sys

from simulator import Simulator, read_space_objects
from api import Agent, Environment

import pykep as pk

START_TIME = 6599.95
SIMULATION_STEP = 0.01
END_TIME = 6600.05

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
    args = parser.parse_args(args)

    visualize = args.visualize.lower() == "true"
    start_time, end_time, step = args.start_time, args.end_time, args.step

    osc = read_space_objects("data/collision.osc", "osc")
    protected = osc[0]
    debris = [osc[1]]

    agent = Agent()
    start_time = pk.epoch(start_time, "mjd2000")
    env = Environment(protected, debris, start_time)

    simulator = Simulator(agent, env, print_out=False)
    simulator.run(end_time=end_time, step=step, visualize=visualize)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
