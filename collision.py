# Example 1 runs simulator with provided parameteres.
# We use data from https://www.celestrak.com/NORAD/elements/stations.txt
# to provide ISS - protected object and other stellites as debris.

import argparse
import sys

from simulator import Simulator, read_space_objects
from api import Agent, Environment

import pykep as pk



def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", type=str,
                        default="True", required=False)
    parser.add_argument("-t", "--end_time", type=float,
                        default=6600.1, required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=0.1, required=False)
    args = parser.parse_args(args)

    visualize = args.visualize.lower() == "true"
    end_time, step = args.end_time, args.step

    osc = read_space_objects("data/collision.osc", "osc")
    protected = osc[0]
    debris = [osc[1]]

    agent = Agent()
    start_time = pk.epoch(6599.9)
    env = Environment(protected, debris, start_time)

    simulator = Simulator(agent, env, print_out=False)
    simulator.run(end_time=end_time, step=step, visualize=visualize)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
