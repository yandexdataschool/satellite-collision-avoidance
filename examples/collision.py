# Collision runs simulator with two objects, having collision
# at time 6600.

import argparse
import sys

import numpy as np
import pykep as pk

from space_navigator.api import Environment
from space_navigator.simulator import Simulator
from space_navigator.agent import TableAgent, PytorchAgent
from space_navigator.utils import read_environment, get_agent

SIMULATION_STEP = 0.0001


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model_path", type=str,
                        default=None, required=False)
    parser.add_argument("-v", "--visualize", type=str,
                        default="True", required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=SIMULATION_STEP, required=False)
    parser.add_argument("-u", "--update_r_p_step", type=int,
                        default=20, required=False)
    parser.add_argument("-print", "--print_out", type=str,
                        default="True", required=False)
    parser.add_argument("-env", "--environment", type=str,
                        default="data/environments/collision.env", required=False)
    parser.add_argument("-a", "--agent_type", type=str,
                        default="table", required=False)

    args = parser.parse_args(args)

    visualize = args.visualize.lower() == "true"
    print_out = args.print_out.lower() == "true"
    step, update_r_p_step = args.step, args.update_r_p_step

    env_path = args.environment
    env = read_environment(env_path)

    model_path = args.model_path
    agent_type = args.agent_type

    agent = get_agent(agent_type, model_path)

    simulator = Simulator(
        agent, env, step=step, update_r_p_step=update_r_p_step)
    simulator.run(visualize=visualize, print_out=print_out)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
