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

PROPAGATION_STEP = 0.000001


def main(args):
    parser = argparse.ArgumentParser()

    # simulator initialization
    parser.add_argument("-model", "--model_path", type=str,
                        default=None, required=False)
    parser.add_argument("-env", "--environment", type=str,
                        default="data/environments/collision.env", required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=PROPAGATION_STEP, required=False)
    parser.add_argument("-a", "--agent_type", type=str,
                        default="table", required=False)

    # simulator run args
    parser.add_argument("-log", "--logging", type=str,
                        default="True", required=False)
    parser.add_argument("-print", "--print_out", type=str,
                        default="True", required=False)

    args = parser.parse_args(args)

    # simulator initialization
    model_path = args.model_path
    env_path = args.environment
    step = args.step
    agent_type = args.agent_type

    # simulator run args
    log = args.logging.lower() == "true"
    print_out = args.print_out.lower() == "true"

    # simulation
    env = read_environment(env_path)
    agent = get_agent(agent_type, model_path)
    simulator = Simulator(
        agent=agent, environment=env, step=step)
    simulator.run(visualize=False, n_steps_vis=None, log=log,
                  each_step_propagation=True, print_out=print_out,
                  json_log=True)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
