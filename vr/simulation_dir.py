# Collision runs simulator for all action tables in directory.

import argparse
import sys
import os

from space_navigator.api import Environment
from space_navigator.simulator import Simulator
from space_navigator.agent import TableAgent, PytorchAgent
from space_navigator.utils import read_environment, get_agent

PROPAGATION_STEP = 0.000001


def main(args):
    parser = argparse.ArgumentParser()

    # path args
    parser.add_argument("-models_dir", "--models_dir_path", type=str,
                        required=True)
    parser.add_argument("-json_dir", "--json_log_dir_path", type=str,
                        default="vr/examples/json_log", required=False)

    # simulator initialization
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
                        default="False", required=False)

    args = parser.parse_args(args)

    # simulator initialization
    env_path = args.environment
    step = args.step
    agent_type = args.agent_type

    # simulator run args
    log = args.logging.lower() == "true"
    print_out = args.print_out.lower() == "true"

    # path args
    models_dir_path = args.models_dir_path
    json_log_dir_path = args.json_log_dir_path
    if not os.path.exists(json_log_dir_path):
        os.makedirs(json_log_dir_path)

    # models
    models = {}
    for file in os.listdir(models_dir_path):
        if file.endswith("_act.csv"):
            name = file[:-8]
            path = os.path.join(models_dir_path, file)
            models[name] = path

    # simulation
    env = read_environment(env_path)
    for name, path in models.items():
        json_log_path = os.path.join(json_log_dir_path, name + ".json")
        print(f"\nModel: {name}\npath: {path}\njson path: {json_log_path}")
        agent = get_agent(agent_type, path)
        simulator = Simulator(
            agent=agent, environment=env, step=step)
        simulator.run(visualize=False, n_steps_vis=None, log=log,
                      each_step_propagation=True, print_out=print_out,
                      json_log=True, json_log_path=json_log_path)
        env.reset()

    return


if __name__ == "__main__":
    main(sys.argv[1:])
