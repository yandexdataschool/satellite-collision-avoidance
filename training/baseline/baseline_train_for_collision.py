# Train agent for collision at time 6600.

import argparse
import sys

from space_navigator.utils import read_environment
from space_navigator.models.baseline import Baseline

PROPAGATION_STEP = 0.000001


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_s", "--n_samples", type=int,
                        default=100, required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=PROPAGATION_STEP, required=False)
    parser.add_argument("-save_path", "--save_action_table_path", type=str,
                        default="training/agents_tables/Baseline/action_table_baseline.csv", required=False)
    parser.add_argument("-print", "--print_out", type=str,
                        default="False", required=False)
    parser.add_argument("-env", "--environment", type=str,
                        default="data/environments/collision.env", required=False)

    args = parser.parse_args(args)

    n_samples = args.n_samples
    step = args.step
    save_action_table_path = args.save_action_table_path
    print_out = args.print_out.lower() == "true"
    env_path = args.environment

    # create environment
    env = read_environment(env_path)

    # Baseline
    action_table = Baseline(env, step)
    action_table.train(n_samples, print_out)
    action_table.save_action_table(save_action_table_path)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
