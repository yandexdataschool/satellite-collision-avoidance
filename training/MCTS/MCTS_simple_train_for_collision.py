# Train agent for collision at time 6600.

import argparse
import sys

from space_navigator.utils import read_environment
from space_navigator.models.MCTS import DecisionTree

PROPAGATION_STEP = 0.000001


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_i", "--n_iterations", type=int,
                        default=200, required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=PROPAGATION_STEP, required=False)
    parser.add_argument("-save_path", "--save_action_table_path", type=str,
                        default="training/agents_tables/MCTS/action_table_MCTS_simple.csv", required=False)
    parser.add_argument("-print", "--print_out", type=str,
                        default="False", required=False)
    parser.add_argument("-env", "--environment", type=str,
                        default="data/environments/collision.osc", required=False)

    args = parser.parse_args(args)

    n_iterations = args.n_iterations
    step = args.step
    save_action_table_path = args.save_action_table_path
    print_out = args.print_out.lower() == "true"
    env_path = args.environment

    # create environment
    env = read_environment(env_path)

    # MCTS
    model = DecisionTree(env, step)
    model.train(n_iterations, n_steps_ahead=0, print_out=print_out)
    model.save_action_table(save_action_table_path)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
