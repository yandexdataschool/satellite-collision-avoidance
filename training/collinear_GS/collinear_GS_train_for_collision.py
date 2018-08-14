# Train agent for collision at time 6600.

import argparse
import sys

from space_navigator.utils import read_environment
from space_navigator.models.collinear_GS import CollinearGridSearch

PROPAGATION_STEP = 0.000001


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("-n_s", "--n_sessions", type=int,
                        default=100, required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=PROPAGATION_STEP, required=False)
    parser.add_argument("-r", "--reverse", type=str,
                        default="True", required=False)

    parser.add_argument("-save_path", "--save_action_table_path", type=str,
                        default="training/agents_tables/collinear_GS/action_table_collinear_GS.csv", required=False)
    parser.add_argument("-print", "--print_out", type=str,
                        default="False", required=False)
    parser.add_argument("-env", "--environment", type=str,
                        default="data/environments/collision.env", required=False)

    args = parser.parse_args(args)

    n_sessions = args.n_sessions
    step = args.step
    reverse = args.reverse.lower() == "true"
    save_action_table_path = args.save_action_table_path
    print_out = args.print_out.lower() == "true"
    env_path = args.environment

    # create environment
    env = read_environment(env_path)

    # model
    model = CollinearGridSearch(env, step, reverse)
    iteration_kwargs = {
        "n_sessions": n_sessions,
    }
    model.train(1, print_out, **iteration_kwargs)
    model.save_action_table(save_action_table_path)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
