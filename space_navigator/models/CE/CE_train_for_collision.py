# Train agent for collision at time 6600.

import argparse
import sys

from simulator import read_space_objects
from api import MAX_FUEL_CONSUMPTION
from CE.CE import CrossEntropy

START_TIME = 6599.95
SIMULATION_STEP = 0.0001
END_TIME = 6600.05


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_a", "--n_actions", type=int,
                        default=3, required=False)
    parser.add_argument("-n_i", "--n_iterations", type=int,
                        default=10, required=False)
    parser.add_argument("-n_s", "--n_sessions", type=int,
                        default=10, required=False)
    parser.add_argument("-n_b", "--n_best_actions", type=int,
                        default=1, required=False)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=0.7, required=False)
    parser.add_argument("-s_coef", "--sigma_coef", type=float,
                        default=0.9, required=False)
    parser.add_argument("-lr_coef", "--learning_rate_coef", type=float,
                        default=0.9, required=False)
    parser.add_argument("-start", "--start_time", type=float,
                        default=START_TIME, required=False)
    parser.add_argument("-end", "--end_time", type=float,
                        default=END_TIME, required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=SIMULATION_STEP, required=False)
    parser.add_argument("-save_path", "--save_action_table_path", type=str,
                        default="data/action_table_CE.csv", required=False)
    parser.add_argument("-print", "--print_out", type=str,
                        default="False", required=False)
    parser.add_argument("-progress", "--show_progress", type=str,
                        default="False", required=False)

    args = parser.parse_args(args)

    n_actions, n_iterations = args.n_actions, args.n_iterations
    n_sessions, n_best_actions = args.n_sessions, args.n_best_actions
    learning_rate, sigma_coef, learning_rate_coef = args.learning_rate, args.sigma_coef, args.learning_rate_coef

    start_time, end_time, step = args.start_time, args.end_time, args.step
    save_action_table_path = args.save_action_table_path
    print_out = args.print_out.lower() == "true"
    show_progress = args.show_progress.lower() == "true"

    osc = read_space_objects("data/collision.osc", "osc")
    protected = osc[0]
    debris = [osc[1]]

    max_fuel_cons = MAX_FUEL_CONSUMPTION
    fuel_level = protected.get_fuel()

    action_table = CrossEntropy(
        protected, debris, start_time, end_time, step, n_actions)
    action_table.train(n_iterations, n_sessions, n_best_actions,
                       learning_rate, sigma_coef, learning_rate_coef,
                       print_out, show_progress)
    action_table.save_action_table(save_action_table_path)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
