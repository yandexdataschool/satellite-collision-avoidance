# Train agent for collision at time 6600.

import argparse
import sys

from space_navigator.utils import read_space_objects
from space_navigator.api import MAX_FUEL_CONSUMPTION
from space_navigator.models.CE import CrossEntropy

START_TIME = 6599.95
SIMULATION_STEP = 0.0001
END_TIME = 6600.05


def main(args):
    parser = argparse.ArgumentParser()

    # train parameters
    parser.add_argument("-n_a", "--n_actions", type=int,
                        default=3, required=False)
    parser.add_argument("-n_i", "--n_iterations", type=int,
                        default=20, required=False)
    parser.add_argument("-n_s", "--n_sessions", type=int,
                        default=100, required=False)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=0.7, required=False)
    parser.add_argument("-lr_d", "--learning_rate_decay", type=float,
                        default=0.98, required=False)
    parser.add_argument("-perc", "--percentile", type=float,
                        default=80, required=False)
    parser.add_argument("-p_g", "--percentile_growth", type=float,
                        default=1.005, required=False)
    parser.add_argument("-s_d", "--sigma_decay", type=float,
                        default=0.98, required=False)

    # output parameteres
    parser.add_argument("-save_path", "--save_action_table_path", type=str,
                        default="training/agents_tables/CE/action_table_CE.csv", required=False)
    parser.add_argument("-print", "--print_out", type=str,
                        default="False", required=False)
    parser.add_argument("-progress", "--show_progress", type=str,
                        default="False", required=False)

    # simulation parameteres
    parser.add_argument("-env", "--environment", type=str,
                        default="data/environments/collision.osc", required=False)
    parser.add_argument("-start", "--start_time", type=float,
                        default=START_TIME, required=False)
    parser.add_argument("-end", "--end_time", type=float,
                        default=END_TIME, required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=SIMULATION_STEP, required=False)

    args = parser.parse_args(args)

    n_actions, n_iterations, n_sessions = args.n_actions, args.n_iterations, args.n_sessions
    percentile, learning_rate = args.percentile, args.learning_rate
    sigma_decay, learning_rate_decay = args.learning_rate_decay, args.learning_rate_decay
    percentile_growth = args.percentile_growth
    env = args.environment

    start_time, end_time, step = args.start_time, args.end_time, args.step
    save_action_table_path = args.save_action_table_path
    print_out = args.print_out.lower() == "true"
    show_progress = args.show_progress.lower() == "true"

    osc = read_space_objects(env, "osc")
    protected = osc[0]
    debris = [osc[1]]

    max_fuel_cons = MAX_FUEL_CONSUMPTION
    fuel_level = protected.get_fuel()

    action_table = CrossEntropy(
        protected, debris, start_time, end_time, step,
        max_fuel_cons, fuel_level, n_actions)
    action_table.train(n_iterations, n_sessions, learning_rate, percentile,
                       sigma_decay, learning_rate_decay, percentile_growth,
                       print_out, show_progress)
    action_table.save_action_table(save_action_table_path)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
