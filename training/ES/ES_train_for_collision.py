# Train agent for collision at time 6600.

import argparse
import sys

import numpy as np
import pykep as pk

from space_navigator.api import MAX_FUEL_CONSUMPTION
from space_navigator.api import Environment
from space_navigator.models import ProgressPlotter
from space_navigator.models.ES import EvolutionStrategies
from space_navigator.utils import read_environment


PROPAGATION_STEP = 0.000001
ACTION_SIZE = 4


def main(args):
    parser = argparse.ArgumentParser()

    # train parameteres
    parser.add_argument("-n", "--n_actions", type=int,
                        default=2, required=False)
    parser.add_argument("-i", "--iterations", type=int,
                        default=10, required=False)
    parser.add_argument("-pop_size", "--population_size", type=int,
                        default=10, required=False)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=0.1, required=False)
    parser.add_argument("-d", "--decay", type=float,
                        default=0.99, required=False)
    parser.add_argument("-sigma", "--sigma_coef", type=float,
                        default=0.5, required=False)

    # output parameteres
    parser.add_argument("-progress", "--show_progress", type=str,
                        default="False", required=False)
    parser.add_argument("-out", "--output_path", type=str,
                        default=".", required=False, help="Output folder from progress plots.")

    parser.add_argument("-print", "--print_out", type=str,
                        default="False", required=False)

    # simulation parameteres
    parser.add_argument("-save_path", "--save_action_table_path", type=str,
                        default="training/agents_tables/ES/action_table_ES.csv", required=False)
    parser.add_argument("-env", "--environment", type=str,
                        default="data/environments/collision.env", required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=PROPAGATION_STEP, required=False)

    args = parser.parse_args(args)

    iterations = args.iterations
    population_size = args.population_size
    n_actions = args.n_actions
    learning_rate, decay, sigma_coef,  = args.learning_rate, args.decay, args.sigma_coef

    step = args.step
    table_path = args.save_action_table_path
    print_out = args.print_out.lower() == "true"
    show_progress = args.show_progress.lower() == "true"
    output_path = args.output_path

    # create environment
    env_path = args.environment
    env = read_environment(env_path)

    # ES parameteres
    weights_shape = (n_actions, ACTION_SIZE)
    sigma_table = np.full(weights_shape, sigma_coef)
    sigma_table[:, -1] = step

    model = EvolutionStrategies(env, step, weights_shape,
                                population_size=population_size, sigma=sigma_table, learning_rate=learning_rate, decay=decay)
    model.train(iterations, print_out=print_out)
    model.save(table_path)

    if show_progress:
        plotter = ProgressPlotter(output_path, model)
        plotter.plot_all_rewards("all_rewards.png")
        plotter.plot_mean_reward_per_iteration("mean_rewards.png")

    return


if __name__ == "__main__":
    main(sys.argv[1:])
