# Train agent for collision at time 6600.

import argparse
import sys

from space_navigator.utils import read_environment
from space_navigator.models.CE import CrossEntropy

PROPAGATION_STEP = 0.000001


def main(args):
    parser = argparse.ArgumentParser()

    # train parameters
    parser.add_argument("-n_m", "--n_maneuvers", type=int,
                        default=2, required=False)
    parser.add_argument("-n_i", "--n_iterations", type=int,
                        default=50, required=False)
    parser.add_argument("-n_s", "--n_sessions", type=int,
                        default=30, required=False)

    parser.add_argument("-r", "--reverse", type=str,
                        default="True", required=False)
    parser.add_argument("-f_man", "--first_maneuver_time", type=str,
                        default="early", required=False,
                        help="early or auto")
    parser.add_argument("-dv", "--dV_angle", type=str,
                        default="auto", required=False,
                        help="auto or complanar or collinear")
    parser.add_argument("-low_r_step", "--step_if_low_reward", type=str,
                        default="False", required=False,
                        help="step if new reward is lower than current")
    parser.add_argument("-early_stop", "--early_stopping", type=str,
                        default="True", required=False)

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

    # simulation parameteres
    parser.add_argument("-env", "--environment", type=str,
                        default="data/environments/collision.env", required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=PROPAGATION_STEP, required=False)

    # output parameteres
    parser.add_argument("-save_path", "--save_action_table_path", type=str,
                        default="training/agents_tables/CE/action_table_CE.csv", required=False)
    parser.add_argument("-print", "--print_out", type=str,
                        default="False", required=False)
    parser.add_argument("-progress", "--show_progress", type=str,
                        default="False", required=False)

    args = parser.parse_args(args)

    # train parameters
    n_maneuvers, n_iterations, n_sessions = args.n_maneuvers, args.n_iterations, args.n_sessions
    reverse = args.reverse.lower() == "true"
    first_maneuver_time, dV_angle = args.first_maneuver_time.lower(), args.dV_angle.lower()
    step_if_low_reward = args.step_if_low_reward.lower() == "true"
    early_stopping = args.early_stopping.lower() == "true"
    percentile, learning_rate = args.percentile, args.learning_rate
    sigma_decay, learning_rate_decay = args.sigma_decay, args.learning_rate_decay
    percentile_growth = args.percentile_growth

    # simulation parameteres
    env_path = args.environment
    step = args.step

    # output parameteres
    save_action_table_path = args.save_action_table_path
    print_out = args.print_out.lower() == "true"
    show_progress = args.show_progress.lower() == "true"

    # create environment
    env = read_environment(env_path)

    # CE
    model = CrossEntropy(env, step, reverse, first_maneuver_time,
                         n_maneuvers, learning_rate, percentile)
    iteration_kwargs = {
        "n_sessions": n_sessions,
        "sigma_decay": sigma_decay,
        "show_progress": show_progress,
        "dV_angle": dV_angle,
        "step_if_low_reward": step_if_low_reward,
        "early_stopping": early_stopping,
    }
    model.train(n_iterations, print_out, **iteration_kwargs)
    model.save_action_table(save_action_table_path)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
