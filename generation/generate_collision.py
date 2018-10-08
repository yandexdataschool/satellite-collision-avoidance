# Generate random collision environment.

import argparse
import sys

from space_navigator.generator import Generator


def main(args):
    parser = argparse.ArgumentParser()

    # generator initialization
    parser.add_argument("-n_d", "--n_debris", type=int,
                        default=2, required=False)
    parser.add_argument("-start", "--start_time", type=float,
                        default=6600, required=False)
    parser.add_argument("-end", "--end_time", type=float,
                        default=6600.1, required=False)
    parser.add_argument("-before", "--time_before_start_time", type=float,
                        default=0, required=False)

    # debris parameters
    parser.add_argument("-p_s", "--pos_sigma", type=float,
                        default=0, required=False)
    parser.add_argument("-v_r_s", "--vel_ratio_sigma", type=float,
                        default=0.05, required=False)
    parser.add_argument("-i_t", "--i_threshold", type=float,
                        default=0.5, required=False)

    # other parameters
    parser.add_argument("-save_path", "--environment_save_path", type=str,
                        default="data/environments/generated_collision.env", required=False)

    # args parsing
    args = parser.parse_args(args)

    n_debris, start_time, end_time = args.n_debris, args.start_time, args.end_time
    time_before_start_time = args.time_before_start_time

    pos_sigma, vel_ratio_sigma = args.pos_sigma, args.vel_ratio_sigma
    i_threshold = args.i_threshold

    save_path = args.environment_save_path

    # generation
    generator = Generator(start_time, end_time)

    generator.add_protected()
    for _ in range(n_debris):
        generator.add_debris(pos_sigma, vel_ratio_sigma, i_threshold)

    generator.save_env(save_path, time_before_start_time)

    return

if __name__ == "__main__":
    main(sys.argv[1:])
