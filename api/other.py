from space_navigator.generator import Generator
import json

import os

start_time = 6600  # TODO: start and end time to import json
end_time = 6600.1

ENV_PATH = "generated_collision_api.env"
MANEUVERS_PATH = "maneuvers.csv"


def generate_env(protected_params, save_path=ENV_PATH):
    n_debris = 5
    time_before_start_time = 0.1
    pos_sigma = 0
    vel_ratio_sigma = 0.05
    i_threshold = 0.5

    generator = Generator(start_time, end_time)
    generator.add_protected_by_params(protected_params)
    for _ in range(n_debris):
        generator.add_debris(pos_sigma, vel_ratio_sigma, i_threshold)
    generator.save_env(save_path, time_before_start_time)


if __name__ == "__main__":
    # generate debris
    json_path = 'protected_params_api.json'
    with open(json_path, "r") as read_file:
        protected_params = json.load(read_file)
    generate_env(protected_params)

    # calculate maneuvers
    # TODO: without OS
    os.system(
        f'python training/CE/CE_train_for_collision.py -env {ENV_PATH} -print true -save_path {MANEUVERS_PATH} \
-r false -n_m 1')

    # TODO: sent graphs and gif
    # run simulator
    os.system(f'python examples/collision.py -env {ENV_PATH} -model {MANEUVERS_PATH} -v False')
