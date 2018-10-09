# Train different models for the environment (abbreviated parameter set).

import argparse
import sys
import os
import pandas as pd

from space_navigator.utils import read_environment
from space_navigator.agent.table_agent import TableAgent
from space_navigator.simulator import Simulator

from space_navigator.models.baseline import Baseline
from space_navigator.models.collinear_GS import CollinearGridSearch
from space_navigator.models.CE import CrossEntropy

PROPAGATION_STEP = 0.000001
REVERSE = False


def model_info(env, model, step):
    index = [
        "coll prob", "fuel (|dV|)",
        "dev a (m)", "dev e", "dev i (rad)",
        "dev W (rad)", "dev w (rad)", "dev M (rad)",
    ]
    v = "value"
    r = "reward"
    columns = [r, v]
    df = pd.DataFrame(index=index, columns=[v, r])

    agent = TableAgent(model.get_action_table())
    simulator = Simulator(agent, env, step)
    simulator.run()

    df["threshold"] = [env.coll_prob_thr,
                       env.fuel_cons_thr] + list(env.traj_dev_thr)
    df.loc["coll prob", v] = env.get_total_collision_probability()
    df.loc["fuel (|dV|)", v] = env.get_fuel_consumption()
    df.loc[2:, v] = env.get_trajectory_deviation()
    reward_components = env.get_reward_components()
    df.loc["coll prob", r] = reward_components["coll_prob"]
    df.loc["fuel (|dV|)", r] = reward_components["fuel"]
    df.loc[2:, r] = reward_components["traj_dev"]

    env.reset()
    return df


def save_result(model, name, dir_path, env, step):
    path = os.path.join(dir_path, name)
    model.save_action_table(path + "_act.csv")
    model_info(env, model, step).to_csv(path + "_inf.csv")


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("-full", "--full_train", type=str,
                        default="True", required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=PROPAGATION_STEP, required=False)

    parser.add_argument("-save_dir", "--save_action_table_dir", type=str,
                        default="vr/training/", required=False)
    parser.add_argument("-print", "--print_out", type=str,
                        default="False", required=False)
    parser.add_argument("-env", "--environment", type=str,
                        default="vr/test.env", required=False)

    args = parser.parse_args(args)

    full_train = args.full_train.lower() == "true"
    step = args.step
    save_action_table_dir = args.save_action_table_dir
    if not os.path.exists(save_action_table_dir):
        os.makedirs(save_action_table_dir)
    print_out = args.print_out.lower() == "true"
    env_path = args.environment

    # create environment
    env = read_environment(env_path)

    # in-track one (collinear): baseline

    # print("\nIn-track. One object.\n")
    # model1 = Baseline(env, step, REVERSE)
    # iteration_kwargs1 = {
    #     "n_sessions": 1000 if full_train else 50,
    # }
    # model1.train(1, print_out, **iteration_kwargs1)
    # name1 = "in-track_one_obj"
    # save_result(model1, name1, save_action_table_dir, env, step)

    # in-track all (collinear): collinear GS
    print("\nIn-track. All objects.\n")
    model2 = CollinearGridSearch(env, step, REVERSE)
    iteration_kwargs2 = {
        "n_sessions": 1000 if full_train else 50,
    }
    model2.train(1, print_out, **iteration_kwargs2)
    name2 = "in-track_all_obj"
    save_result(model2, name2, save_action_table_dir, env, step)

    # in plane (complanar): CE or collinear GS + CE tuning
    print("\nIn-plane. All objects.\n")
    # CE
    model3 = CrossEntropy(env, step, REVERSE, first_maneuver_time="early",
                          n_maneuvers=1, lr=0.7, percentile=80)
    iteration_kwargs3 = {
        "n_sessions": 30 if full_train else 3,
        "sigma_decay": 0.98,
        "lr_decay": 0.98,
        "percentile_growth": 1.005,
        "show_progress": False,
        "dV_angle": "complanar",
        "step_if_low_reward": False,
        "early_stopping": True,
    }
    n_iterations3 = 100 if full_train else 3
    model3.train(n_iterations3, print_out, **iteration_kwargs3)
    # collinear GS + CE tuning
    model4 = CrossEntropy(env, step, REVERSE, first_maneuver_time="early",
                          n_maneuvers=1, lr=0.7, percentile=90)
    model4.set_action_table(model2.get_action_table())
    iteration_kwargs4 = {
        "n_sessions": 50 if full_train else 5,
        "sigma_decay": 0.9,
        "lr_decay": 0.98,
        "percentile_growth": 1.01,
        "show_progress": False,
        "dV_angle": "complanar",
        "step_if_low_reward": False,
        "early_stopping": True,
    }
    n_iterations4 = 100 if full_train else 3
    # save best
    name34 = "in-plane_all_obj"
    if model3.get_reward() > model4.get_reward():
        model34 = model3
    else:
        model34 = model4
    save_result(model34, name34, save_action_table_dir, env, step)

    # out-plane: CE
    print("\nOut-plane. All objects.\n")
    model5 = CrossEntropy(env, step, REVERSE, first_maneuver_time="early",
                          n_maneuvers=1, lr=0.5, percentile=80)
    iteration_kwargs5 = {
        "n_sessions": 40 if full_train else 4,
        "sigma_decay": 0.99,
        "lr_decay": 0.99,
        "percentile_growth": 1.003,
        "show_progress": False,
        "dV_angle": "auto",
        "step_if_low_reward": False,
        "early_stopping": True,
    }
    n_iterations5 = 200 if full_train else 3
    model5.train(n_iterations5, print_out, **iteration_kwargs5)
    name5 = "out-plane_all_obj"
    save_result(model5, name5, save_action_table_dir, env, step)

    # print end

#     print(f"""
# Total Reward:
# in-track_one_obj:  {model1.get_reward():.5};
# in-track_all_obj:  {model2.get_reward():.5};
# in-plane_all_obj:  {model34.get_reward():.5};
# out-plane_all_obj: {model4.get_reward():.5}.
# """)

    print(f"""
Total Reward:
in-track_all_obj:  {model2.get_reward():.5};
in-plane_all_obj:  {model34.get_reward():.5};
out-plane_all_obj: {model4.get_reward():.5}.
""")

    # TODO: add NN.

    return


if __name__ == "__main__":
    main(sys.argv[1:])
