import argparse
import copy
import glob
import os

import evaluate_ppl
import human_eval
import yaml
from accelerate.commands import launch


def run_exp(exp_dict, exp_name, savedir, args):
    if exp_name.startswith("ppl"):
        evaluate_ppl.main_dict(exp_dict)
    elif exp_name.startswith("humaneval"):
        accelerate_launch("human_eval.py", exp_dict, args.gpus)
    elif exp_name.startswith("rew"):
        accelerate_launch("evaluate_reward.py", exp_dict, args.gpus)
    elif exp_name.startswith("rlhf"):
        accelerate_launch("er_training.py", exp_dict, args.gpus)
    elif exp_name.startswith("rm"):
        accelerate_launch("reward_modeling.py", exp_dict, args.gpus)
    elif exp_name.startswith("sft"):
        accelerate_launch("supervised_finetuning.py", exp_dict, args.gpus)


def accelerate_launch(training_file, training_args_dict, num_gpus=1):
    parser = launch.launch_command_parser()
    training_cmd_args = []
    if num_gpus > 1:
        training_cmd_args.append("--multi_gpu")
        training_cmd_args.extend(["--num_machines", "1"])
        training_cmd_args.extend(["--num_processes", str(num_gpus)])
    training_cmd_args.append(training_file)
    for key, val in training_args_dict.items():
        training_cmd_args.append(f"--{key}")
        if not (isinstance(val, bool) and val == True):
            training_cmd_args.append(str(val))
    args = parser.parse_args(training_cmd_args)
    launch.launch_command(args)


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group",
        help="Define the experiment group to run.",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        default="results",
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument(
        "-n", "--gpus", default=1, type=int, help="number of gpus to use for experiment"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="disable wandb", default=False
    )

    args, _ = parser.parse_known_args()

    with open(args.exp_group, "r") as fp:
        exp_dict = yaml.safe_load(fp)

    exp_name = os.path.basename(args.exp_group)

    run_exp(exp_dict, exp_name, args.savedir_base, args)
