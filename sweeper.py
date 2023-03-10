import os
import yaml
import itertools as it
import sys
from pathlib import Path
import argparse
from time import sleep

default_checkpoint_dir = "/homedtcl/mmahaut/projects/experiments"


def get_opts(arguments):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--params_path",
        type=str,
        required=True,
        help="path to the yaml file containing the parameters to sweep through",
    )
    arg_parser.add_argument(
        "--log_path",
        type=str,
        default="/homedtcl/mmahaut/projects/llm2llm/experiments",
        help="path where everything will be saved",
    )
    arg_parser.add_argument(
        "--default_params_path",
        type=str,
        default="/homedtcl/mmahaut/projects/llm2llm/PPO_finetuning/local_gpu_config.yaml",
        help="path to the yaml file containing the default parameters",
    )
    arg_parser.add_argument(
        "--memory",
        type=str,
        default="200G",
        help="assigned memory in GB",
    )
    arg_parser.add_argument(
        "--job_name",
        type=str,
        default="job",
        help="name of the job. If no checkpoint_dir is given, this is used as the name of the folder in which the job is stored",
    )
    arg_parser.add_argument(
        "--sbatch_dir",
        type=str,
        default="/homedtcl/mmahaut/projects/manual_slurm",
        help="path to the directory where the sbatch file is stored",
    )
    arg_parser.add_argument(
        "--partition",
        type=str,
        default="alien",
        help="slurm partition on which the jobs are run",
    )

    arg_parser.add_argument(
        "--time",
        type=str,
        default="3-00:00:00",
        help="time allocated for each job",
    )
    arg_parser.add_argument(
        "--qos",
        type=str,
        default="alien",
        help="slurm qos for each jobs",
    )
    arg_parser.add_argument(
        "--acc_config_path",
        type=str,
        default="/homedtcl/mmahaut/projects/llm2llm/PPO_finetuning/default_accelerate.yaml",
        help="path to the accelerate config file",
    )

    opts = arg_parser.parse_args(arguments)
    return opts


def sweep_params(opts):
    with open(opts.params_path, "r") as f:
        params = yaml.safe_load(f)

        log_path = Path(opts.log_path) / opts.job_name

        for i, values in enumerate(it.product(*(params[key] for key in params))):
            job_path = log_path / str(i)
            job_path.mkdir(parents=True, exist_ok=True)
            n_gpus = build_yaml(params.keys(), values, opts.default_params_path, job_path)
            sbatch_file = write_sbatch(
                i,
                opts.job_name,
                job_path,
                opts.partition,
                n_gpus,
                opts.time,
                opts.memory,
                opts.qos,
            )
            # adjust the port number
            adjust_port_number(opts.acc_config_path, job_path, i)
            _return = os.system(f"sbatch {sbatch_file}")
    os.system(f"cp {opts.params_path} {log_path / 'params.yaml'}")

def adjust_port_number(acc_config_path, log_path, job_idx):
    with open(acc_config_path, "r") as f:
        acc_config = yaml.safe_load(f)
    acc_config["main_process_port"] = 6000 + job_idx
    with open(log_path / "accelerate_config.yaml", "w") as f:
        yaml.dump(acc_config, f)

def build_yaml(keys, values, default_path, log_path):
    """
    takes the default yaml file and replaces the specified parameters.
    for now does not support changing the model type
    """
    # open default yaml file
    with open(default_path, "r") as f:
        _default = f.read()
        default = yaml.load(_default, Loader=yaml.FullLoader)
    # replace the parameters
    for i, key in enumerate(keys):
        default["rl_script_args"][key] = values[i]
    default["rl_script_args"]["log_dir"] = str(log_path)
    default["rl_script_args"]["log_file"] = str(log_path / f"log.txt")
    default["lamorel_args"]["accelerate_args"]["config_file"] = str(log_path / "accelerate_config.yaml")
    # write the new yaml file
    with open(log_path / f"config.yaml", "w") as f:
        yaml.dump(default, f)
    # return the number of gpus
    return default["lamorel_args"]["llm_args"]["parallelism"]["model_parallelism_size"]




def write_sbatch(
    job_idx,
    jobname,
    log_path: Path,
    partition,
    n_gpus,
    time,
    mem,
    qos,
):
    """
    writes a sbatch file for the current job
    """
    sbatch_path = log_path / f"{jobname}.sh"
    with open(sbatch_path, "w") as f:
        f.write(
            f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --qos={qos}
#SBATCH --nodes=1
#SBATCH --exclude=node044
#SBATCH --nice=42
#SBATCH --ntasks-per-node=1
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output={log_path}/%j.out
#SBATCH --error={log_path}/%j.err

source /homedtic/mmahaut/.bashrc
conda activate llm2llm

python -m lamorel_launcher.launch --config-path {log_path} --config-name config rl_script_args.path=/homedtcl/mmahaut/projects/llm2llm/PPO_finetuning/main.py
echo "done"
"""
        )
    return sbatch_path


if __name__ == "__main__":
    sweep_params(get_opts(sys.argv[1:]))
