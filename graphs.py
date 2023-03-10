import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import itertools as it
from pathlib import Path

def extract_exp_data(
    exp_name=None,
    exp_path="/homedtcl/mmahaut/projects/llm2llm/experiments",
    config_name="params.yaml",
    ):
    # get config file
    exp_path = Path(exp_path) / exp_name if exp_name else exp_path
    params = {}
    with open(exp_path / config_name, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    values = [v for v in it.product(*(params[key] for key in params))]
    
    exp_data = []
    for f in os.listdir(exp_path):
        _dir = exp_path / f
        if _dir.is_dir():
            # get the log file ending in out
            log_path = [ _dir / _f for _f in os.listdir(_dir) if _f.endswith(".out")][0] 
            # extract data from log file and append to exp_data as dataframes
            exp_data.append(pd.DataFrame(extract_log_data(log_path)))
            # add the params to the dataframe
            _vals = values[int(f)]
            for i, k in enumerate(params):
                if k != "affixes":
                    exp_data[-1][k] = _vals[i]
                else:
                    exp_data[-1][k] = _vals[i][0][0][0]
    return exp_data

def extract_log_data(log_path):
    log_data = {}
    # open log file
    with open(log_path, "r") as f:
        # check each line for the keyword
        for line in f:
            # add the data to the dict
            data = extract_data_from_line(line)
            if data:
                for k,v in extract_data_from_line(line).items():
                    # check if the key is in the dict
                    if k not in log_data:
                        log_data[k] = []
                    log_data[k].append(v)
    return log_data

def extract_data_from_line(line):
    # format : Episode 2 --> Acc: 0/100 Reward: 6.030714985172381e-07
    if "Reward" in line and "Acc" in line:
        # extract the reward
        reward = float(line.split("Reward: ")[1].split(" ")[0])
        # extract the accuracy
        acc = float(line.split("Acc: ")[1].split("/")[0])
        # extract accuracy numerator
        acc_num = float(line.split("Acc: ")[1].split("/")[1].split(" ")[0])
        # return as a dict
        return {"reward": reward, "acc": acc/acc_num}


def plot_data(data, metric="reward", save_path=None, visualize=False, title=None, hue=None):
    # use seaborn to plot the data
    sns.set()
    # create a dataframe with the data
    df = pd.concat(data)
    # plot the data
    sns.lineplot(x=df.index, y=metric, hue=hue, data=df)
    # add title
    if title is not None:
        plt.title(title)
    # save the plot
    if save_path is not None:
        plt.savefig(save_path)
    # show the plot
    if visualize:
        plt.show()
    # close the plot
    plt.close()


def plot_data_from_exp(
    exp_name=None,
    exp_path="/homedtcl/mmahaut/projects/llm2llm/experiments",
    config_name="params.yaml",
    metric="reward",
    hue=None,
    save_path=None,
    visualize=False,
    ):
    assert exp_name is not None, "exp_name must be specified"
    if save_path is None and visualize is False:
        save_path = f"{exp_path}/{exp_name}/{metric}.png"
    # extract data
    data = extract_exp_data(exp_name, exp_path, config_name)
    # plot the data
    plot_data(data, metric, save_path, visualize, hue=hue)
