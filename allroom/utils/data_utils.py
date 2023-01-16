
import os
import numpy as np
import yaml
import math
import collections
import torch
import random
import gym
import matplotlib.pyplot as plt
from all.experiments.plots import load_returns_100_data, subplot_returns_100
from all.core.state import State, StateArray
import copy

# parse a config yaml file to a dictionary
def parse_config(config):
    """
    Parse yaml config file
    """
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    if isinstance(config, collectionsAbc.Mapping):
        return config
    else:
        assert isinstance(config, str)

    if not os.path.exists(config):
        raise IOError(
            "config path {} does not exist. Please either pass in a dict or a string that represents the file path to the config yaml.".format(
                config
            )
        )
    with open(config, "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data

def get_device_str(config):
    if torch.cuda.is_available():
        return "cuda:{}".format(int(config.get("gpu_id")))
    else:
        return "cpu"

def seed_env(env: gym.Env, seed: int) -> None:
    """Set the random seed of the environment."""
    # if seed is None:
    #     seed = np.random.randint(2 ** 31 - 1)
    seed = int(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def seed_other(seed: int):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

# plot all runs under the folder
# the runs should be a combindation of envs and algorithms
# each run result: training results saved as results100.csv
# test results saved as results-test.csv
def plot_returns_100(runs_dir, runs_name, timesteps=-1):
    data = load_returns_100_data(runs_dir)
    lines = {}
    fig, axes = plt.subplots(1, len(data))
    if len(data) == 1:
        axes = [axes]
    for i, env in enumerate(sorted(data.keys())):
        ax = axes[i]
        subplot_returns_100(ax, env, data[env], lines, timesteps=timesteps)
    fig.legend(list(lines.values()), list(lines.keys()), loc="center right")
    plt.savefig(os.path.join(runs_dir, 'plot_%s.png'%(runs_name)))
    plt.show()

def clone_state_array(state_array):
    x = {}
    for key, value in state_array.items():
        x[key] = copy.deepcopy(value)
    
    new_state_array = StateArray(x=x, shape=state_array.shape, device=state_array.device)

    return new_state_array

# cat a state array to a new state array
def cat_states_goals(states):
    # [B, state_dim+goal_dim] = [B, state_dim] + [B, goal_dim]
    cat_tensor = torch.cat([states['observation'], states['desired_goal']], dim=1).to(device=states.device)

    new_states = clone_state_array(states)
    new_states['observation'] = cat_tensor
    
    return new_states

def clone_state(state):
    x = {}
    for key, value in state.items():
        x[key] = copy.deepcopy(value)
    
    new_state = State(x=x, device=state.device)

    # print('**********************')
    # print(state.device)
    # print(new_state.device)
    # print('**********************')

    return new_state
    
# cat a single state to a new state
def cat_state_goal(state):
    # [state_dim+goal_dim] = [state_dim] + [goal_dim]
    # torch.cat: [1, state_dim+goal_dim] = [1, state_dim] + [1, goal_dim]
    cat_tensor = torch.cat([state['observation'].unsqueeze(0), state['desired_goal'].unsqueeze(0)], dim=1).squeeze(0).to(device=state.device)
    
    new_state = clone_state(state)
    new_state['observation'] = cat_tensor

    return new_state