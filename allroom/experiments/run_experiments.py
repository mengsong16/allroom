import os
from all.experiments import run_experiment
from all.environments import GymEnvironment
from all.presets.builder import PresetBuilder
from all.presets.classic_control.models import fc_relu_q
from all.presets.classic_control.dqn import DQNClassicControlPreset
from allroom.utils.path import *
from allroom.utils.data_utils import *
from allroom.envs.common import *
from allroom.agents.models import goal_fc_relu_q
from allroom.agents.goal_dqn import GoalDQNPreset

def main(config_file_name):
    # the hyperparameter dictionary should not include keywords device and name
    hyperparameters = parse_config(os.path.join(config_path, config_file_name))

    # device
    device = get_device_str(hyperparameters)

    # runs directory
    runs_dir = os.path.join(runs_path, hyperparameters["runs_name"])


    if hyperparameters["goal_conditioned"]:
        # q networks
        hyperparameters["model_constructor"] = goal_fc_relu_q
        # agent
        agent = PresetBuilder(default_name=hyperparameters["algorithm_name"], 
                default_hyperparameters=hyperparameters, 
                constructor=GoalDQNPreset,
                device=device)
        # environment
        env = GoalGymEnvironment(id=hyperparameters["env_id"], device=device)
    else:
        # q networks
        hyperparameters["model_constructor"] = fc_relu_q
        # need to explicitly ensure the agent and env are on the same device
        # agent
        agent = PresetBuilder(default_name=hyperparameters["algorithm_name"], 
                default_hyperparameters=hyperparameters, 
                constructor=DQNClassicControlPreset,
                device=device)
        # environment
        env = GymEnvironment(id=hyperparameters["env_id"], device=device)       
    
    
    # seed
    # assume non-parallel environments
    seed_env(env=env, seed=hyperparameters["seed"]) 
    seed_other(seed=hyperparameters["seed"])
    
    # train and test
    run_experiment(
        agents=[agent],
        envs=[env],
        frames=hyperparameters["timesteps"],
        logdir=runs_dir,
        quiet=False, # if False print info to standard output 
        verbose=True, # whether or not to log all data or only summary metrics
    )

    # plot training results
    plot_returns_100(runs_dir, timesteps=hyperparameters["timesteps"])

if __name__ == "__main__":
    #main(config_file_name = "dqn-cartpole.yaml")
    main(config_file_name = "goal-dqn-bitflip.yaml")