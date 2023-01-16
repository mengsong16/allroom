import os
#from all.experiments import run_experiment
from all.environments import GymEnvironment
from all.presets.builder import PresetBuilder
from all.presets.classic_control.models import fc_relu_q
from all.presets.classic_control.dqn import DQNClassicControlPreset
from allroom.utils.path import *
from allroom.utils.data_utils import *
from allroom.envs.common import *
from allroom.agents.models import goal_fc_relu_q
from allroom.agents.goal_dqn import GoalDQNPreset
#from all.experiments.single_env_experiment import SingleEnvExperiment
from allroom.experiments.single_env_experiment import SingleEnvExperiment
from all.experiments.parallel_env_experiment import ParallelEnvExperiment
from all.presets import ParallelPreset

def run_experiment(
        agents,
        envs,
        frames,
        logdir='runs',
        quiet=False,
        render=False,
        test_episodes=100,
        verbose=True,
        logger="tensorboard"
):
    if not isinstance(agents, list):
        agents = [agents]

    if not isinstance(envs, list):
        envs = [envs]

    for env in envs:
        for preset_builder in agents:
            # experiment type is decided by the environment type, i.e. vector environment or not
            preset = preset_builder.env(env).build()
            make_experiment = get_experiment_type(preset)
            experiment = make_experiment(
                preset,
                env,
                train_steps=frames,
                logdir=logdir,
                quiet=quiet,
                render=render,
                verbose=verbose,
                logger=logger
            )
            experiment.train(frames=frames)
            experiment.save()
            experiment.test(episodes=test_episodes)
            experiment.close()


def get_experiment_type(preset):
    if isinstance(preset, ParallelPreset):
        return ParallelEnvExperiment
    return SingleEnvExperiment


def main(config_file_name):
    # the hyperparameter dictionary should not include keywords device and name
    hyperparameters = parse_config(os.path.join(config_path, config_file_name))

    # device
    device = get_device_str(hyperparameters)

    # runs directory
    runs_dir = os.path.join(runs_path, hyperparameters["runs_name"])

    goal_conditioned = hyperparameters.get("goal_conditioned", False)
    print("======> Goal conditioned: %s"%goal_conditioned)
    if goal_conditioned:
        # q networks
        hyperparameters["model_constructor"] = goal_fc_relu_q
        # agent
        agent = PresetBuilder(default_name=hyperparameters["algorithm_name"], 
                default_hyperparameters=hyperparameters, 
                constructor=GoalDQNPreset,
                device=device)
        # environment
        if hyperparameters["env_id"] in ['bitflip-v0', 'empty-maze-v0', 'umaze-v0', 'four-room-v0']:
            env = GoalGymEnvironment(id=hyperparameters["env_id"], device=device, 
                random_start = hyperparameters["random_start"],
                random_goal = hyperparameters["random_goal"])
            inner_env = env.env.unwrapped
            print("======> Random start: %s"%inner_env.random_start)
            print("======> Random goal: %s"%inner_env.random_goal)
        else:
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
    plot_returns_100(runs_dir=runs_dir, 
        runs_name=hyperparameters["runs_name"], 
        timesteps=hyperparameters["timesteps"])

if __name__ == "__main__":
    #main(config_file_name = "dqn-cartpole.yaml")
    #main(config_file_name = "goal-dqn-bitflip.yaml")
    #main(config_file_name = "her-dqn-bitflip.yaml")
    #main(config_file_name = "goal-dqn-fourroom.yaml")
    main(config_file_name = "her-dqn-fourroom.yaml")