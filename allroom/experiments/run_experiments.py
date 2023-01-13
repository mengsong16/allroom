from all.experiments import run_experiment, plot_returns_100
#from all.presets.classic_control import dqn
from all.environments import GymEnvironment
from all.presets.builder import PresetBuilder
from all.presets.classic_control.models import fc_relu_q
from all.presets.classic_control.dqn import DQNClassicControlPreset
from allroom.utils.path import *

def main():
    hyperparameters = {
        # Common settings
        "discount_factor": 0.99,
        # Adam optimizer settings
        "lr": 1e-3,
        # Training settings
        "minibatch_size": 64,
        "update_frequency": 1,
        "target_update_frequency": 100,
        "timesteps": 40000,
        # Replay buffer settings
        "replay_start_size": 1000,
        "replay_buffer_size": 10000,
        # Explicit exploration
        "initial_exploration": 1.,
        "final_exploration": 0.,
        "final_exploration_step": 10000,
        "test_exploration": 0.001,
        # Model construction
        "model_constructor": fc_relu_q,
    }
    
    # the hyperparameter dictionary should not include keywords device and name
    device = "cuda"
    # need to explicitly ensure the agent and env are on the same device
    agent = PresetBuilder(default_name='dqn', 
            default_hyperparameters=hyperparameters, 
            constructor=DQNClassicControlPreset,
            device=device)

    run_experiment(
        agents=[agent],
        envs=[GymEnvironment(id='CartPole-v0', device=device)],
        frames=hyperparameters["timesteps"],
        logdir=runs_path,
        quiet=False, # if False print info to standard output 
        verbose=True, # whether or not to log all data or only summary metrics
    )

    plot_returns_100(runs_path, timesteps=hyperparameters["timesteps"])

if __name__ == "__main__":
    main()