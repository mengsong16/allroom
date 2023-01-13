from all.experiments import run_experiment, plot_returns_100
from all.presets.classic_control import dqn
from all.environments import GymEnvironment


def main():
    device = 'cuda'
    timesteps = 40000

    run_experiment(
        agents=[dqn.device(device)],
        envs=[GymEnvironment(id='CartPole-v0', device=device)],
        frames=timesteps,
    )
    plot_returns_100('runs', timesteps=timesteps)

if __name__ == "__main__":
    main()