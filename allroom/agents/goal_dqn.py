import copy
from torch.optim import Adam

from all.agents import DQN, DQNTestAgent
from all.approximation import QNetwork, FixedTarget
from all.logging import DummyLogger
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from all.presets.builder import PresetBuilder
from all.presets.preset import Preset
from all.presets.classic_control.models import fc_relu_q
from all.presets.classic_control.dqn import DQNClassicControlPreset
from all.agents.dqn import DQN
from all.core.state import State, StateArray
from all.agents._agent import Agent
from allroom.utils.data_utils import *
from allroom.utils.path import *
from allroom.memory.her_replay_buffer import HERReplayBuffer
from allroom.utils.data_utils import cat_state_goal, cat_states_goals
import copy


class GoalDQN(DQN):
    # state: State
    def act(self, state):
        #print(state.shape)
        
        # store (s,a,r,s')
        self.replay_buffer.store(copy.deepcopy(self._state), self._action, copy.deepcopy(state))
        # if necessary, update networks
        self._train()
        # update previous state self._state
        self._state = state
        # choose action
        with torch.no_grad():
            cat_state = cat_state_goal(state)
        #print(cat_state.shape)
        
        self._action = self.policy.no_grad(cat_state)
        return self._action

    def eval(self, state):
        with torch.no_grad():
            cat_state = cat_state_goal(state)
        # print(cat_state.device)
        # print(self.policy.device)
        # exit()
        return self.policy.eval(cat_state)

    def _train(self):
        if self._should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size)
            # forward pass
            cat_states = cat_states_goals(states)
            values = self.q(cat_states, actions)
            #values = self.q(states, actions)
            # compute targets
            cat_next_states = cat_states_goals(next_states)
            targets = rewards + self.discount_factor * torch.max(self.q.target(cat_next_states), dim=1)[0]
            # compute loss
            loss = self.loss(values, targets)
            # backward pass
            self.q.reinforce(loss)

class GoalDQNTestAgent(Agent):
    def __init__(self, policy):
        self.policy = policy

    def act(self, state):
        with torch.no_grad():
            cat_state = cat_state_goal(state)

        return self.policy.eval(cat_state)

class GoalDQNPreset(DQNClassicControlPreset):
    """
    Goal conditioned Deep Q-Network (DQN) Classic Control Preset.

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        name (str): A human-readable name for the preset.
        device (torch.device): The device on which to load the agent.

    Keyword Args:
        discount_factor (float, optional): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        target_update_frequency (int): Number of timesteps between updates the target network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        initial_exploration (float): Initial probability of choosing a random action,
            decayed over course of training.
        final_exploration (float): Final probability of choosing a random action.
        final_exploration_step (int): The step at which exploration decay is finished
        test_exploration (float): The exploration rate of the test Agent
        model_constructor (function): The function used to construct the neural model.
    """

    def __init__(self, env, name, device, **hyperparameters):
        super().__init__(env, name, device, **hyperparameters)
        self.env = env

    def agent(self, logger=DummyLogger(), train_steps=float('inf')):
        optimizer = Adam(self.model.parameters(), lr=self.hyperparameters['lr'])

        q = QNetwork(
            self.model,
            optimizer,
            target=FixedTarget(self.hyperparameters['target_update_frequency']),
            logger=logger
        )

        policy = GreedyPolicy(
            q,
            self.n_actions,
            epsilon=LinearScheduler(
                self.hyperparameters['initial_exploration'],
                self.hyperparameters['final_exploration'],
                self.hyperparameters['replay_start_size'],
                self.hyperparameters['final_exploration_step'] - self.hyperparameters['replay_start_size'],
                name="exploration",
                logger=logger
            )
        )

        if self.hyperparameters['her']:
            replay_buffer = HERReplayBuffer(
                size=self.hyperparameters['replay_buffer_size'],
                device=self.device,
                env=self.env,
                relabel_strategy=self.hyperparameters['relabel_strategy'], 
                num_relabel=self.hyperparameters['num_relabel']
            )
        else:
            replay_buffer = ExperienceReplayBuffer(
                size=self.hyperparameters['replay_buffer_size'],
                device=self.device
            )

        return GoalDQN(
            q,
            policy,
            replay_buffer,
            discount_factor=self.hyperparameters['discount_factor'],
            minibatch_size=self.hyperparameters['minibatch_size'],
            replay_start_size=self.hyperparameters['replay_start_size'],
            update_frequency=self.hyperparameters['update_frequency'],
        )

    def test_agent(self):
        q = QNetwork(copy.deepcopy(self.model).to(self.device))
        
        policy = GreedyPolicy(q, self.n_actions, epsilon=self.hyperparameters['test_exploration'])

        return GoalDQNTestAgent(policy)

if __name__ == "__main__":
    hyperparameters = parse_config(os.path.join(config_path, "goal-dqn-bitflip.yaml"))
    dqn = PresetBuilder('dqn', hyperparameters, GoalDQNPreset)
