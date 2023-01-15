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

def clone_state_array(state_array):
    x = {}
    for key, value in state_array.items():
        x[key] = value
    
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
        x[key] = value
    
    
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

class GoalDQN(DQN):
    # state: State
    def act(self, state):
        #print(state.shape)
        
        # store (s,a,r,s')
        self.replay_buffer.store(self._state, self._action, state)
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

    # def __init__(self, env, name, device, **hyperparameters):
    #     super().__init__(name, device, hyperparameters)
    #     self.model = hyperparameters['model_constructor'](env).to(device)
    #     self.n_actions = env.action_space.n

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

        replay_buffer = ExperienceReplayBuffer(
            self.hyperparameters['replay_buffer_size'],
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
