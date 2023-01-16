import numpy as np
import random
import gym
from gym.spaces import Dict, Box, Discrete
from gym import spaces
import torch

from allroom.envs.gcsl_envs.room_env import PointmassGoalEnv, PointmassRoomsEnv
from allroom.envs.gcsl_envs.sawyer_push import SawyerPushGoalEnv
from allroom.envs.gcsl_envs.sawyer_door import SawyerDoorGoalEnv
from allroom.envs.gcsl_envs.lunarlander import LunarEnv
from allroom.envs.gcsl_envs.claw_env import ClawEnv
from allroom.envs.gcsl_envs import goal_env
from allroom.envs.mazelab.env import EmptyMazeEnv, UMazeEnv, FourRoomEnv
from allroom.envs.bitflip import BitFlippingGymEnv

from all.environments.gym import GymEnvironment
from all.environments.duplicate_env import DuplicateEnvironment
from all.core.state import State

class GymToGoalReaching(gym.ObservationWrapper):
    """Wrap a Gym env into a GoalReaching env.
       Need to provide goal_space, achieved_goal, desired_goal
    """

    def __init__(self, env):
        super(GymToGoalReaching, self).__init__(env)
        
        self.achieved_goal = None
        self.desired_goal = None

        assert self.env.observation_space["achieved_goal"] == self.env.observation_space["desired_goal"]
        self.goal_space = self.env.observation_space["achieved_goal"]

        self.observation_space = self.env.observation_space

    def reset(self):
        observation = super(GymToGoalReaching, self).reset()
        
        return observation

    def step(self, action):
        """Take a step in the environment."""
        observation, reward, done, info = super(GymToGoalReaching, self).step(action)

        return observation, reward, done, info

    # return dictionary {observation, achieved_goal, desired_goal}
    def observation(self, state):
        """Fetch the environment observation."""
        self.achieved_goal = state['achieved_goal']
        self.desired_goal = state['desired_goal']

        return state
          
    

class GCSLToGoalReaching(gym.ObservationWrapper):
    """Wrap a GCSL env into a GoalReaching env.
       Need to provide goal_space, achieved_goal, desired_goal
    """

    def __init__(self, env):
        super(GCSLToGoalReaching, self).__init__(env)
        
        self.achieved_goal = None
        self.desired_goal = None

        self.observation_space = spaces.Dict(
            observation=self.env.observation_space,
            achieved_goal=self.env.goal_space,
            desired_goal=self.env.goal_space
        )

        self.goal_space = self.env.goal_space

        self.set_goal_threshold()

    def set_goal_threshold(self):
        env_name = self.env.spec.id
        
        self.env.spec.max_episode_steps = 1e6
        if env_name == 'pusher':
            self.goal_threshold = 0.05
        elif 'pick' in env_name:
            self.goal_threshold = 0.05
        elif env_name == 'door':
            self.goal_threshold = 0.05
        elif 'pointmass' in env_name:
            self.goal_threshold = 0.08
            self.env.spec.max_episode_steps = 2e5
        elif env_name == 'lunar':
            self.goal_threshold = 0.08
            self.env.spec.max_episode_steps = 2e5
        elif env_name == 'claw':
            self.goal_threshold = 0.1
        else:
            self.goal_threshold = 0.05

    def reset(self):
        """Reset the environment and the desired goal"""
        desired_goal_state = self.env.sample_goal()
        self.desired_goal = self.env.extract_goal(desired_goal_state)

        observation = super(GCSLToGoalReaching, self).reset()
        
        return observation

    def step(self, action):
        """Take a step in the environment."""
        observation, reward, done, info = super(GCSLToGoalReaching, self).step(action)

        info["is_success"] = self.is_success()

        return observation, reward, done, info

    def observation(self, state):
        """Fetch the real environment observation and achieved goal"""
        self.achieved_goal = self.env.extract_goal(state)

        assert self.achieved_goal.shape[0] == self.desired_goal.shape[0]

        state_dict = {"observation":self.env.observation(state), 
            "achieved_goal":self.achieved_goal.copy(), 
            "desired_goal": self.desired_goal.copy()}

        return state_dict
    
    # Euclidean distance
    def goal_distance(self):
        diff = self.achieved_goal - self.desired_goal
        #return np.linalg.norm(diff, axis=-1) 
        return np.linalg.norm(diff.flatten(), axis=-1) 

    def is_success(self):
        dist = self.goal_distance()
        success = (dist < self.goal_threshold)

        return success


def is_instance_gym_goal_env(env): 
    if not isinstance(env.observation_space, gym.spaces.Dict):
        return False

    if "achieved_goal" in env.observation_space.spaces.keys() and "desired_goal" in env.observation_space.spaces.keys():
        return True
    else:
        return False

def is_instance_gcsl_env(env):
    # Note that gym.make returns a gym wrapper instead of the real class
    # unwrapped removes the register wrapper
    if not isinstance(env.unwrapped, goal_env.GoalEnv):
        return False
    else:
        return True   

def is_instance_goalreaching_env(env):
    if isinstance(env, GCSLToGoalReaching) or isinstance(env, GymToGoalReaching):
        return True
    else:
        return False    

def create_env(env_id):
    env = gym.make(env_id)
    # wrap goal conditioned env
    if is_instance_gym_goal_env(env):
        env = GymToGoalReaching(env)
    elif is_instance_gcsl_env(env):
        env = GCSLToGoalReaching(env)
    else:
        print("The environment in neither a GCSL env nor a Gym goal env")

    print("======> Environment created: %s"%(env_id))

    return env	

# only consider wrapper GymToGoalReaching or GCSLToGoalReaching
def get_wrapper_class(env):
    if isinstance(env, GymToGoalReaching) or isinstance(env, GCSLToGoalReaching):
        return type(env)
    else:    
        return None

# wrapper goal conditioned env for All
class GoalGymEnvironment(GymEnvironment):
    def __init__(self, id, device, name=None):
        self._env = create_env(id)
        # check whether the environment is goal conditioned
        if not is_instance_goalreaching_env(self._env):
            print("Error: not a goal conditioned environment")
            exit()

        self._id = id
        self._name = name if name else id
        self._state = None
        self._action = None
        self._reward = None
        self._done = True
        self._info = None
        self._device = device
    
    def reset(self):
        dict_state = self._env.reset()
        # Have to put each value to the correct device
        self._state = State(x={
            'observation': torch.from_numpy(
                np.array(dict_state['observation'],
                dtype=np.float32)
            ).to(self._device),
            'done': False,
            'reward': -1.,
            'desired_goal': torch.from_numpy(
                np.array(dict_state['desired_goal'],
                dtype=np.float32)
            ).to(self._device),
            'achieved_goal': torch.from_numpy(
                np.array(dict_state['achieved_goal'],
                dtype=np.float32)
            ).to(self._device),
            'is_success': False # assert info has key 'is_success'
        },device=self._device)

        #print("reset to done: "+str(self._device))
        
        return self._state

    def step(self, action):
        dict_state, reward, done, info = self._env.step(self._convert(action))
        # Have to put each value to the correct device
        data = {
            'observation': torch.from_numpy(
                np.array(dict_state['observation'],
                dtype=np.float32)
            ).to(self._device),
            'done': done,
            'reward': float(reward), # the reward got at current step
            'desired_goal': torch.from_numpy(
                np.array(dict_state['desired_goal'],
                dtype=np.float32)
            ).to(self._device),
            'achieved_goal': torch.from_numpy(
                np.array(dict_state['achieved_goal'],
                dtype=np.float32)
            ).to(self._device)
        }

        for key in info:
            data[key] = info[key]

        self._state = State(x=data, device=self._device)

        return self._state
    
    # for vector env
    def duplicate(self, n):
        return DuplicateEnvironment([GoalGymEnvironment(self._id, device=self.device, name=self._name) for _ in range(n)])
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._env.compute_reward(achieved_goal, desired_goal, info)

    @property
    def observation_space(self):
        return self._env.observation_space
    
    @property
    def goal_space(self):
        return self._env.goal_space

