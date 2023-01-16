
import numpy as np
import torch
from all.memory.replay_buffer import ExperienceReplayBuffer
from allroom.utils.data_utils import clone_state
import random
import math
import copy


# for HER
class HERReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, size, env, device, relabel_strategy, num_relabel):
        super().__init__(size, device)
        self.env = env
        self.relabel_strategy = relabel_strategy
        self.num_relabel = num_relabel

    # relabel each state of this trajectory with a new goal for self.num_relabel times
    # and store them to the replay buffer
    def relabel(self, state_trajectory, action_trajectory):
        N = len(state_trajectory)
        assert N==len(action_trajectory), "The state trajectory and the action trajectory should have the same length: %d, %d"%(N, len(action_trajectory))
        
        # trajectory should have at least two states
        if N < 2:
            return
        
        # relabel from s0 until the second to the last point
        for i, s in enumerate(state_trajectory[:-1]):
            for k in list(range(self.num_relabel)):
                # sample new goal
                if self.relabel_strategy == "final":
                    # [N-1]
                    new_goal_index = -1
                elif self.relabel_strategy == "future":
                    # [i+1, N-1]   
                    new_goal_index = np.random.randint(i+1, N)
                elif self.relabel_strategy == "random":
                    # [0, N-1]
                    new_goal_index = np.random.randint(0, N)          
                else:
                    print("Error: Undefined relabel strategy") 
                    exit()

                # compute new reward under new goal
                # Note that the reward is calculated and measured at s'
                new_reward = self.env.compute_reward(
                    achieved_goal=state_trajectory[i+1]['achieved_goal'], 
                    desired_goal=state_trajectory[new_goal_index]['achieved_goal'], 
                    info=None)
                
                # store new data point: (s,a,s')
                new_s = clone_state(s)
                new_s['desired_goal'] = copy.deepcopy(state_trajectory[new_goal_index]['achieved_goal'])
                new_next_s = clone_state(state_trajectory[i+1])
                new_next_s['desired_goal'] = copy.deepcopy(state_trajectory[new_goal_index]['achieved_goal'])
                new_next_s['reward'] = new_reward  # Note thta new_reward is obtained at s', not s
                self.store(new_s, action_trajectory[i], new_next_s)
    
