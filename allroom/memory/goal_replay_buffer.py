
import numpy as np
import torch
from all.memory.replay_buffer import ExperienceReplayBuffer
from all.core import State
import random
import math


# For goal conditioned RL: separate s and g (although may not necessary)
# (s,a,r,s',g)
# when store, s, s' are States, a and g are arrays, r are scalars
# after sampling (reshape), s, s', g are all single State
# agent has two method to get s and g: separate or concatenate
class GoalConditionedReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, size, device=torch.device('cpu')):
        super().__init__(size, device)

    # add one (s,a,r,s',g) sample 
    # s,s' are single State, a and g are arrays
    def store(self, state, action, reward, next_state, goal_state):
        if state is not None and not state.done:
            self._add((state, action, reward, next_state, goal_state))

    def _reshape(self, minibatch, weights):
        states = State.from_list([sample[0] for sample in minibatch])
        # stack actions
        actions = torch.cat([sample[1] for sample in minibatch])
        rewards = torch.tensor([sample[2] for sample in minibatch], device=self.device).float()
        next_states = State.from_list([sample[3] for sample in minibatch])
        goal_states = State.from_list([sample[4] for sample in minibatch])
        #goals = np.vstack((sample[4] for sample in minibatch))
        #print(np.shape(goals))
        #goal_states = self._make_goal(goals)

        return (states, actions, rewards, next_states, goal_states, weights)

        

# HER
# derived from GoalConditionedReplayBuffer
class HERReplayBuffer(GoalConditionedReplayBuffer):
    def __init__(self, size, env, relabel_strategy="future", num_relabel=4, device=torch.device('cpu')):
        super().__init__(size, device)
        self.env = env
        self.relabel_strategy = relabel_strategy
        self.num_relabel = num_relabel

    # relabel each state of this trajectory with a new goal for k times
    def relabel(self, trajectory):
        N = len(trajectory)

        # trajectory should have at least two points
        if N < 2:
            return
        
        # relabel until the second to last point
        for i, p in enumerate(trajectory[:-1]):
            # (0-s, 1-a, 2-r, 3-s', 4-achieved_g_s, 5-achieved_g_s')
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
                elif self.relabel_strategy == "next":
                    # [i+1]
                    new_goal_index = i+1
                elif self.relabel_strategy == "next_next":
                    # [i+1, i+2]
                    new_goal_index = np.random.randint(i+1, min(i+3,N))         
                else:
                    print("Invalid relabel strategy HER not used") 
                    return

                # compute new reward
                new_reward = self.env.compute_reward(p[5], trajectory[new_goal_index][4], p[0].info)    
                # store new data point: (s,a,r,s',desired_goal)
                self.store(p[0], p[1], new_reward, p[3], self.env._make_goal(trajectory[new_goal_index][4]))
    
