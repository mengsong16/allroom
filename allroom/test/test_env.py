import gym
from gym import spaces
import os
import numpy as np
import d4rl
from allroom.envs.common import *
from allroom.envs.gcsl_envs import goal_env
from allroom.envs.mazelab.env import EmptyMazeEnv, UMazeEnv, FourRoomEnv
from allroom.envs.bitflip import BitFlippingGymEnv
import highway_env
import multiworld.envs.gridworlds

def test_env(env_id, render=False, **kwarg):
    env = create_env(env_id, **kwarg)
    # if hasattr(env.unwrapped, 'random_start'):
    #     env.unwrapped.random_start = False
    # if hasattr(env.unwrapped, 'random_goal'):
    #     env.unwrapped.random_goal = False
    
    for episode in range(3):
        print("--------------------------------------")
        print('Episode: {}'.format(episode))
        state = env.reset()
        
        print("State: %s"%state)
        if is_instance_goalreaching_env(env):
            print("Achieved goal: %s"%env.achieved_goal)
            print("Desired goal: %s"%env.desired_goal)
        
        for i in range(500):  # max steps per episode
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            #print(info)
            #if is_instance_gcsl_env(env):
            #         print(env.goal_distance())
            #    print(env.is_success())
            if render:
                env.render()
            #print(state)
            #print(reward)
            if done:
                #print("success: %s"%info["is_success"])
                # if is_instance_gcsl_env(env):
                #     print(env.is_success())  
                break
                    
        print('Episode {} finished after {} timesteps.'.format(episode,i+1))
    
    print("***********************************")   
    print("Action space: %s"%(env.action_space)) 
    # action is continuous
    if isinstance(env.action_space, spaces.Box):
        act_dim = int(np.prod(env.action_space.shape))
    else:
        act_dim = env.action_space.n
    print("Action number: %d"%(act_dim))
    print("--------------------------------------")    
    print("Observation space: %s"%(env.observation_space)) 
    # assume state is continuous
    if isinstance(env.observation_space, spaces.Box):
        state_dim = int(np.prod(env.observation_space.shape))
        print("Observation dimension: %d"%(state_dim)) 
    print("--------------------------------------")
    if is_instance_goalreaching_env(env):
        print("Goal space: %s"%(env.goal_space))   
        if isinstance(env.goal_space, spaces.Box):
            goal_dim = int(np.prod(env.goal_space.shape))
            print("Goal dimension: %d"%(goal_dim))
    print("--------------------------------------")
    if hasattr(env.spec, "max_episode_steps") and env.spec.max_episode_steps is not None:
        print("Env timestep limit: %d"%env.spec.max_episode_steps)
    print("***********************************")    
    
    env.close() 

def test_envs(env_list):
    print('-----------------------------')
    for env_id in env_list:
        test_env(env_id = env_id)
        print("%s Done"%env_id)
        print('-----------------------------') 

def test_all_env(env_id, **kwarg):
    env = GoalGymEnvironment(id=env_id, device="cuda", **kwarg)
    
    print("====> Environment is wrapped for ALL: %s"%(env_id))

    for episode in range(2):
        print("--------------------------------------")
        print('Episode: {}'.format(episode))
        state = env.reset()
        print("State: %s"%state)

        for i in range(500):  # max steps per episode
            action = env.action_space.sample()
            state = env.step(action)

            #print("State: %s"%state)
            
            if state.done: 
                break
                    
        print('Episode {} finished after {} timesteps.'.format(episode,i+1))
    
    print('-----------------------------')
    print(env.observation_space)
    print(env.state_space)
    print(env.goal_space)
    print('-----------------------------')
    
    env.close() 

if __name__ == "__main__":  
    # ---------------- gym ---------------
    # 'LunarLander-v2'  # action: 4(d), state: 8(c), max_len: 1000
    # 'InvertedDoublePendulum-v2' # action: 1(c), state: 11(c), max_len: 1000
    # 'Swimmer-v2' # action: 2(c), state: 8(c), max_len: 1000
    # ---------------- d4rl ---------------
    # 'minigrid-fourrooms-v0' # action: 7(d), state: 7*7*3(c), max_len: 50
    # 'maze2d-umaze-v1' # action: 2(c), state: 4(c), max_len: 300
    # 'antmaze-umaze-v0' # action: 8(c), state: 29(c), max_len: 700
    # 'kitchen-complete-v0' # action: 9(c), state: 60(c), max_len: 280
    # ---------------- gcsl ---------------
    # 'pointmass-{rooms/wall/empty}-v0' # action: 2(c), state: 2(c) = goal: 2(c), fixed start
    # 'sawyerpush-v0' # action: 2(c), state: 4(c) = goal: 4(c), fixed start
    # 'sawyerdoor-v0' # action: 3(c), state: 4(c) != goal: 1(c), fixed start
    # 'claw-v0' # action: 9(c), state: 11(c) != goal: 2(c), fixed start
    # 'lunargoal-v0' # action: 4(d), state: 8(c) != goal: 5(c), random start
    # random goal
    # ---------------- gym goal reaching -----------
    # 'FetchPickAndPlace-v1' # action: 4(c), state: 25(c) = goal: 3(c), fixed start, fixed goal, max_len: 50
    # 'FetchPush-v1' , # action: 4(c), state: 25(c) != goal: 3(c), fixed start, fixed goal, max_len: 50
    # 'FetchReach-v1', # action: 4(c), state: 10(c) != goal: 3(c), fixed start, fixed goal, max_len: 50
    # 'FetchSlide-v1' # action: 4(c), state: 25(c) != goal: 3(c), fixed start, fixed goal, max_len: 50
    # 'parking-v0' # action: 2(c), state: 6(c) == goal: 6(c), random goal, max_len: unknown
    # has 'is_success' in info
    # ---------------- 2D maze -----------
    # 'empty-maze-v0': # action: 4(d), state: 2(c) = goal: 2(c), max_len: 200
    # 'umaze-v0': # action: 4(d), state: 2(c) = goal: 2(c), max_len: 200
    # 'four-room-v0': # action: 4(d), state: 2(c) = goal: 2(c), max_len: 200
    # fixed goal or not, fixed start or not
    # has 'is_success' in info
    # ---------------- bitflip -----------
    # 'bitflip-v0': # action: 20(d), state: 20(d) = goal: 20(d), max_len: 20
    # fixed goal or not, fixed start or not
    # has 'is_success' in info
    
    gym_env_list = ['LunarLander-v2', 'InvertedDoublePendulum-v2', 'Swimmer-v2']
    d4rl_env_list = ['minigrid-fourrooms-v0', 'maze2d-umaze-v1', 'antmaze-umaze-v0', 'kitchen-complete-v0'] # 'minigrid-fourrooms-random-v0'
    gcsl_env_list = ['pointmass-rooms-v0', 'pointmass-wall-v0', 'pointmass-empty-v0',
        'sawyerpush-v0', 'sawyerdoor-v0', 'claw-v0', 'lunargoal-v0']
    gym_goal_env_list = ['FetchPickAndPlace-v1', 'FetchPush-v1', 'FetchReach-v1', 'FetchSlide-v1']
    maze_list = ['empty-maze-v0', 'umaze-v0', 'four-room-v0']
    bit_flip = ['bitflip-v0']

    # test_envs(d4rl_env_list)
    # test_envs(gym_goal_env_list)
    # test_envs(gcsl_env_list)
    # test_envs(maze_list)

    #test_env('four-room-v0', render=False)
    #test_env('GoalGridworld-v0', render=True)
    #test_env("parking-v0", render=False)
    #env = create_env("parking-v0")
    #env = create_env('four-room-v0')
    #env = gym.make('four-room-v0')
    #env = gym.make("parking-v0")
    #print(get_wrapper_class(env))
    #test_env(env_id="pointmass-rooms-v0")
    #test_env(env_id="bitflip-v0", random_start=False, random_goal=False)

    test_all_env('four-room-v0')
    #test_all_env('bitflip-v0', random_start=False, random_goal=False)