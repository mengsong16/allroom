from all import nn
import numpy as np

# two layer MLP
def goal_fc_relu_q(env, hidden=64):
    state_dim = int(np.prod(env.state_space["observation"].shape))
    goal_dim = int(np.prod(env.goal_space.shape))
    
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(state_dim+goal_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, env.action_space.n),
    )