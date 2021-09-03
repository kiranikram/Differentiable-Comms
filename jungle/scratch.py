import  numpy as np
import random
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
def get_obs(goal,pose):
    return np.hstack([goal, pose])
    
def seed( seed=1):
        random_state, seed = seeding.np_random(seed)
        return random_state

world_shape = [5, 5]
random_state = seed()
pose = random_state.uniform((0, 0), world_shape)
goal = random_state.randint((0, 0), world_shape)

global_state = np.zeros([2,4] + [3], dtype=np.uint8)

high=float(max[5,5])

obs = spaces.Box(low=0.0,high=float(max(5,5)),shape=(4,))