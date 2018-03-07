"""
"""

import gym
from gym import spaces

env = gym.make('CartPole-v0')

space = spaces.Discrete(8)  # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
print(x)
assert space.contains(x)
assert space.n == 8
