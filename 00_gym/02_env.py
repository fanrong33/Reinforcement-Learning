# encoding: utf-8
"""
gym的核心接口是Env，作为统一的环境接口。Env包含下面几个核心方法：

1、reset(self): 重置环境的状态，返回观察。
2、step(self,action): 推进一个时间步长，返回observation，reward，done，info
3、render(self,mode=’human’,close=False): 重绘环境的一帧。

http://blog.csdn.net/cs123951/article/details/71171260

"""

import gym

env = gym.make('CartPole-v0')

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random action

