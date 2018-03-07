# encoding: utf-8
""" CartPole-v0
一个杆通过一个非致动关节连接到一个沿无摩擦轨道移动的小车上。
该系统通过施加+1或-1的力来控制。
钟摆开始直立，目标是防止它翻倒。
为每杆提供+1的奖励，杆保持直立。
当杆子与垂直方向相差超过15度时，情节结束，或者车子从中心移动超过2.4个单位。

http://blog.csdn.net/cs123951/article/details/71171260

"""

import gym

env = gym.make('CartPole-v0')
env = env.unwrapped  # 不做这个会有很多限制

print(env.action_space.n)           # 2         动作空间的左右两个动作，还是得看下源码
print(env.observation_space)        # Box(4,)   观察空间的四个变量
print(env.observation_space.high)   # [  4.80000019e+00   3.40282347e+38   4.18879032e-01   3.40282347e+38]
print(env.observation_space.low)    # [ -4.80000019e+00  -3.40282347e+38  -4.18879032e-01  -3.40282347e+38]
print(env.action_space.sample())
