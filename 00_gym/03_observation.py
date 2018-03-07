# encoding: utf-8
""" 观察

如果我们想做得好一点，观察周围的环境是必须的。环境的step函数返回四个值：

Observation(object): 返回一个特定环境的对象，描述对环境的观察。比如，来自相机的像素数据，机器人的关节角度和关节速度，或棋盘游戏中的棋盘状态。
Reward(float): 返回之前动作收获的总的奖励值。不同的环境计算方式不一样，但总体的目标是增加总奖励。
Done(boolean): 返回是否应该重新设置（reset）环境。大多数游戏任务分为多个环节（episode），当done=true的时候，表示这个环节结束了。
Info(dict): 用于调试的诊断信息（一般没用）。

"""

import gym
import time

env = gym.make('CartPole-v0')

for i_episode in range(10):
    observation = env.reset()
    # t_done = 0
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        x, x_dot, theta, theta_dot = observation   # 细分开, 为了修改原配的 reward

        # x 是车的水平位移[-2.5, 2.5], 所以 r1 是车越偏离中心, 分越少
        # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高

        # print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            # if t_done == 0:
            #     t_done = t
            # time.sleep(2)
            # if t_done - t > 5:
            #     t_done = 0
            break

