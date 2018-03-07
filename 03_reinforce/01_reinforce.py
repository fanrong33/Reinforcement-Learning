# encoding: utf-8
"""

强化学习是一个通过奖惩来学习正确行为的机制. 家族中有很多种不一样的成员, 有学习奖惩值, 根据自己认为的高价值选行为,
比如 Q learning, Deep Q Network, 也有不通过分析奖励值, 直接输出行为的方法, 这就是今天要说的 Policy Gradients 了.
甚至我们可以为 Policy Gradients 加上一个神经网络来输出预测的动作. 对比起以值为基础的方法, Policy Gradients
直接输出动作的最大好处就是, 它能在一个连续区间内挑选动作, 而基于值的, 比如 Q-learning, 它如果在无穷多的动作中计算价值,
从而选择行为, 这, 它可吃不消.

https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/5-1-A-PG/

"""
import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()
# print(args)
''' Namespace(gamma=0.99, log_interval=10, render=False, seed=543) '''


env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def choose_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    # 让 Policy 神经网络生成所有 action 的值, 并选择值最大的 action
    probs = policy(Variable(state))

    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.data[0]


def learn():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()

    # 计算，更新神经网络参数
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]



def main():
    running_reward = 10
    for i_episode in count(1):
        # 初始化环境
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
        # while True:
            # 刷新环境
            # if args.render:
            #     env.render()
            # env.render()

            # Policy 选择行为
            action = choose_action(state)

            # 环境根据行为给出下一个 state, reward, 是否终止
            state_, reward, done, info = env.step(action)

            policy.rewards.append(reward)

            state = state_

            if done:
                break
        
        running_reward = running_reward * 0.99 + t * 0.01
        learn()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
