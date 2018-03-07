# encoding: utf-8
"""
Dependencies:
torch: 0.3
gym: 0.8.1
numpy
"""
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import numpy as np
import gym

# 超参数
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy 最优选择动作百分比 e_greedy
GAMMA = 0.9  # reward discount 奖励递减参数 reward decay
TARGET_REPLACE_ITER = 100  # target update frequency   Q 现实网络的更新频率
# 每 100 步替换一次 target_net 的参数
MEMORY_CAPACITY = 2000  # 记忆库大小  记忆上限
env = gym.make('CartPole-v0')  # 立杆子游戏
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 杆子能做的动作，2
N_STATES = env.observation_space.shape[0]  # 杆子能获取的环境信息数，4
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
                              int) else env.action_space.sample().shape  # to confirm the shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


"""
简化的DQN体系是这样，我们有两个net, 有选动作机制，有存经历机制, 有学习机制
class DQN(object):
    def __init__(self):
        # 建立 target net 和 eval net 还有 memory

    def choose_action(self, x):
        # 根据环境观测值选择动作的机制
        return action

    def store_transition(self, s, a, r, s_):
        # 存储记忆

    def learn(self):
        # target 网络更新
        # 学习记忆库中的记忆
"""


class DQN(object):
    def __init__(self):
        # 比较推荐的方式是搭建两个神经网络,
        # target_net 用于预测 q_target 值, 他不会及时更新参数.
        # eval_net 用于预测 q_eval, 这个神经网络拥有最新的神经网络参数.

        # 两个神经网络是为了固定住一个神经网络(target_net) 的参数, target_net 是 eval_net 的一个历史版本,
        # 拥有 eval_net 很久之前的一组参数, 而且这组参数被固定一段时间, 然后再被 eval_net 的新参数所替换.
        # 而 eval_net 是不断在被提升的, 所以是一个可以被训练的网络 trainable = True.而
        # target_net 的 trainable = False.

        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # 用于 target 更新计时
        self.memory_counter = 0  # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # torch 的优化器
        self.loss_func = nn.MSELoss()  # 误差公式

    def choose_action(self, x):  # x 为 observation
        # 统一 observation 的 shape (1, size_of_observation)
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))

        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:  # random 随机选择
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    # DQN 的精髓部分之一: 记录下所有经历过的步, 这些步可以进行反复的学习, 所以是一种 off-policy 方法,
    # 你甚至可以自己玩, 然后记录下自己玩的经历, 让这个 DQN 学习你是如何通关的
    def store_transition(self, s, a, r, s_):
        # 记录一条 [s a r s_] 记录
        transition = np.hstack((s, [a, r], s_))

        # 总memory 大小是固定的，如果超出总大小，旧 memory 就被新 memory 替换
        # 人的记忆不可能无穷，总有上限，旧的记忆会被新的覆盖
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):  # 最重要的一步，就是在 DeepQNetwork 中, 是如何学习, 更新参数的. 这里涉及了 target_net 和 eval_net 的交互使用
        # 检查是否更新替换 target_net 目标神经网络的参数
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 从 memory 中随机抽取 batch_size 这么多记忆
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
        # 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
        # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
        # 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
        # q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.
        # q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
        # 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.

        # 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
        # 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
        # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
        # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.
        # 具体在下面还有一个举例说明.

        # 从做过的动作 b_a 经验中来选 q_eval 的值，(q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(400):
    # 初始化环境
    s = env.reset()
    ep_r = 0
    while True:
        # 刷新环境
        env.render()

        # DQN 根据观测值选择行为
        a = dqn.choose_action(s)

        # 环境根据行为给出下一个 state, reward, 是否终止
        s_, r, done, info = env.step(a)

        # 修改 reward, 使 DQN 快速学习
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        # DQN 存储记忆
        dqn.store_transition(s, a, r, s_)

        # 累计奖励
        ep_r += r

        # 控制学习起始时间和频率（先积累一些记忆再开始学习）学习需要一段时间记忆，没有记忆怎么学习，下秒就忘？
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: %d, Ep_r: %.2f' % (i_episode, round(ep_r, 2)))
                # episode 插曲 关卡 回合

        # 将下一个 state_ 变为 下次循环的 state
        s = s_

        # 如果回合结束，进入下回合
        if done:
            break
