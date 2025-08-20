import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """策略网络（Actor）

    在强化学习的Actor-Critic框架中，Actor负责根据当前状态生成动作，
    即学习“状态→动作”的映射关系，目标是输出能最大化累积奖励的动作。
    本实现适用于连续动作空间的多智能体雾计算环境（如资源分配、任务卸载决策）。
    """
    def __init__(self, state_dim, action_dim, max_action):
        """初始化策略网络
        Args:
            state_dim: 状态维度（输入特征数量，如雾节点的资源状态、任务属性等）
            action_dim: 动作维度（输出决策数量，如每个节点的资源分配比例）
            max_action: 动作最大值（用于缩放输出，控制动作幅度范围）
        """
        super(Actor, self).__init__()
        # 全连接层1：输入状态特征，输出256维隐藏特征
        self.fc1 = nn.Linear(state_dim, 256)
        # 全连接层2：输入256维特征，输出256维特征（加深网络提高表达能力）
        self.fc2 = nn.Linear(256,256)
        # 全连接层3：输出动作维度，与环境交互的决策值
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action# 记录动作最大值，用于最终缩放

    def forward(self, state):
        """前向传播计算动作

                Args:
                    state: 状态张量（shape: [batch_size, state_dim]）

                Returns:
                    动作张量（shape: [batch_size, action_dim]）：范围[-max_action, max_action]
        """
        x = F.relu(self.fc1(state)) # 第一层输出经ReLU激活（引入非线性）
        x = F.relu(self.fc2(x))# 第二层输出经ReLU激活
        # tanh激活将输出限制在[-1, 1]，再乘以max_action缩放至目标范围
        return self.max_action * torch.tanh(self.fc3(x))  # scale action to max_action

class Critic(nn.Module):
    """价值网络（Critic）
        在Actor-Critic框架中，Critic负责评估Actor生成的“状态-动作对”的价值（Q值），
        即预测在当前状态下执行该动作后能获得的累积奖励总和，为Actor的更新提供指导。
    """
    def __init__(self, state_dim, action_dim):
        """初始化价值网络

               Args:
                   state_dim: 状态维度（与Actor输入一致）
                   action_dim: 动作维度（与Actor输出一致）
        """
        super(Critic, self).__init__()
        # 全连接层1：输入为状态+动作的拼接（联合评估两者的价值），输出256维特征
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        # 全连接层2：输入256维特征，输出256维特征
        self.fc2 = nn.Linear(256,256)
        # 全连接层3：输出1维Q值（评估该状态-动作对的价值）
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        """前向传播计算Q值

                Args:
                    state: 状态张量（shape: [batch_size, state_dim]）
                    action: 动作张量（shape: [batch_size, action_dim]）

                Returns:
                    Q值张量（shape: [batch_size, 1]）：越大表示该状态-动作对越优
        """
        # 拼接状态和动作特征（dim=1表示按特征维度合并，保持batch_size不变）
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))# 第一层输出经ReLU激活
        x = F.relu(self.fc2(x))# 第二层输出经ReLU激活
        return self.fc3(x)# 输出Q值（无激活函数，因价值可正可负）
