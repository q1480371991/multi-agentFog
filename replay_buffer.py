import random
import numpy as np

class ReplayBuffer:
    """经验回放缓冲区

        用于存储智能体与环境交互产生的经验（状态、动作、奖励等），
        并支持随机采样批量经验用于训练，打破样本间的相关性，提高训练稳定性。
    """
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        """初始化缓冲区

                Args:
                    state_dim: 状态维度（输入特征数量）
                    action_dim: 动作维度（输出决策数量）
                    max_size: 缓冲区最大容量（超过后覆盖旧数据）
        """
        self.state_dim = state_dim # 状态维度（记录用于校验）
        self.action_dim = action_dim# 动作维度（记录用于校验）
        self.max_size = max_size # 最大容量（默认100万条经验）
        self.buffer = [] # 存储经验的列表，每条经验为元组
        self.idx = 0# 循环指针（用于覆盖旧数据）
        # 注意：原代码中存在笔误，`pointer`未在__init__中定义，实际应使用self.idx

    def add(self, state, action, next_state, reward, done):
        """添加一条经验到缓冲区

               Args:
                   state: 当前状态（s）
                   action: 执行的动作（a）
                   next_state: 执行动作后的状态（s'）
                   reward: 获得的奖励（r）
                   done: 是否结束标志（终端状态标记）
        """
        if len(self.buffer) < self.max_size:
            # 缓冲区未满时直接追加
            self.buffer.append((state, action, next_state, reward, done))
        else:
            # 缓冲区已满时覆盖最旧的数据（循环队列逻辑）
            self.buffer[self.idx] = (state, action, next_state, reward, done)
        # 更新指针（取模保证在0~max_size-1范围内）
        # self.pointer = (self.pointer + 1) % self.max_size
        self.idx = (self.idx + 1) % self.max_size
        # 原代码中使用self.pointer，此处修正为self.idx以保持一致性


    def sample(self, batch_size):
        """随机采样一批经验

                Args:
                    batch_size: 采样的批量大小

                Returns:
                    包含状态、动作、下一状态、奖励、结束标志的numpy数组元组
        """
        # 从缓冲区中随机选择batch_size条经验
        batch = random.sample(self.buffer, batch_size)
        # 解压批量经验，将同类型数据聚合为列表
        states, actions, next_states, rewards, dones = zip(*batch)
        # 转换为numpy数组返回（便于后续转换为张量输入网络）
        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)

    def size(self):
        """返回当前缓冲区中的经验数量"""
        return len(self.buffer)
