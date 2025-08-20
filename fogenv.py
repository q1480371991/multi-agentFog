import numpy as np
import torch

class FogNode:
    """雾节点类，模拟雾计算环境中的单个节点实体
    该类封装了雾节点的硬件资源（CPU、内存、带宽）、通信属性（信噪比、链路质量）
    和任务处理能力，提供资源检查、任务分配、延迟计算等核心功能，是构建多节点环境的基础单元。
    """
    def __init__(self, task_size, task_deadline, cpu_req, dep_prob, cpu_avail, ram_avail, queue_len, 
                 offload_succ_rate, bandwidth, net_delay, link_fail_prob, snr, plr, jitter, channel_status,
                 received_power=0.01, noise_power=0.001):
        # 任务相关属性
        self.task_size = task_size # 任务数据大小
        self.task_deadline = task_deadline# 任务截止时间
        self.cpu_req = cpu_req# 任务CPU需求
        self.dep_prob = dep_prob# 任务依赖其他任务的概率（影响调度）

        # 节点资源属性
        self.cpu_avail = cpu_avail# 可用CPU资源
        self.ram_avail = ram_avail# 可用内存资源
        self.queue_len = queue_len# 任务队列长度
        self.offload_succ_rate = offload_succ_rate  # 任务卸载成功率（卸载到其他节点的成功率）
        self.bandwidth = bandwidth# 可用带宽
        self.net_delay = net_delay # 基础网络延迟（固定通信延迟）

        # 链路属性
        self.link_fail_prob = link_fail_prob# 链路失败概率（通信中断的可能性）
        self.snr = self.calculate_snr(received_power, noise_power) # 信噪比（通信质量指标）
        self.plr = plr# 分组丢失率（数据传输中数据包丢失比例）
        self.jitter = jitter  # 抖动（延迟变化量，影响实时性）
        self.channel_status = channel_status   # 信道状态（1:空闲，0:繁忙）

        # 物理参数
        self.SPEED_OF_LIGHT_FIBER = 3e8# 光纤中光速
        self.distance = torch.rand(1, device='cpu') * 90 + 10# 随机生成节点间距离（10-100单位）

    def calculate_snr(self, received_power, noise_power):
        """计算信噪比（线性值）   信噪比 = 接收功率 / 噪声功率，值越大表示通信质量越好"""
        return received_power / noise_power

    def calculate_snr_db(self, received_power, noise_power):
        """计算信噪比（分贝值） 分贝值信噪比 = 10 * log10(接收功率 / 噪声功率)，便于直观理解增益  """
        return 10 * np.log10(received_power / noise_power)

    def can_handle_task(self, task_cpu, task_bandwidth, task_ram):
        """判断节点是否有足够资源处理任务  检查当前可用CPU、带宽和内存是否满足任务需求，返回布尔值"""
        return (self.cpu_avail >= task_cpu and# CPU足够
                self.bandwidth >= task_bandwidth and # 带宽足够
                self.ram_avail >= task_ram)# 内存足够

    def assign_task(self, task_cpu, task_bandwidth, task_ram):
        """为节点分配任务，消耗相应资源 若资源充足则分配任务并扣减资源，返回分配成功与否的标志"""
        if self.can_handle_task(task_cpu, task_bandwidth, task_ram):
            self.cpu_avail -= task_cpu# 减少可用CPU
            self.bandwidth -= task_bandwidth# 减少可用带宽
            self.ram_avail -= task_ram# 减少可用内存
            return True
        return False
    
    def reset(self):
        """重置节点资源到初始状态"""
        self.cpu_avail = 3.0  # 重置CPU容量
        self.bandwidth = 100   # 重置带宽
        self.ram_avail = 2.0   # 重置内存

    def calculate_pd(self, distance):
        """计算传播延迟（基于距离和光速） 传播延迟 = 距离 / 光速"""
        return distance / self.SPEED_OF_LIGHT_FIBER

    def calculate_transmission_latency(self, task_size, bandwidth, distance):
        """计算传输延迟 = 传输时间 + 传播延迟"""
        pd = self.calculate_pd(distance)# 传播延迟 传播延迟 = 距离 / 光速（物理信号传播耗时）
        return task_size / bandwidth + pd# 传输时间（数据量/带宽）+ 传播延迟


class FogToFogEnv:
    """雾到雾计算环境类，模拟多雾节点协同的任务分配与资源管理场景
        该类是强化学习智能体的交互接口，负责：
        1. 管理多个FogNode实例组成的节点网络
        2. 生成任务并根据智能体动作分配到节点
        3. 计算任务处理的延迟、能耗等指标
        4. 基于指标生成奖励信号，引导智能体学习优化策略
    """
    def __init__(self, num_fog_nodes=6, arrival_rate=3 * 10**3, task_execution_rate=2.0):
        self.num_fog_nodes = num_fog_nodes# 雾节点数量（默认6个，对应6个智能体）
        self.arrival_rate = arrival_rate  # 任务到达率（泊松分布参数，控制任务生成频率）
        self.task_execution_rate = task_execution_rate #任务执行率（单位时间处理任务数）
        # 初始化雾节点列表（参数为初始默认值）
        self.fog_nodes = [FogNode(0, 0, 0, 0, 3.0, 2.0, 4, 0.8, 100, 30, 0.02, 1.0, 0.01, 1.0, 1) for _ in range(num_fog_nodes)]
        self.task_counter = 0# 任务计数器（记录总处理任务数）
        self.result_ratio = 0.1# 任务结果数据量与原任务的比例（影响反馈数据传输）
        self.state = self._generate_initial_state() # 初始状态
        # 评估指标的最大值（用于归一化，避免数值范围波动过大）
        self.max_latency = 100  # 最大延迟阈值（占位符）
        self.max_energy = 10  # 最大能量消耗阈值（占位符）
        self.max_successful_tasks = 10   #最大成功任务数阈值（占位符）
        # 奖励平滑参数
        self.smoothed_reward = 0
        self.previous_energy = 0  # 用于能量平滑的历史值

    def _generate_initial_state(self):
        """生成环境初始状态，包含每个雾节点的12个特征"""
        # 状态特征涵盖节点的通信状态、资源能力、任务属性等，是智能体决策的输入依据
        # 生成各特征的随机初始值（基于正态分布或离散选择）
        self.S_channel = np.random.choice([0, 1], size=(self.num_fog_nodes,)) # 信道状态（0/1）
        self.S_power = np.random.normal(3.0, 0.5, size=(self.num_fog_nodes,))# 功率
        self.S_gain = np.random.normal(1.0, 0.2, size=(self.num_fog_nodes,)) # 增益
        self.S_size = np.random.normal(25, 10, size=(self.num_fog_nodes,))# 任务大小
        self.S_cycle = np.random.normal(3.0, 0.5, size=(self.num_fog_nodes,))# 周期
        self.S_ddl = np.random.normal(25, 10, size=(self.num_fog_nodes,)) # 截止时间
        self.S_res = np.random.normal(2.0, 0.5, size=(self.num_fog_nodes,)) # 资源
        self.S_com = np.random.normal(100, 20, size=(self.num_fog_nodes,)) # 通信能力
        self.S_epsilon = np.random.normal(0.01, 0.005, size=(self.num_fog_nodes,))# 小参数
        self.S_new_feature1 = np.random.normal(1.0, 0.2, size=(self.num_fog_nodes,))# 新特征1
        self.S_new_feature2 = np.random.normal(1.0, 0.2, size=(self.num_fog_nodes,))# 新特征2
        self.S_new_feature3 = np.random.normal(1.0, 0.2, size=(self.num_fog_nodes,))# 新特征3

        # 组合所有特征，形成每个节点的状态向量（12维）
        return np.array([
        [self.S_channel[n], self.S_power[n], self.S_gain[n], self.S_size[n], self.S_cycle[n],
         self.S_ddl[n], self.S_res[n], self.S_com[n], self.S_epsilon[n],
         self.S_new_feature1[n], self.S_new_feature2[n], self.S_new_feature3[n]]
        for n in range(self.num_fog_nodes)
        ])

    def calculate_reward(self, total_latency, total_energy, successful_tasks):
        """计算奖励值，综合考虑延迟、能量和任务成功率"""
        # 奖励因子（权重）
        latency_penalty_factor = 0.6# 延迟惩罚权重
        energy_penalty_factor = 0.18# 能量惩罚权重
        success_reward_factor = 0.3# 成功任务奖励权重

        # 能量平滑（减少波动）
        smoothed_energy = self.smooth_energy(total_energy)
        # 延迟惩罚（超过阈值时惩罚增大）
        latency_penalty = np.log1p(total_latency) * latency_penalty_factor if total_latency > self.max_latency else 1
        # 能量惩罚
        energy_penalty = np.log1p(smoothed_energy) * energy_penalty_factor
        # 任务成功奖励
        task_success_bonus = success_reward_factor * (successful_tasks / self.max_successful_tasks)
        # 综合奖励计算（加权组合）
        reward = 0.4 * task_success_bonus + 0.2 * latency_penalty + 0.1 * energy_penalty
        # 奖励平滑（指数移动平均）
        self.smoothed_reward = 0.90 * self.smoothed_reward + 0.1 * reward
        # 奖励裁剪到[-1, 1]范围
        return np.clip(self.smoothed_reward, -1, 1)

    def smooth_energy(self, current_energy, smoothing_factor=0.1):
        """对能量消耗进行指数平滑，减少瞬时波动"""
        smoothed_energy = smoothing_factor * current_energy + (1 - smoothing_factor) * self.previous_energy
        self.previous_energy = smoothed_energy   # 更新历史值
        return smoothed_energy

    def reset(self):
        """重置环境到初始状态"""
        self.task_counter = 0# 重置任务计数器
        for node in self.fog_nodes:
            node.reset()# 重置所有雾节点资源
        self.previous_energy = 0  # 重置能量平滑历史值
        return self._generate_initial_state()# 返回新的初始状态

    def step(self, actions):
        """执行一步环境交互，处理任务分配和资源更新"""
        # 随机生成任务需求参数（CPU、带宽、内存、大小）
        task_cpu, task_bandwidth, task_ram, task_size = np.random.uniform(0.5, 2.0, 4)
        self.task_counter += 1# 任务计数+1

        # 基于泊松分布生成到达的任务数量
        num_tasks_arrived = np.random.poisson(self.arrival_rate)

        # 若没有任务到达，直接返回当前状态
        if num_tasks_arrived == 0:
            return self.state, 0, False, {}
        # 处理动作：裁剪到[0,1]范围，归一化得到卸载概率
        actions = np.clip(np.array(actions).flatten(), 0, 1)
        action_sum = np.sum(actions)
        # 若动作和为0，则平均分配概率；否则归一化
        offloading_probabilities = actions / action_sum if action_sum > 0 else np.ones_like(actions) / len(actions)
        # 初始化评估指标
        total_latency, total_energy, successful_tasks = 0, 0, 0

        # 计算任务执行率（基于可用资源的处理能力）
        task_execution_rate = min(self.task_execution_rate, self.fog_nodes[0].cpu_avail)

        # 计算本地处理的任务资源需求（扣除卸载部分）
        local_cpu = task_cpu * max(0, 1 - np.sum(offloading_probabilities))  # 确保非负
        local_bandwidth = task_bandwidth * max(0, 1 - np.sum(offloading_probabilities))
        local_ram = task_ram * max(0, 1 - np.sum(offloading_probabilities))

        # 处理本地任务（第0个节点）
        if self.fog_nodes[0].can_handle_task(local_cpu, local_bandwidth, local_ram):
            self.fog_nodes[0].assign_task(local_cpu, local_bandwidth, local_ram) # 分配资源
            
            # Recalculate available CPU after assignment  计算本地任务的延迟和能量消耗
            available_cpu = self.fog_nodes[0].cpu_avail + local_cpu  # 调整后的可用CPU
            total_latency += local_cpu / max(available_cpu, 1e-6)  # 避免除零错误
            total_energy += available_cpu * local_cpu * 1e9  # Corrected energy calculation 能量计算
            successful_tasks += 1# 本地任务成功计数
            # 处理卸载到其他节点的任务
            for i in range(self.num_fog_nodes):
                # 若有卸载概率且信道空闲
                if offloading_probabilities[i] > 0 and self.fog_nodes[i].channel_status == 1:
                    # 计算执行延迟（任务CPU需求/节点可用CPU）
                    execution_latency = task_cpu / max(self.fog_nodes[i].cpu_avail, 1e-6)
                    # 计算传输延迟（任务传输+结果回传）
                    transmission_latency = self.fog_nodes[i].calculate_transmission_latency(task_size, self.fog_nodes[i].bandwidth, self.fog_nodes[i].distance.item())
                    result_transmission_latency = self.fog_nodes[i].calculate_transmission_latency(task_size * self.result_ratio, self.fog_nodes[i].bandwidth, self.fog_nodes[i].distance.item())

                    # 总任务延迟取各部分最大值
                    total_task_latency = transmission_latency + execution_latency + result_transmission_latency
                    total_latency = max(total_latency, total_task_latency)  # Take the max latency

                    # 计算传输和执行的能量消耗
                    transmission_energy = transmission_latency * self.fog_nodes[i].bandwidth * 1e-9
                    execution_energy = execution_latency * self.fog_nodes[i].cpu_avail * 1e-9
                    total_energy += transmission_energy + execution_energy
                    
                    successful_tasks += 1# 卸载任务成功计数
        # 计算奖励
        reward = self.calculate_reward(total_latency, total_energy, successful_tasks)
        # 判断回合是否结束（任务数达到100）
        done = self.task_counter >= 100
        # 生成下一状态
        next_state = self._generate_initial_state()
        # 打包信息（奖励、延迟、能量、成功任务数）
        info = {"Reward": reward, "Total Latency": total_latency, "Total Energy": total_energy, "Successful Tasks": successful_tasks}
        return next_state, reward, done, info
