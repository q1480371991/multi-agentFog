import numpy as np
import torch

class FogNode:
    def __init__(self, task_size, task_deadline, cpu_req, dep_prob, cpu_avail, ram_avail, queue_len, 
                 offload_succ_rate, bandwidth, net_delay, link_fail_prob, snr, plr, jitter, channel_status,
                 received_power=0.01, noise_power=0.001):
        self.task_size = task_size
        self.task_deadline = task_deadline
        self.cpu_req = cpu_req
        self.dep_prob = dep_prob
        self.cpu_avail = cpu_avail
        self.ram_avail = ram_avail
        self.queue_len = queue_len
        self.offload_succ_rate = offload_succ_rate
        self.bandwidth = bandwidth
        self.net_delay = net_delay
        self.link_fail_prob = link_fail_prob
        self.snr = self.calculate_snr(received_power, noise_power)
        self.plr = plr
        self.jitter = jitter
        self.channel_status = channel_status  # 1: Free, 0: Busy
        self.SPEED_OF_LIGHT_FIBER = 3e8
        self.distance = torch.rand(1, device='cpu') * 90 + 10

    def calculate_snr(self, received_power, noise_power):
        return received_power / noise_power

    def calculate_snr_db(self, received_power, noise_power):
        return 10 * np.log10(received_power / noise_power)

    def can_handle_task(self, task_cpu, task_bandwidth, task_ram):
        return (self.cpu_avail >= task_cpu and
                self.bandwidth >= task_bandwidth and
                self.ram_avail >= task_ram)

    def assign_task(self, task_cpu, task_bandwidth, task_ram):
        if self.can_handle_task(task_cpu, task_bandwidth, task_ram):
            self.cpu_avail -= task_cpu
            self.bandwidth -= task_bandwidth
            self.ram_avail -= task_ram
            return True
        return False
    
    def reset(self):
        self.cpu_avail = 3.0  # Reset to initial capacity
        self.bandwidth = 100   # Reset to initial bandwidth
        self.ram_avail = 2.0   # Reset to initial RAM

    def calculate_pd(self, distance):
        return distance / self.SPEED_OF_LIGHT_FIBER

    def calculate_transmission_latency(self, task_size, bandwidth, distance):
        pd = self.calculate_pd(distance)
        return task_size / bandwidth + pd


class FogToFogEnv:
    def __init__(self, num_fog_nodes=6, arrival_rate=3 * 10**3, task_execution_rate=2.0):
        self.num_fog_nodes = num_fog_nodes
        self.arrival_rate = arrival_rate  # Task arrival rate (lambda)
        self.task_execution_rate = task_execution_rate  # Task execution rate (tasks per unit time)
        self.fog_nodes = [FogNode(0, 0, 0, 0, 3.0, 2.0, 4, 0.8, 100, 30, 0.02, 1.0, 0.01, 1.0, 1) for _ in range(num_fog_nodes)]
        self.task_counter = 0
        self.result_ratio = 0.1
        self.state = self._generate_initial_state()
        self.max_latency = 100  # Placeholder max latency
        self.max_energy = 10  # Placeholder max energy
        self.max_successful_tasks = 10  # Placeholder max successful tasks
        self.smoothed_reward = 0
        self.previous_energy = 0  # Initialize previous energy for smoothing

    def _generate_initial_state(self):
        self.S_channel = np.random.choice([0, 1], size=(self.num_fog_nodes,))
        self.S_power = np.random.normal(3.0, 0.5, size=(self.num_fog_nodes,))
        self.S_gain = np.random.normal(1.0, 0.2, size=(self.num_fog_nodes,))
        self.S_size = np.random.normal(25, 10, size=(self.num_fog_nodes,))
        self.S_cycle = np.random.normal(3.0, 0.5, size=(self.num_fog_nodes,))
        self.S_ddl = np.random.normal(25, 10, size=(self.num_fog_nodes,))
        self.S_res = np.random.normal(2.0, 0.5, size=(self.num_fog_nodes,))
        self.S_com = np.random.normal(100, 20, size=(self.num_fog_nodes,))
        self.S_epsilon = np.random.normal(0.01, 0.005, size=(self.num_fog_nodes,))
        self.S_new_feature1 = np.random.normal(1.0, 0.2, size=(self.num_fog_nodes,))
        self.S_new_feature2 = np.random.normal(1.0, 0.2, size=(self.num_fog_nodes,))
        self.S_new_feature3 = np.random.normal(1.0, 0.2, size=(self.num_fog_nodes,))
        
        return np.array([
        [self.S_channel[n], self.S_power[n], self.S_gain[n], self.S_size[n], self.S_cycle[n],
         self.S_ddl[n], self.S_res[n], self.S_com[n], self.S_epsilon[n],
         self.S_new_feature1[n], self.S_new_feature2[n], self.S_new_feature3[n]]
        for n in range(self.num_fog_nodes)
        ])

    def calculate_reward(self, total_latency, total_energy, successful_tasks):
        latency_penalty_factor = 0.6
        energy_penalty_factor = 0.18
        success_reward_factor = 0.3
        smoothed_energy = self.smooth_energy(total_energy)
        latency_penalty = np.log1p(total_latency) * latency_penalty_factor if total_latency > self.max_latency else 1
        energy_penalty = np.log1p(smoothed_energy) * energy_penalty_factor
        task_success_bonus = success_reward_factor * (successful_tasks / self.max_successful_tasks)

        reward = 0.4 * task_success_bonus + 0.2 * latency_penalty + 0.1 * energy_penalty
        self.smoothed_reward = 0.90 * self.smoothed_reward + 0.1 * reward
        
        return np.clip(self.smoothed_reward, -1, 1)

    def smooth_energy(self, current_energy, smoothing_factor=0.1):
        # Apply exponential smoothing to the energy values to reduce fluctuations
        smoothed_energy = smoothing_factor * current_energy + (1 - smoothing_factor) * self.previous_energy
        self.previous_energy = smoothed_energy  # Update previous energy for the next step
        return smoothed_energy

    def reset(self):
        self.task_counter = 0
        for node in self.fog_nodes:
            node.reset()
        self.previous_energy = 0  # Reset previous energy for smoothing
        return self._generate_initial_state()

    def step(self, actions):
        task_cpu, task_bandwidth, task_ram, task_size = np.random.uniform(0.5, 2.0, 4)
        self.task_counter += 1

        # Simulate task arrivals based on lambda (Poisson distribution)
        num_tasks_arrived = np.random.poisson(self.arrival_rate)

        # If no tasks arrived, skip the step
        if num_tasks_arrived == 0:
            return self.state, 0, False, {}

        actions = np.clip(np.array(actions).flatten(), 0, 1)
        action_sum = np.sum(actions)
        offloading_probabilities = actions / action_sum if action_sum > 0 else np.ones_like(actions) / len(actions)

        total_latency, total_energy, successful_tasks = 0, 0, 0

        # Calculate task execution rate (tasks processed per time unit based on available resources)
        task_execution_rate = min(self.task_execution_rate, self.fog_nodes[0].cpu_avail)

        # Task execution and offloading loop
        local_cpu = task_cpu * max(0, 1 - np.sum(offloading_probabilities))  # Ensure non-negative
        local_bandwidth = task_bandwidth * max(0, 1 - np.sum(offloading_probabilities))
        local_ram = task_ram * max(0, 1 - np.sum(offloading_probabilities))

        if self.fog_nodes[0].can_handle_task(local_cpu, local_bandwidth, local_ram):
            self.fog_nodes[0].assign_task(local_cpu, local_bandwidth, local_ram)
            
            # Recalculate available CPU after assignment
            available_cpu = self.fog_nodes[0].cpu_avail + local_cpu  # Adjusted available CPU

            total_latency += local_cpu / max(available_cpu, 1e-6)  # Prevent division by zero
            total_energy += available_cpu * local_cpu * 1e9  # Corrected energy calculation
            successful_tasks += 1

            for i in range(self.num_fog_nodes):
                if offloading_probabilities[i] > 0 and self.fog_nodes[i].channel_status == 1:
                    execution_latency = task_cpu / max(self.fog_nodes[i].cpu_avail, 1e-6)
                    transmission_latency = self.fog_nodes[i].calculate_transmission_latency(task_size, self.fog_nodes[i].bandwidth, self.fog_nodes[i].distance.item())
                    result_transmission_latency = self.fog_nodes[i].calculate_transmission_latency(task_size * self.result_ratio, self.fog_nodes[i].bandwidth, self.fog_nodes[i].distance.item())

                    total_task_latency = transmission_latency + execution_latency + result_transmission_latency
                    total_latency = max(total_latency, total_task_latency)  # Take the max latency

                    transmission_energy = transmission_latency * self.fog_nodes[i].bandwidth * 1e-9
                    execution_energy = execution_latency * self.fog_nodes[i].cpu_avail * 1e-9
                    total_energy += transmission_energy + execution_energy
                    
                    successful_tasks += 1

        reward = self.calculate_reward(total_latency, total_energy, successful_tasks)
        done = self.task_counter >= 100
        next_state = self._generate_initial_state()
        info = {"Reward": reward, "Total Latency": total_latency, "Total Energy": total_energy, "Successful Tasks": successful_tasks}
        return next_state, reward, done, info
