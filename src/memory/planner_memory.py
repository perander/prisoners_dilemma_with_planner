import numpy as np
import torch


class PlannerMemory(object):
    def __init__(self, obs_dim, batch_size, capacity, n_agents):
        self.obs = torch.zeros((capacity, obs_dim))
        self.next_obs = torch.zeros((capacity, obs_dim))
        self.actions = torch.zeros((capacity, n_agents))
        self.rewards = torch.zeros(capacity)
        self.values = torch.zeros(capacity)
        self.advantages = torch.zeros(capacity)
        self.probs = torch.zeros(capacity)

        self.capacity = capacity
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.obs_dim = obs_dim

    def generate_batches(self, device):
        n_obs = len(self.obs)
        batch_starts = np.arange(0, n_obs, self.batch_size)
        indices = np.arange(n_obs, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_starts]

        return (
            self.obs,
            self.next_obs,
            self.actions,
            self.probs,
            self.values,
            self.rewards,
            batches,
        )

    def store_memory(self, i, obs, next_obs, action, reward, prob, val):
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.probs[i] = prob
        self.values[i] = val
        self.rewards[i] = reward

    def clear_memory(self):
        self.obs = []
        self.next_obs = []
        self.probs = []
        self.actions = []
        self.rewards = []
        # self.dones = [] # TODO add
        self.vals = []

        self.obs = torch.zeros((self.capacity, self.obs_dim))
        self.next_obs = torch.zeros((self.capacity, self.obs_dim))
        self.actions = torch.zeros((self.capacity, self.n_agents))
        self.rewards = torch.zeros(self.capacity)
        self.values = torch.zeros(self.capacity)
        self.advantages = torch.zeros(self.capacity)
        self.probs = torch.zeros(self.capacity)

    def __len__(self):
        return len(self.obs)
