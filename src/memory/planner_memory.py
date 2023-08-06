import numpy as np
import torch


class PlannerMemory(object):
    def __init__(self, obs_dim, capacity, n_agents):
        self.obs = torch.zeros((capacity, obs_dim))
        self.actions = torch.zeros((capacity, n_agents))
        self.rewards = torch.zeros((capacity, n_agents))
        self.agent_state = torch.zeros((capacity, obs_dim))

        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim

    def generate_batches(self):
        return (
            self.obs,
            self.actions,
            self.rewards,
            self.agent_state,
        )

    def store_memory(self, i, obs, action, reward, agent_state):
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.agent_state[i] = agent_state

    def clear_memory(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.agent_state = []

        self.obs = torch.zeros((self.capacity, self.obs_dim))
        self.actions = torch.zeros((self.capacity, self.n_agents))
        self.rewards = torch.zeros((self.capacity, self.n_agents))
        self.agent_state = torch.zeros((self.capacity, self.obs_dim))

    def __len__(self):
        return len(self.obs)
