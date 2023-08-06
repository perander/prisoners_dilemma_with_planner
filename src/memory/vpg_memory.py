import numpy as np

class VPGMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity

        self.obs = np.zeros(capacity)
        self.actions = np.zeros(capacity)
        self.rewards = np.zeros(capacity)


    def generate_batches(self):
        return (
            self.obs,
            self.actions,
            self.rewards,
        )

    def store_memory(self, i, obs, action, reward):
        """_summary_

        Args:
            i (_type_): memory index
            obs (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
        """

        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward

    def clear_memory(self):
        self.obs = []
        self.actions = []
        self.rewards = []

        self.obs = np.zeros(self.capacity)
        self.actions = np.zeros(self.capacity)
        self.rewards = np.zeros(self.capacity)

    def __len__(self):
        return len(self.obs)
