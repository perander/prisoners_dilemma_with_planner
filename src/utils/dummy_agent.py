import numpy as np


class DummyAgent:
    def __init__(
            self,
            training_frequency,
            t_learning_starts
        ):
        self.training_frequency = training_frequency
        self.t_learning_starts = t_learning_starts
        self.action_probs = [0.50, 0.50]
        self.obs = [0, 0]
        self.increment = 0.05

    def remember(self, i, obs, next_obs, action, reward, prob, val):
        self.reward = reward
        print(self.reward)

    def choose_action(self, obs):
        action = np.random.choice([0, 1], p=self.action_probs)
        return action, np.log(self.action_probs[action]), None, None, self.action_probs

    def learn(self, step, n_steps):
        self.action_probs[0] = max(self.action_probs[0] + self.reward * self.increment, 0)
        self.action_probs[1] = min(self.action_probs[1] - self.reward * self.increment, 1)
        self.action_probs = np.exp(self.action_probs) / (np.exp(self.action_probs)).sum()
        print(self.action_probs)
        return 0