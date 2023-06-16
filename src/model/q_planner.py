import numpy as np
import torch
from gymnasium.spaces.multi_discrete import MultiDiscrete
from utils.planner_utils import get_actions

class QLearningPlanner:
    def __init__(self, states, actions, lr, gamma, epsilon, min_epsilon, n_actions_per_agent, max_reward, n_states, agent_names, t_learning_starts, training_frequency):
        self.states = states
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.initial_epsilon = epsilon
        self.agent_names = agent_names
        self.t_learning_starts = t_learning_starts
        self.training_frequency = training_frequency

        self.max_reward = max_reward
        self.n_states = n_states
        self.state_bins = [(0,1), (0,1)]

        self.n_actions_per_agent = n_actions_per_agent
        self.action_space = MultiDiscrete(np.array([self.n_actions_per_agent, self.n_actions_per_agent]))
        self.actions = self.action_space.nvec
        self.action_bins, _ = get_actions(self.max_reward, self.n_actions_per_agent)
        self.n_action_pairs = len(self.action_bins) ** 2
        
        self.q_table = torch.zeros((len(self.state_bins) ** 2, len(self.action_bins) ** 2))
        print("q table of size", self.q_table.shape)

    def discretize_state(self, state):
        return state[0]*2 + state[1]
    

    def discretize_action(self, action):
        return ((self.action_bins == action[0]).nonzero(as_tuple=True)[0]) * len(self.action_bins) + ((self.action_bins == action[1]).nonzero(as_tuple=True)[0])

    def update_epsilon(self, step):
        return max(self.min_epsilon, self.initial_epsilon / (1.0 + 0.00001 * step))

    def choose_action(self, state):
        state = [state[self.agent_names[0]], state[self.agent_names[1]]]
        state_index = self.discretize_state(state)

        if np.random.uniform(0, 1) < self.epsilon:
            action_index = int(np.random.choice(len(self.action_bins) ** 2))  # explore
        else:
            action_index = torch.argmax(self.q_table[state_index, :])  # exploit

        action_1 = self.action_bins[action_index // len(self.action_bins)]
        action_2 = self.action_bins[action_index % len(self.action_bins)]

        return action_1, action_2

    def learn(self, state, action, reward, next_state, step):
        self.epsilon = self.update_epsilon(step)

        state = [state[self.agent_names[0]], state[self.agent_names[1]]]
        next_state = [next_state[self.agent_names[0]], next_state[self.agent_names[1]]]
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)
        action_index = self.discretize_action(action).item()

        current_q = self.q_table[state, action_index]
        next_q = torch.max(self.q_table[next_state, :])
        target_q = reward + self.gamma * next_q

        self.q_table[state, action_index] += self.lr * (target_q - current_q)

        return self.q_table


