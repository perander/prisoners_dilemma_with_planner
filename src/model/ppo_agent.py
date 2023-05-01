import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from memory.ppo_memory import PPOMemory

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, lr):
        super().__init__()
        self.output_dims= n_actions
        self.input_dims = input_dims
        self.actor = nn.Sequential(
            nn.Linear(self.input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dims),
            nn.Softmax(dim=-1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        dist = self.actor(x / self.output_dims)
        dist = Categorical(dist)

        return dist


class CriticNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, lr):
        super(CriticNetwork, self).__init__()

        self.output_dims = n_actions
        self.input_dims = input_dims

        self.critic = nn.Sequential(
            nn.Linear(self.input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.critic(x / self.output_dims)


class Agent:
    def __init__(
        self,
        n_actions,
        input_dims,
        device,
        gamma=0.99,
        lr=0.0003,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        memory_size=1000000,
        ent_coef=0.01,
        vf_coef=0.5,
        training_frequency=256,
        t_learning_starts=0
    ):
        self.gamma = gamma
        self.lr = lr
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_actions = n_actions

        self.training_frequency = training_frequency
        self.t_learning_starts = t_learning_starts

        self.actor = ActorNetwork(n_actions, input_dims, lr)
        self.critic = CriticNetwork(n_actions, input_dims, lr)
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = PPOMemory(input_dims, self.batch_size, self.memory_size)

        self.device = device
        self.actor.to(self.device)
        self.critic.to(self.device)

    def remember(self, i, obs, next_obs, action, reward, prob, val):
        obs = torch.tensor([obs])
        next_obs = torch.tensor([next_obs])
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        prob = torch.tensor([prob])
        val = torch.tensor([val])

        self.memory.store_memory(i, obs, next_obs, action, reward, prob, val)

    def choose_action(self, obs):
        obs = torch.tensor(obs)
        obs = obs.unsqueeze(0)

        dist = self.actor(obs)
        value = self.critic(obs)

        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, dist.entropy(), value

    def compute_gae(self, rewards, values, next_values, gamma, gae_lambda):
        num_steps = len(rewards)
        advantages = np.zeros(num_steps)
        last_gae = 0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                delta = rewards[t] + gamma * next_values[t] - values[t]
            else:
                delta = rewards[t] + gamma * values[t + 1] - values[t]

            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae

        return advantages

    def normalize_advantages(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def get_policy_loss(self, states, actions, old_probs, advantages, policy_clip):
        dist = self.actor(states)
        new_probs = dist.log_prob(actions)
        prob_ratio = (new_probs - old_probs).exp()

        weighted_probs = -advantages * prob_ratio
        weighted_clipped_probs = -advantages * torch.clamp(
            prob_ratio, 1 - policy_clip, 1 + policy_clip
        )
        policy_loss = torch.max(weighted_probs, weighted_clipped_probs).mean()

        return policy_loss

    def get_value_loss(self, states, advantages, values):
        returns = advantages + values
        critic_value = self.critic(states)
        critic_value = torch.squeeze(critic_value)

        critic_loss = ((returns - critic_value) ** 2).mean()

        return critic_loss

    def learn(self, step, n_steps):
        for epoch in range(self.n_epochs):
            (
                state_arr,
                next_state_arr,
                action_arr,
                old_prob_arr,
                vals_arr,
                reward_arr,
                # dones_arr, TODO add
                batches,
            ) = self.memory.generate_batches(self.device)

            next_states = torch.Tensor(next_state_arr).to(self.device)
     
            next_values = self.critic(next_states.unsqueeze(1))
            advantages_arr = self.compute_gae(
                reward_arr, vals_arr, next_values, self.gamma, self.gae_lambda
            )

            # For each batch, update actor and critic
            for batch in batches:
                states = torch.Tensor(state_arr[batch]).to(self.device)
                states = states.unsqueeze(1)
                old_probs = torch.Tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.Tensor(action_arr[batch]).to(self.device)
                values = torch.Tensor(vals_arr[batch]).to(self.device)
                advantages = torch.Tensor(advantages_arr[batch]).to(self.device)

                advantages = self.normalize_advantages(advantages)

                # compute losses
                policy_loss = self.get_policy_loss(
                    states, actions, old_probs, advantages, self.policy_clip
                )

                value_loss = self.get_value_loss(states, advantages, values)

                dist = self.actor(states)
                entropy = dist.entropy()
                entropy_loss = entropy.mean()

                total_loss = (
                    policy_loss
                    - self.ent_coef * entropy_loss
                    + value_loss * self.vf_coef
                )

                # step
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()

                for param in self.actor.parameters():
                    param.grad.data.clamp_(-1, 1)
                for param in self.critic.parameters():
                    param.grad.data.clamp_(-1, 1)

                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
        return total_loss.item()
