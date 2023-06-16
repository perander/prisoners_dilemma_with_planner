import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from memory.planner_memory import PlannerMemory
from gymnasium.spaces.multi_discrete import MultiDiscrete
from utils.planner_utils import get_actions, unmap


class PlannerActor(nn.Module):
    def __init__(self, input_dims, n_agents, n_planner_actions, lr):
        super(PlannerActor, self).__init__()

        self.input_dims = input_dims  # 2 because planner gets both agents' actions as input
        self.output_dims = n_planner_actions # 5 or whatever hyperparameters.json has

        self.actor = nn.Sequential(
            nn.Linear(self.input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.head_0 = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, self.output_dims[0]),
            nn.Softmax(dim=-1)
        )

        self.head_1 = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, self.output_dims[1]),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
       
    def forward(self, x):  # x is both agents' actions
        x = self.actor(x / 2) # TODO replace hard coded normalization with max agent action (currently 2)
        
        # print(list(Categorical(x).probs))
        # split_logits = torch.split(x, self.output_dims.tolist(), dim=1)
        split_logits = [self.head_0(x), self.head_1(x)]
        dist = [Categorical(logit) for logit in split_logits]

        return dist

class PlannerCritic(nn.Module):
    def __init__(self, input_dims, n_agents, n_planner_actions, lr):
        super(PlannerCritic, self).__init__()

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
        return self.critic(x.to(torch.float))  # TODO normalize (now fine since there are just 2 actions ([0,1]))
        

class Planner:
    def __init__(
        self,
        n_actions_per_agent,
        n_agents,
        agent_names,
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
        max_reward=1,
        training_frequency=256,
        t_learning_starts=0,
        anneal_lr=False,
        continuous=False
    ):
    
        self.gamma = gamma
        self.lr = lr
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.training_frequency = training_frequency
        self.t_learning_starts = t_learning_starts

        self.n_agents = n_agents
        self.agent_names = agent_names
        self.n_actions_per_agent = n_actions_per_agent
        
        action_space = MultiDiscrete(np.array([n_actions_per_agent, n_actions_per_agent]))
        self.actions = action_space.nvec
        print("planner actions in init", self.actions)
        
        self.max_reward = max_reward
        self.actions_mapped, self.actions_unmapped = get_actions(self.max_reward, self.n_actions_per_agent)

        self.actor = PlannerActor(input_dims, n_agents, self.actions, lr)
        self.critic = PlannerCritic(input_dims, n_agents, self.actions, lr)
        
        self.batch_size = batch_size
        self.memory_size = memory_size
        print("input dims", input_dims)
        self.memory = PlannerMemory(input_dims, self.batch_size, self.memory_size, self.n_agents)

        self.device = device
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actions_mapped = self.actions_mapped.to(self.device)
        self.anneal_lr = anneal_lr

        self.continuous = continuous


    def remember(self, step, i, obs, next_obs, action, reward, prob, val):
        if step > 0:
            obs = torch.tensor([obs[self.agent_names[0]], obs[self.agent_names[1]]])
            next_obs = torch.tensor([next_obs[self.agent_names[0]], next_obs[self.agent_names[1]]])
            action = torch.tensor([unmap(action, self.actions_mapped)])
            reward = torch.tensor([reward])
            prob = torch.tensor([prob])
            val = torch.tensor([val])

            self.memory.store_memory(i, obs, next_obs, action, reward, prob, val)


    def choose_action(self, planner_obs):
        obs = torch.tensor([planner_obs[self.agent_names[0]], planner_obs[self.agent_names[1]]])
        obs = obs.unsqueeze(0)

        obs = obs.to(self.device)

        dists = self.actor(obs)
        value = self.critic(obs)

        action = torch.stack([dist.sample() for dist in dists])
        probs = torch.stack([dist.log_prob(a) for a, dist in zip(action, dists)])
        prob = probs.sum(0)
        entropies = torch.stack([dist.entropy() for dist in dists])
        entropy = entropies.sum(0)
        
        mapped_action = self.actions_mapped[action].T[0]
        # print([list(dist.probs) for dist in dists])

        return mapped_action, prob, entropy, value, [dist.probs for dist in dists]

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

        new_probs = torch.stack([dist.log_prob(actions[:,i]) for i, dist in enumerate(dist)])

        new_probs = new_probs.sum(0)

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

    def get_annealed_lr(self, lr, step, n_steps):
        frac = 1.0 - (step - 1.0) / n_steps
        new_lr = frac * lr

        self.actor.optimizer.param_groups[0]["lr"] = new_lr
        self.critic.optimizer.param_groups[0]["lr"] = new_lr

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
            next_values = self.critic(next_states)
            
            advantages_arr = self.compute_gae(
                reward_arr, vals_arr, next_values, self.gamma, self.gae_lambda
            )

            if self.anneal_lr:
                self.get_annealed_lr(self.lr, step, n_steps)

            # For each batch, update actor and critic
            for batch in batches:
                states = torch.Tensor(state_arr[batch]).to(self.device)
                old_probs = torch.Tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.Tensor(action_arr[batch]).to(self.device)
                values = torch.Tensor(vals_arr[batch]).to(self.device)
                advantages = torch.Tensor(advantages_arr[batch]).to(self.device)
                # print(values)

                advantages = self.normalize_advantages(advantages)

                # compute losses
                policy_loss = self.get_policy_loss(
                    states, actions, old_probs, advantages, self.policy_clip
                )

                value_loss = self.get_value_loss(states, advantages, values)

                dist = self.actor(states)
                entropies = [d.entropy() for d in dist]
                entropy = sum(entropies)
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


