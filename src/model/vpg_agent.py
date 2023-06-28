import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from memory.vpg_memory import VPGMemory

class Actor(nn.Module):
    def __init__(self, n_actions, input_dims, lr):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = n_actions

        self.actor = nn.Sequential(
            nn.Linear(self.input_dims, 8),
            nn.ReLU(),
            nn.Linear(8, self.output_dims),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x):
        dist = self.actor(x / self.output_dims)
        dist = Categorical(dist)

        return dist
    
class Critic(nn.Module):
    def __init__(self, n_actions, input_dims, lr):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = n_actions

        self.critic = nn.Sequential(
            nn.Linear(self.input_dims, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
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
        batch_size=64,
        n_epochs=10,
        memory_size=1000000,
        training_frequency=256,
        t_learning_starts=0,
        anneal_lr=False
    ):
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.device = device
        
        self.gamma = gamma
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.memory_size = training_frequency
        self.training_frequency = training_frequency
        self.t_learning_starts = t_learning_starts
        
        self.memory = VPGMemory(self.input_dims, self.batch_size, self.memory_size)
        
        self.actor = Actor(n_actions=self.n_actions, input_dims=self.input_dims, lr=self.lr)
        self.critic = Critic(n_actions=self.n_actions, input_dims=self.input_dims, lr=self.lr)


        self.actor.to(self.device)
        self.critic.to(self.device)


    def remember(self, i, obs, next_obs, action, reward, prob, val):
        obs = torch.tensor([obs])
        obs = obs.to(self.device)
        next_obs = torch.tensor([next_obs])
        next_obs = next_obs.to(self.device)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        prob = torch.tensor([prob])
        val = torch.tensor([val])

        self.memory.store_memory(i, obs, next_obs, action, reward, prob, val)


    def choose_action(self, obs):
        obs = torch.tensor(obs)
        obs = obs.unsqueeze(0)
        obs = obs.to(self.device)

        dist = self.actor(obs)
        value = self.critic(obs)

        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        
        return action, probs, None, value, dist.probs

    def compute_advantage(self, rewards):
        print("rewards", rewards, len(rewards))
        advantages = []

        for i in range(len(rewards)):
            new_g = 0
            power = 0
            for j in range(i, len(rewards)):
                new_g = new_g + ((self.gamma**power)*rewards[j])
                power += 1
            
            advantages.append(new_g)
        
        advantages = torch.tensor(advantages).to(self.device)
        print(advantages)
        # print("normalized", advantages / torch.max(torch.abs(advantages)))
        return advantages

        return sum(rewards) # placeholder

    def get_policy_loss(self):
        pass

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

            states = torch.Tensor(state_arr).unsqueeze(1).to(self.device)
            next_states = torch.Tensor(next_state_arr).unsqueeze(1).to(self.device)
            next_values = self.critic(next_states)
            actions = torch.Tensor(action_arr).to(self.device)

            advantages_arr = self.compute_advantage(reward_arr)
            print(advantages_arr)

            print("actions", actions)
            dists = self.actor(states)
            print("dists", dists.probs)

            probs_per_action = dists.probs.gather(dim=1, index=actions.long().view(-1,1)).squeeze()

            print(probs_per_action)
            print("advantages", advantages_arr)

            loss = -torch.sum(torch.log(probs_per_action) * advantages_arr)
            print("loss", loss)

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()

            loss.backward()

            self.actor.optimizer.step()
            self.critic.optimizer.step()

        self.memory.clear_memory()
        return loss.item()





