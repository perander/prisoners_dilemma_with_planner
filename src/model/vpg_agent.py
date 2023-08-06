import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from memory.vpg_memory import VPGMemory


class Actor(torch.nn.Module):
    # logistic regression with just one parameter
    def __init__(self, lr):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(1))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        
    def forward(self, x):
        # replace 0s (because sigmoid(self.theta * 0) = 0 always)
        x = x.detach().clone()
        x[x == 0] = -1

        if torch.isnan(self.theta):
            self.theta = nn.Parameter(torch.randn(1))
        
        p_cooperate = torch.sigmoid(self.theta * x)

        if x.shape[0] == 1:
            # dist = torch.tensor([p_cooperate, 1-p_cooperate], requires_grad=True).view(-1,1)
            dist = torch.cat((p_cooperate, 1-p_cooperate), 0)
            # print("dist in forward", dist)
        else:
            dist = torch.cat((p_cooperate, 1-p_cooperate), 1)
        
        dist = Categorical(dist)

        return dist, self.theta
    
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
        n_epochs=10,
        training_frequency=256,
        t_learning_starts=0,
        anneal_lr=False,
        clip = 5
    ):
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.device = device
        
        self.gamma = gamma
        self.lr = lr
        self.n_epochs = n_epochs
        self.memory_size = training_frequency
        self.training_frequency = training_frequency
        self.t_learning_starts = t_learning_starts
        
        self.memory = VPGMemory(self.memory_size)
        
        self.actor = Actor(self.lr)
        self.critic = Critic(n_actions=self.n_actions, input_dims=self.input_dims, lr=self.lr)

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.clip = clip


    def reset(self):     
        self.actor = Actor(self.lr)
        self.critic = Critic(n_actions=self.n_actions, input_dims=self.input_dims, lr=self.lr)

        self.actor.to(self.device)
        self.critic.to(self.device)


    def remember(self, i, obs, action, reward):
        obs = torch.tensor([obs]).to(self.device)
        action = torch.tensor([action])
        reward = torch.tensor([reward])

        self.memory.store_memory(i, obs, action, reward)


    def choose_action(self, obs):
        obs = torch.tensor(obs)
        obs = obs.unsqueeze(0)
        obs = obs.to(self.device)

        dist, theta = self.actor(obs)
        value = self.critic(obs)

        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        
        return action, probs, None, value, dist.probs, theta

    def compute_advantage(self, rewards):
        advantages = []

        for i in range(len(rewards)):
            new_g = 0
            power = 0
            for j in range(i, len(rewards)):
                new_g = new_g + ((self.gamma**power)*rewards[j])
                power += 1
            
            advantages.append(new_g)
        
        advantages = torch.tensor(advantages).to(self.device)
        # print("normalized", advantages / torch.max(torch.abs(advantages)))
        advantages = advantages / torch.max(torch.abs(advantages))
        return advantages

    def get_policy_loss(self, states, actions, rewards):
        advantages = self.compute_advantage(rewards)

        dists, _ = self.actor(states)

        if actions.shape[0] > 1:
            probs_per_action = dists.probs.gather(dim=1, index=actions.long().view(-1,1)).squeeze()
        else:
            probs_per_action = dists.probs.gather(dim=0, index=actions.unsqueeze(0).long()).squeeze()

        return -torch.sum(torch.log(probs_per_action) * advantages)


    def learn(self, step, n_steps):
        for epoch in range(self.n_epochs):
            (
                state_arr,
                action_arr,
                reward_arr,
            ) = self.memory.generate_batches()

            states = torch.Tensor(state_arr).unsqueeze(1).to(self.device)
            actions = torch.Tensor(action_arr).to(self.device)

            loss = self.get_policy_loss(states, actions, reward_arr)

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip)

            self.actor.optimizer.step()
            self.critic.optimizer.step()

        self.memory.clear_memory()
        return loss.item()





