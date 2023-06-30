import numpy as np
import torch
import logging
from model.ppo_planner import PlannerActor, PlannerCritic
from memory.planner_memory import PlannerMemory
from gymnasium.spaces.multi_discrete import MultiDiscrete
from utils.planner_utils import get_actions, unmap

class Planner:
    def __init__(
        self,
        n_actions_per_agent,
        n_agents,
        agent_names,
        max_reward,
        # n_actions,
        input_dims,
        device,
        gamma=0.99,
        lr=0.0003,
        batch_size=64,
        n_epochs=10,
        memory_size=1000000,
        training_frequency=256,
        t_learning_starts=0,
        anneal_lr=False,
    ):
        self.input_dims = input_dims
        self.device = device
        
        self.gamma = gamma
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.memory_size = training_frequency
        self.training_frequency = training_frequency
        self.t_learning_starts = t_learning_starts

        # self.n_actions = n_actions
        self.n_agents = n_agents
        self.agent_names = agent_names
        self.n_actions_per_agent = n_actions_per_agent

        action_space = MultiDiscrete(np.array([n_actions_per_agent, n_actions_per_agent]))
        self.actions = action_space.nvec
        logging.debug(f"planner actions in init: {self.actions}")

        self.max_reward = max_reward
        self.actions_mapped, self.actions_unmapped = get_actions(self.max_reward, self.n_actions_per_agent)

        self.actor = PlannerActor(input_dims, n_agents, self.actions, lr)
        self.critic = PlannerCritic(input_dims, n_agents, self.actions, lr)

        self.memory = PlannerMemory(input_dims, self.batch_size, self.memory_size, self.n_agents)

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actions_mapped.to(self.device)

    
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
    
    def compute_advantage(self, rewards):
        # print("rewards", rewards, len(rewards))
        advantages = []

        for i in range(len(rewards)):
            new_g = 0
            power = 0
            for j in range(i, len(rewards)):
                new_g = new_g + ((self.gamma**power)*rewards[j])
                power += 1
            
            advantages.append(new_g)
        
        advantages = torch.tensor(advantages).to(self.device)
        # print(advantages)
        # print("normalized", advantages / torch.max(torch.abs(advantages)))
        advantages = advantages / torch.max(torch.abs(advantages))
        return advantages


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

            states = torch.Tensor(state_arr).to(self.device)
            # print("states", states)
            next_states = torch.Tensor(next_state_arr).to(self.device)
            next_values = self.critic(next_states)
            actions = torch.Tensor(action_arr).to(self.device)

            advantages_arr = self.compute_advantage(reward_arr)
            # print(advantages_arr)

            # print("actions", actions)
            dists = self.actor(states)
            # print("dists", [d.probs for d in dists])

            dist_probs = [d.probs for d in dists]

            probs_per_action = [dist_probs[i].gather(dim=1, index=actions.long()[:,i].view(-1,1)).squeeze() for i in range(self.n_agents)]

            probs_per_action_pair = torch.mul(*probs_per_action)

            loss = -torch.sum(torch.log(probs_per_action_pair) * advantages_arr)

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()

            loss.backward()

            self.actor.optimizer.step()
            self.critic.optimizer.step()

        self.memory.clear_memory()
        return loss.item()