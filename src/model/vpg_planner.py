import torch
import torch.nn as nn
import torch.optim as optim
from model.ppo_planner import PlannerCritic
from memory.planner_memory import PlannerMemory

class VPGActor(nn.Module):
    def __init__(self, input_dims, n_agents, lr):
        super(VPGActor, self).__init__()

        self.input_dims = input_dims
        self.output_dims = n_agents

        self.actor = nn.Sequential(
            nn.Linear(self.input_dims, self.output_dims),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.actor(x.float())


class Planner:
    def __init__(
        self,
        n_agents,
        agent_names,
        max_reward,
        input_dims,
        device,
        gamma=0.99,
        lr=0.0003,
        n_epochs=10,
        training_frequency=256,
        t_learning_starts=0,
        anneal_lr=False,
        clip=5
    ):
        self.input_dims = input_dims
        self.device = device
        
        self.gamma = gamma
        self.lr = lr
        self.n_epochs = n_epochs
        self.memory_size = training_frequency
        self.training_frequency = training_frequency
        self.t_learning_starts = t_learning_starts

        self.n_agents = n_agents
        self.agent_names = agent_names

        self.max_reward = max_reward

        self.actor = VPGActor(input_dims, n_agents, lr)
        self.critic = PlannerCritic(input_dims, n_agents, lr)

        self.memory = PlannerMemory(input_dims, self.memory_size, self.n_agents)

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.clip = clip

    def remember(self, step, i, action, obs, next_obs, reward, agent_state, meta):
        obs = torch.tensor([obs[self.agent_names[0]], obs[self.agent_names[1]]])
        reward = torch.tensor([reward[self.agent_names[0]], reward[self.agent_names[1]]])
        agent_state = torch.tensor([agent_state[self.agent_names[0]], agent_state[self.agent_names[1]]])

        self.memory.store_memory(i, obs, action, reward, agent_state)

    
    def choose_action(self, planner_obs):
        obs = torch.tensor([planner_obs[self.agent_names[0]], planner_obs[self.agent_names[1]]])

        obs = obs.to(self.device)

        actions = self.actor(obs)
        value = self.critic(obs)

        return self.max_reward * actions.squeeze(0), value

    def compute_advantage(self, rewards):
        # not really advantage but just a sum of rewards, TODO rename
        advantages = []

        for i in range(len(rewards)):
            new_g = 0
            power = 0
            for j in range(i, len(rewards)):
                new_g = new_g + ((self.gamma**power)*rewards[j])
                power += 1
            
            advantages.append(new_g)
        
        advantages = torch.tensor(advantages).to(self.device)
        advantages = advantages / torch.max(torch.abs(advantages))
        return advantages

    def get_policy_loss(self, states, actions, rewards):
        rewards = [torch.add(*individual_rewards) for individual_rewards in rewards]

        advantages = self.compute_advantage(rewards)

        dists = self.actor(states)

        dist_probs = [d.probs for d in dists]

        probs_per_action = [dist_probs[i].gather(dim=1, index=actions.long()[:,i].view(-1,1)).squeeze() for i in range(self.n_agents)]

        pi_p = torch.mul(*probs_per_action)

        return -torch.sum(torch.log(pi_p) * advantages)

    
    def get_policy_loss_predictive(self, states, actions, rewards, agent_lrs, agents, agent_state):
        agent_0 = agents[0][1]
        agent_1 = agents[1][1]

        total_loss = 0

        for i, agent in enumerate([agent_0, agent_1]):
            # calculate agent g_log_pi
            dists, _ = agent.actor(agent_state[:,i].unsqueeze(1))
            agent_actions = states[:,i].unsqueeze(1)

            pi = dists.probs.gather(dim=1, index=agent_actions.long().view(-1,1)).squeeze()

            log_pi = torch.log(pi)

            agent.actor.optimizer.zero_grad()
            log_pi[0].backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), self.clip)
            g_log_pi = [param.grad.detach() for _, param in agent.actor.named_parameters()][0]

            # treat the rewards as they are (do not calculate advantages? do not even sum?)
            actions = self.actor(states)
            g_Vp = g_log_pi * actions[0][i]

            g_V = g_log_pi * torch.sum(rewards[0])
            
            # loss per agent
            loss = -agent_lrs[i] * g_Vp * g_V

            total_loss += loss
            # print(f"loss for agent {i}", loss, "total loss", total_loss)

        return total_loss


    def learn(self, step, n_steps, agent_1_lr, agent_2_lr, agents):
        for epoch in range(self.n_epochs):
            (
                state_arr,
                action_arr,
                reward_arr,
                agent_state,
            ) = self.memory.generate_batches()

            states = torch.Tensor(state_arr).to(self.device)
            agent_state = torch.Tensor(agent_state).to(self.device)
            actions = torch.Tensor(action_arr).to(self.device)

            loss = self.get_policy_loss_predictive(states, actions, reward_arr, [agent_1_lr, agent_2_lr], agents, agent_state)

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip)

            self.actor.optimizer.step()
            self.critic.optimizer.step()

        self.memory.clear_memory()
        return loss.item()