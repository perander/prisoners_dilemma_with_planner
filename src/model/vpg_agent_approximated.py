import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from scipy.optimize import minimize

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


class VPG_Approximated:
    def __init__(
        self,
        lr,
        history_length
    ):
        self.lr = lr
        self.actor = Actor(lr)
        self.history_length = history_length

        self.history = torch.full((self.history_length,), -1)


    def set_theta(self, new_theta):
        self.actor.theta = nn.Parameter(data=new_theta.clone().detach(), requires_grad=True)

    
    def update_theta(self, agent_actions):
        self.update_history(agent_actions)
        p_cooperate_historical = (self.history == -1).sum() / self.history.shape[0]
        
        print(f"p(c) from history vs model: {p_cooperate_historical.item():.2f}, {torch.sigmoid(self.actor.theta * torch.tensor([-1])).data.item():.2f}")

        w_init = torch.tensor([1])

        def loss(w, p_cooperate_historical):
            w = torch.tensor(w)
            return torch.abs(p_cooperate_historical - torch.mean(torch.sigmoid(w * torch.tensor([-1]))))
        
        result = minimize(loss, w_init, args=(p_cooperate_historical))

        new_theta = torch.tensor(result.x[0])
        self.set_theta(new_theta)


    def update_history(self, agent_actions):
        if agent_actions == 0:
            agent_actions = -1
        self.history[:-1] = self.history[1:].detach().clone()
        self.history[-1] = agent_actions
