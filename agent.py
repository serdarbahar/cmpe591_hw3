import torch
from torch import optim

from model import VPG, ValueNet
import torch.nn.functional as F

from collections import deque


class VPG_Agent():
    def __init__(self,gamma,LR):
        self.model = VPG()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.rewards = []
        self.log_probs = []
        self.gamma = gamma
        self.baseline_buffer = deque(maxlen=100) # store the last 100 episodes' mean of discounted returns (reward-to-go)
        
    def decide_action(self, state, variance):
        state = torch.tensor(state, dtype=torch.float32)
        action_mean, act_std = self.model(state).chunk(2, dim=-1)
        action_std = F.softplus(act_std) + variance # increase variance to stimulate exploration
        
        action = torch.distributions.Normal(action_mean, action_std).sample() # sample action, action is clipped in the environment
        log_prob = torch.distributions.Normal(action_mean, action_std).log_prob(action) # log-prob of the action
        log_prob = log_prob.sum(dim=-1)
        self.log_probs.append(log_prob) 
        return action.detach()

    def update_model(self):
        # compute discounted rewards
        discounted_returns = []
        G = 0
        # compute the discounted returns (reward-to-go: causality trick)
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            discounted_returns.insert(0, G)
        discounted_returns = torch.tensor(discounted_returns)
        self.baseline_buffer.append(discounted_returns.mean().item()) # store the mean of the discounted returns
        baseline = torch.mean(torch.tensor(self.baseline_buffer)) # compute baseline
        advantages = discounted_returns - baseline # compute advantage

        # compute policy loss
        log_probs = torch.stack(self.log_probs)
        loss = -torch.mean(log_probs * advantages) # negative for gradient ascent

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # clear the log_probs and rewards for the next episode
        self.log_probs = []
        self.rewards = []
        
    def add_reward(self, reward):
        self.rewards.append(reward)


class AC_Agent():
    def __init__(self, gamma, LR, ValueLR, lam, obs_dim, act_dim):
        self.model = VPG(obs_dim=obs_dim, act_dim=act_dim)
        self.policy_optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.value = ValueNet(obs_dim=obs_dim)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=ValueLR)
        
        self.gamma = gamma
        self.lam = lam
        self.rewards = []
        self.log_probs = []
        self.states = []

    def decide_action(self, state, variance):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        self.states.append(state_tensor)

        out = self.model(state_tensor)
        action_mean, act_std = out.chunk(2, dim=-1)
        action_std = F.softplus(act_std) + variance

        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        self.log_probs.append(log_prob)
        return action.detach()
    
    def add_reward(self, reward):
        self.rewards.append(reward)
    
    def add_state(self, state):
        self.states.append(state)
    
    def compute_gae(self, rewards, values):
        advantages = []
        gae = 0
        values = values + [0]  # V(s_{T+1}) = 0 for terminal
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        returns = [adv + v for adv, v in zip(advantages, values[:-1])]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def update_model(self):
        states = torch.stack(self.states)

        rewards = self.rewards
        with torch.no_grad():
            values = self.value(states).squeeze(-1).tolist()  # Get values for all statese
        
        advantages, returns = self.compute_gae(rewards, values) # compute generalized advantage estimates
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize advantages

        # Policy loss
        log_probs = torch.stack(self.log_probs)
        policy_loss = -torch.mean(log_probs * advantages)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Value loss
        predicted_values = self.value(states).squeeze(-1)
        value_loss = F.mse_loss(predicted_values, returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Clear buffers
        self.log_probs = []
        self.rewards = []
        self.states = []