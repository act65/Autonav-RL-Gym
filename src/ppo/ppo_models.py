import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from torch.distributions import Categorical
import sys
import os
from ppo.storage import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normalise(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-5)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(state_dim, n_latent_var)

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, action_dim)
                )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def value(self, state):
        return self.value_layer(state)

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)[:, action]

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma,
                 K_epochs, eps_clip, save_path, load_path):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.save_path = save_path
        self.load_path = load_path

        if load_path is not None:
            self.load_models()

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory=None):
        def action_unnormalized(action, high, low):
            #
            action = np.clip(action, -1, 1)
            action = low + (action + 1.0) * 0.5 * (high - low)
            return action
        #state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.policy_old.act(state, memory)
        #print("action = " + str(action))
        #print("action = " + str(action[0]))
        #out = np.array([action_unnormalized(action[0], self.v_max, self.v_min),
        #                          action_unnormalized(action[1], self.v_max, self.v_min)])
        #print("out = " + str(out))
        #return action
        #print("action = " + str(action))
        return action

    def value(self, state):
        return self.policy.value(torch.from_numpy(state).float().to(device)).detach()

    def save_models(self, path):
        torch.save(self.policy.state_dict(), os.path.join(self.save_path, 'policy.pth'))
        torch.save(self.policy_old.state_dict(), os.path.join(self.save_path, 'policy_old.pth'))

    def load_models(self):
        self.policy.load_state_dict(torch.load(os.path.join(self.load_path, 'policy.pth'), map_location=torch.device(device)))
        self.policy_old.load_state_dict(torch.load(os.path.join(self.load_path, 'policy_old.pth'), map_location=torch.device(device)))

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        returns = []
        discounted_reward = 0
        for i in reversed(range(len(memory.rewards))):
            discounted_reward = memory.rewards[i] + (self.gamma * discounted_reward)*(1-memory.masks[i])
            returns.insert(0, discounted_reward)

        # Normalizing the rewards:
        returns = torch.tensor(returns).to(device)
        # returns = normalise(returns)  # (returns - returns.mean()) / (returns.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = normalise(state_values - self.policy.value(old_states).mean(1)).detach()
            # advantages = returns - state_values.detach()  # ????
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, returns) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
