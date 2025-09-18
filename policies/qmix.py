import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

class DRQN(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super(DRQN, self).__init__()
        
        layers = []
        input_dim = obs_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        
        self.rnn = nn.GRUCell(hidden_sizes[-1], hidden_sizes[-1])
        self.q_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs, hidden_state):
        x = self.mlp(obs)
        h_in = hidden_state.reshape(-1, self.rnn.hidden_size)
        h_out = self.rnn(x, h_in)
        q_values = self.q_head(h_out)
        return q_values, h_out

class QMixer(nn.Module):
    def __init__(self, num_agents, state_dim, mixing_embed_dim):
        super(QMixer, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        self.hyper_w1 = nn.Linear(self.state_dim, self.embed_dim * self.num_agents)
        self.hyper_w2 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b2 = nn.Linear(self.state_dim, 1)

    def forward(self, agent_q_values, states):
        batch_size = agent_q_values.size(0)
        num_agents_in_batch = agent_q_values.shape[1]
        agent_q_values = agent_q_values.view(-1, 1, num_agents_in_batch)

        w1 = torch.abs(self.hyper_w1(states)).view(-1, num_agents_in_batch, self.embed_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_q_values, w1) + b1)
        
        w2 = torch.abs(self.hyper_w2(states)).view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)
        
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(batch_size, 1)

class EpisodeReplayBuffer:
    def __init__(self, capacity, max_episode_len):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.max_episode_len = max_episode_len

    def store_episode(self, episode_data):
        self.buffer.append(episode_data)

    def sample(self, batch_size):
        sampled_episodes = random.sample(self.buffer, batch_size)
        
        padded_batch = {}
        max_len = max(len(ep['obs']) for ep in sampled_episodes)

        for key in sampled_episodes[0].keys():
            if key == 'actions':
                dtype = torch.long
                pad_value = 0
            elif key == 'dones':
                dtype = torch.bool
                pad_value = True
            else:
                dtype = torch.float32
                pad_value = 0.0

            padded_list = []
            for ep in sampled_episodes:
                ep_len = len(ep[key])
                padding_needed = max_len - ep_len
                
                ep_array = np.array(ep[key])
                
                pad_shape = (padding_needed,) + ep_array.shape[1:]
                padding = np.full(pad_shape, pad_value)
                
                padded_ep = np.concatenate([ep_array, padding], axis=0)
                padded_list.append(padded_ep)
            
            padded_batch[key] = torch.tensor(np.array(padded_list), dtype=dtype)

        return padded_batch, max_len

    def __len__(self):
        return len(self.buffer)

class QMIX:
    def __init__(self, obs_dim, state_dim, action_dim, num_agents, hidden_sizes, mixer_hidden_size, lr, gamma, device):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.device = device

        self.agent_network = DRQN(obs_dim, action_dim, hidden_sizes).to(device)
        self.target_agent_network = DRQN(obs_dim, action_dim, hidden_sizes).to(device)
        self.mixer_network = QMixer(num_agents, state_dim, mixer_hidden_size).to(device)
        self.target_mixer_network = QMixer(num_agents, state_dim, mixer_hidden_size).to(device)

        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        self.target_mixer_network.load_state_dict(self.mixer_network.state_dict())
        
        self.params = list(self.agent_network.parameters()) + list(self.mixer_network.parameters())
        self.optimizer = optim.Adam(self.params, lr=lr)
        
        self.agent_hidden_state = None

    def choose_actions(self, observations, epsilon, deterministic=False):
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        q_values, self.agent_hidden_state = self.agent_network(obs_tensor, self.agent_hidden_state)
        
        greedy_actions = q_values.argmax(dim=1).cpu().numpy()
        if not deterministic:
            random_mask = np.random.rand(observations.shape[0]) < epsilon
            random_actions = np.random.randint(0, self.action_dim, size=observations.shape[0])
            greedy_actions[random_mask] = random_actions[random_mask]
        
        return greedy_actions

    def update(self, batch, max_len):
        obs_batch = batch['obs'].to(self.device)
        actions_batch = batch['actions'].to(self.device)
        rewards_batch = batch['rewards'].to(self.device)
        dones_batch = batch['dones'].to(self.device)
        mask_batch = (1 - dones_batch.float())

        batch_size = obs_batch.shape[0]
        
        q_evals = []
        q_targets = []
        
        hidden_state = torch.zeros(batch_size * self.num_agents, self.hidden_sizes[-1]).to(self.device)
        target_hidden = torch.zeros_like(hidden_state)

        # --- Vectorised RNN rollout ---
        for t in range(max_len):
            obs = obs_batch[:, t].reshape(-1, self.obs_dim)
            q_eval, hidden_state = self.agent_network(obs, hidden_state)
            q_evals.append(q_eval.view(batch_size, self.num_agents, self.action_dim))

            next_obs = obs_batch[:, t+1].reshape(-1, self.obs_dim) if t+1 < max_len else obs
            q_target, target_hidden = self.target_agent_network(next_obs, target_hidden)
            q_targets.append(q_target.view(batch_size, self.num_agents, self.action_dim))

        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        
        chosen_action_qvals = torch.gather(q_evals, dim=3, index=actions_batch.unsqueeze(3)).squeeze(3)
        max_q_targets = q_targets.max(dim=3)[0]
        
        states = obs_batch.view(batch_size, max_len, -1)
        
        q_total_eval = self.mixer_network(chosen_action_qvals.view(-1, self.num_agents), states.view(-1, self.state_dim)).view(batch_size, max_len, 1)
        q_total_target = self.target_mixer_network(max_q_targets.view(-1, self.num_agents), states.view(-1, self.state_dim)).view(batch_size, max_len, 1)

        targets = rewards_batch + self.gamma * mask_batch * q_total_target
        
        td_error = (q_total_eval - targets.detach()) * mask_batch
        loss = (td_error ** 2).sum() / mask_batch.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_targets(self):
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        self.target_mixer_network.load_state_dict(self.mixer_network.state_dict())

    def reset_hidden_states(self, batch_size=1):
        self.agent_hidden_state = torch.zeros(batch_size, self.hidden_sizes[-1]).to(self.device)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'agent_state_dict': self.agent_network.state_dict(),
            'mixer_state_dict': self.mixer_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.agent_network.load_state_dict(checkpoint['agent_state_dict'])
        self.mixer_network.load_state_dict(checkpoint['mixer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_targets()
        print(f"QMIX model loaded from {path}")
