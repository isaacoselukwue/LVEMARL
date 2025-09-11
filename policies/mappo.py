import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import scipy.signal

class MAPPOActorCriticMLPNetwork(nn.Module):
    """
    An Actor-Critic network for MAPPO.
    - The Actor is decentralized: it takes an agent's local observation.
    - The Critic is centralized: it takes the global state (concatenation of all observations).
    """
    def __init__(self, obs_dim, action_dim, global_state_dim, hidden_sizes=(256,)):
        super().__init__()
        
        # --- Actor Network ---
        actor_layers = []
        actor_input_dim = obs_dim
        for h_dim in hidden_sizes:
            actor_layers.append(nn.Linear(actor_input_dim, h_dim))
            actor_layers.append(nn.LayerNorm(h_dim))
            actor_layers.append(nn.ReLU())
            actor_input_dim = h_dim
        self.actor_mlp = nn.Sequential(*actor_layers)
        self.policy_head = nn.Linear(hidden_sizes[-1], action_dim)

        # --- Critic Network ---
        critic_layers = []
        critic_input_dim = global_state_dim
        for h_dim in hidden_sizes:
            critic_layers.append(nn.Linear(critic_input_dim, h_dim))
            critic_layers.append(nn.LayerNorm(h_dim))
            critic_layers.append(nn.ReLU())
            critic_input_dim = h_dim
        self.critic_mlp = nn.Sequential(*critic_layers)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, obs, global_state, action_mask=None):
        """
        Performs a forward pass to get the action distribution and state value.
        """
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        actor_features = self.actor_mlp(obs)
        action_logits = self.policy_head(actor_features)
        
        value = None
        if global_state is not None:
            if len(global_state.shape) == 1:
                global_state = global_state.unsqueeze(0)
            critic_features = self.critic_mlp(global_state)
            value = self.value_head(critic_features)

        if action_mask is not None:
            if len(action_mask.shape) == 1: action_mask = action_mask.unsqueeze(0)
            # action_logits is guaranteed to be 2D.
            action_logits[action_mask == 0] = -1e8
            
        return Categorical(logits=action_logits), value

    def evaluate_actions(self, obs, actions, global_state, action_mask=None):
        """
        Evaluates given actions to get their log probabilities, the state values,
        and the distribution's entropy. Used during training updates.
        """
        dist, value = self.forward(obs, global_state, action_mask)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy

    def get_action(self, obs, action_mask=None, deterministic=False):
        """
        Gets an action from the actor network (decentralized execution).
        """
        if len(obs.shape) == 1: obs = obs.unsqueeze(0)
        actor_features = self.actor_mlp(obs)
        action_logits = self.policy_head(actor_features)
        if action_mask is not None:
            if len(action_mask.shape) == 1: action_mask = action_mask.unsqueeze(0)
            action_logits[action_mask == 0] = -1e8
        dist = Categorical(logits=action_logits)
        action = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        action_probs = dist.probs.cpu().numpy().flatten()
        return action, log_prob, action_probs

    def get_value(self, global_state):
        """
        Gets a state-value from the centralized critic.
        """
        if len(global_state.shape) == 1: global_state = global_state.unsqueeze(0)
        critic_features = self.critic_mlp(global_state)
        value = self.value_head(critic_features)
        return value

class MAPPOBuffer:
    """
    A buffer for storing trajectories for MAPPO.
    Calculates per-agent advantages but uses a centralized value function.
    """
    def __init__(self, obs_shape, global_state_shape, action_dim, buffer_size, num_agents, gamma=0.99, gae_lambda=0.95):
        self.num_agents = num_agents
        self.obs_buffer = np.zeros((buffer_size, num_agents, *obs_shape), dtype=np.float32)
        self.global_state_buffer = np.zeros((buffer_size, *global_state_shape), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, num_agents), dtype=np.int64)
        self.action_mask_buffer = np.zeros((buffer_size, num_agents, action_dim), dtype=np.bool_)
        self.reward_buffer = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.value_buffer = np.zeros((buffer_size), dtype=np.float32)
        self.log_prob_buffer = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.done_buffer = np.zeros((buffer_size, num_agents), dtype=np.bool_)
        
        self.advantage_buffer = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.return_buffer = np.zeros((buffer_size), dtype=np.float32)
        
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ptr, self.max_size = 0, buffer_size

    def store(self, obs, global_state, action, reward, value, log_prob, done, action_masks):
        self.obs_buffer[self.ptr] = obs
        self.global_state_buffer[self.ptr] = global_state
        self.action_buffer[self.ptr] = action
        self.action_mask_buffer[self.ptr] = action_masks
        self.reward_buffer[self.ptr] = reward
        self.value_buffer[self.ptr] = value
        self.log_prob_buffer[self.ptr] = log_prob
        self.done_buffer[self.ptr] = done
        self.ptr += 1

    def _discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_path(self, last_value=0):
        """
        KEY CHANGE: Calculates per-agent advantages using their individual rewards
        but the shared, centralized value function as the baseline.
        """
        tiled_values = np.tile(self.value_buffer, (self.num_agents, 1)).T
        
        rewards_plus_v = np.vstack([self.reward_buffer, np.full((1, self.num_agents), last_value)])
        values_plus_v = np.vstack([tiled_values, np.full((1, self.num_agents), last_value)])

        deltas = rewards_plus_v[:-1] + self.gamma * values_plus_v[1:] - values_plus_v[:-1]
        for agent_idx in range(self.num_agents):
            self.advantage_buffer[:, agent_idx] = self._discount_cumsum(deltas[:, agent_idx], self.gamma * self.gae_lambda)
        
        self.return_buffer = self.advantage_buffer.mean(axis=1) + self.value_buffer

    def get_data(self):
        assert self.ptr == self.max_size, "Buffer must be full before getting data."
        self.ptr = 0
        
        adv_mean = np.mean(self.advantage_buffer)
        adv_std = np.std(self.advantage_buffer) + 1e-8
        self.advantage_buffer = (self.advantage_buffer - adv_mean) / adv_std
        
        ret_mean = np.mean(self.return_buffer)
        ret_std = np.std(self.return_buffer) + 1e-8
        self.return_buffer = (self.return_buffer - ret_mean) / ret_std

        data = dict(
            obs=self.obs_buffer.reshape(-1, *self.obs_buffer.shape[2:]),
            actions=self.action_buffer.flatten(),
            action_masks=self.action_mask_buffer.reshape(-1, self.action_mask_buffer.shape[-1]),
            log_probs=self.log_prob_buffer.flatten(),
            advantages=self.advantage_buffer.flatten(),
            global_states=self.global_state_buffer,
            returns=self.return_buffer
        )
        return data

class MAPPO:
    def __init__(self, obs_space, action_space, global_state_dim, hidden_sizes=(64, 64), lr=3e-4, eps=1e-8, clip_ratio=0.2, gamma=0.99, gae_lambda=0.95, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, target_kl=0.01, update_epochs=10, batch_size=64, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_ratio, self.gamma, self.gae_lambda = clip_ratio, gamma, gae_lambda
        self.value_coef, self.entropy_coef, self.max_grad_norm = value_coef, entropy_coef, max_grad_norm
        self.target_kl, self.update_epochs, self.batch_size = target_kl, update_epochs, batch_size
        
        obs_dim = obs_space.shape[0]
        self.policy = MAPPOActorCriticMLPNetwork(obs_dim, action_space.n, global_state_dim, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=eps)

    def compute_loss(self, data):
        obs = torch.FloatTensor(data['obs']).to(self.device)
        actions = torch.LongTensor(data['actions']).to(self.device)
        advantages = torch.FloatTensor(data['advantages']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        action_masks = torch.BoolTensor(data['action_masks']).to(self.device)

        global_states = torch.FloatTensor(data['global_states']).to(self.device)
        returns = torch.FloatTensor(data['returns']).to(self.device)

        new_log_probs, state_values, entropy = self.policy.evaluate_actions(obs, actions, global_state=global_states, action_mask=action_masks)
        state_values = state_values.squeeze(-1)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
        
        value_loss = F.mse_loss(state_values, returns)
        
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
        
        with torch.no_grad():
            kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()
            
        return loss, policy_loss, value_loss, entropy, kl

    def update(self, data):
        num_transitions = len(data['returns'])
        num_agents = len(data['obs']) // num_transitions
        
        for i in range(self.update_epochs):
            perm = np.random.permutation(num_transitions)
            
            for start in range(0, num_transitions, self.batch_size):
                end = start + self.batch_size
                timestep_indices = perm[start:end]
                
                agent_indices = []
                for idx in timestep_indices:
                    agent_indices.extend(range(idx * num_agents, (idx + 1) * num_agents))

                batch_data = dict(
                    obs=data['obs'][agent_indices],
                    actions=data['actions'][agent_indices],
                    log_probs=data['log_probs'][agent_indices],
                    advantages=data['advantages'][agent_indices],
                    global_states=data['global_states'][timestep_indices],
                    returns=data['returns'][timestep_indices],
                    action_masks=data['action_masks'][agent_indices]
                )

                loss, _, _, _, kl = self.compute_loss(batch_data)

                if self.target_kl > 0 and kl > 1.5 * self.target_kl:
                    break
                
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return {"total_loss": loss.item(), "kl_divergence": kl}

    def get_action(self, obs, action_mask=None, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        mask_tensor = torch.BoolTensor(action_mask).to(self.device) if action_mask is not None else None
        with torch.no_grad():
            action, log_prob, action_probs = self.policy.get_action(obs_tensor, mask_tensor, deterministic)
        return action.cpu().numpy(), log_prob.cpu().numpy(), action_probs

    def get_value(self, global_state):
        state_tensor = torch.FloatTensor(global_state).to(self.device)
        with torch.no_grad():
            value = self.policy.get_value(state_tensor)
        return value.cpu().numpy().flatten()[0]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
