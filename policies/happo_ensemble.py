import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
from copy import deepcopy

# The underlying network architecture is the same as MAPPO's
from .mappo import MAPPOActorCriticMLPNetwork
from .happo import HAPPOBuffer

class HAPPOEnsemble:
    """
    A training-time ensemble of HAPPO policies.
    Manages multiple HAPPO policies, each with its own target critic,
    and combines their outputs for robust decision-making and training.
    """
    def __init__(self, obs_space, action_space, global_state_dim, ensemble_size=5, hidden_sizes=(64, 64), lr=3e-4, eps=1e-8, clip_ratio=0.2, gamma=0.99, gae_lambda=0.95, value_coef=1.0, entropy_coef=0.01, max_grad_norm=0.5, target_kl=0.01, update_epochs=10, batch_size=64, tau=0.005, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ensemble_size = ensemble_size
        
        self.tau = tau
        self.value_clip_ratio = clip_ratio

        self.clip_ratio, self.gamma, self.gae_lambda = clip_ratio, gamma, gae_lambda
        self.value_coef, self.entropy_coef, self.max_grad_norm = value_coef, entropy_coef, max_grad_norm
        self.target_kl, self.update_epochs, self.batch_size = target_kl, update_epochs, batch_size

        obs_dim = obs_space.shape[0]
        action_dim = action_space.n

        self.policies = nn.ModuleList([
            MAPPOActorCriticMLPNetwork(obs_dim, action_dim, global_state_dim, hidden_sizes) 
            for _ in range(ensemble_size)
        ]).to(self.device)
        self.target_policies = nn.ModuleList([deepcopy(p) for p in self.policies]).to(self.device)
        self.optimizers = [optim.Adam(p.parameters(), lr=lr, eps=eps) for p in self.policies]

    def _soft_update_targets(self):
        """Soft update of all target networks in the ensemble."""
        for policy, target_policy in zip(self.policies, self.target_policies):
            for target_param, local_param in zip(target_policy.parameters(), policy.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def get_action(self, obs, action_mask=None, deterministic=False):
        """Gets a decentralized action by averaging actor probabilities."""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        mask_tensor = torch.BoolTensor(action_mask).to(self.device) if action_mask is not None else None
        
        all_logits = []
        with torch.no_grad():
            for policy in self.policies:
                dist, _ = policy.forward(obs_tensor, global_state=None, action_mask=mask_tensor)
                all_logits.append(dist.logits)

        avg_logits = torch.stack(all_logits).mean(dim=0)
        ensemble_dist = Categorical(logits=avg_logits)
        action = torch.argmax(ensemble_dist.probs, dim=-1) if deterministic else ensemble_dist.sample()
        log_prob = ensemble_dist.log_prob(action)
        action_probs = ensemble_dist.probs.cpu().numpy().flatten()
        return action.cpu().numpy(), log_prob.cpu().numpy(), action_probs

    def get_value(self, global_state):
        """Gets a centralized value by averaging the stable target critic estimates."""
        state_tensor = torch.FloatTensor(global_state).to(self.device)
        all_values = []
        with torch.no_grad():
            for target_policy in self.target_policies:
                value = target_policy.get_value(state_tensor)
                all_values.append(value)
        ensemble_value = torch.stack(all_values).mean()
        return ensemble_value.cpu().numpy().flatten()[0]

    def update(self, data):
        """Updates each policy in the ensemble using the HAPPO loss."""
        num_transitions = len(data['returns'])
        num_agents = len(data['obs']) // num_transitions
        total_kl, total_loss, num_updates = 0, 0, 0

        for i in range(self.update_epochs):
            perm = np.random.permutation(num_transitions)
            for start in range(0, num_transitions, self.batch_size):
                end = start + self.batch_size
                timestep_indices = perm[start:end]
                agent_indices = [idx * num_agents + agent_i for idx in timestep_indices for agent_i in range(num_agents)]
                batch_data = {k: v[timestep_indices] if k in ['global_states', 'returns', 'values'] else v[agent_indices] for k, v in data.items()}

                for policy, optimizer in zip(self.policies, self.optimizers):
                    loss, _, _, _, kl = self._compute_and_apply_loss(policy, optimizer, batch_data)
                    total_kl += kl
                    total_loss += loss.item()
                    num_updates += 1
                    if self.target_kl > 0 and kl > 1.5 * self.target_kl: break
                else: continue
                break
        
        self._soft_update_targets()
        return {
            "total_loss": total_loss / num_updates if num_updates > 0 else 0,
            "kl_divergence": total_kl / num_updates if num_updates > 0 else 0
        }

    def _compute_and_apply_loss(self, policy, optimizer, data):
        """Helper to compute and apply HAPPO loss for a single policy."""
        obs = torch.FloatTensor(data['obs']).to(self.device)
        actions = torch.LongTensor(data['actions']).to(self.device)
        advantages = torch.FloatTensor(data['advantages']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        action_masks = torch.BoolTensor(data['action_masks']).to(self.device)
        global_states = torch.FloatTensor(data['global_states']).to(self.device)
        returns = torch.FloatTensor(data['returns']).to(self.device)
        old_values = torch.FloatTensor(data['values']).to(self.device)

        new_log_probs, state_values, entropy = policy.evaluate_actions(obs, actions, global_state=global_states, action_mask=action_masks)
        state_values = state_values.squeeze(-1)

        ratio = torch.exp(new_log_probs - old_log_probs)
        policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages).mean()
        
        value_pred_clipped = old_values + torch.clamp(state_values - old_values, -self.value_clip_ratio, self.value_clip_ratio)
        value_loss = torch.max(F.mse_loss(state_values, returns, reduction='none'), F.mse_loss(value_pred_clipped, returns, reduction='none')).mean()
        
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
        
        with torch.no_grad(): kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()

        optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0: nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
        optimizer.step()
        
        return loss, policy_loss, value_loss, entropy, kl

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {f'policy_state_dict_{i}': p.state_dict() for i, p in enumerate(self.policies)}
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        for i, p in enumerate(self.policies):
            p.load_state_dict(checkpoint[f'policy_state_dict_{i}'])
            self.target_policies[i].load_state_dict(checkpoint[f'policy_state_dict_{i}'])
        print(f"Ensemble HAPPO model loaded from {path}")