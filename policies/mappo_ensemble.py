import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F

from .mappo import MAPPOActorCriticMLPNetwork

class MAPPOEnsemble:
    """
    A training-time ensemble of MAPPO policies.
    It manages multiple MAPPOActorCritic networks, where each has a
    decentralized actor and a centralized critic.
    """
    def __init__(self, obs_space, action_space, global_state_dim, ensemble_size=5, hidden_sizes=(64, 64), lr=3e-4, eps=1e-8, clip_ratio=0.2, gamma=0.99, gae_lambda=0.95, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, target_kl=0.01, update_epochs=10, batch_size=64, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ensemble_size = ensemble_size
        
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        obs_dim = obs_space.shape[0]
        action_dim = action_space.n

        self.policies = nn.ModuleList([
            MAPPOActorCriticMLPNetwork(obs_dim, action_dim, global_state_dim, hidden_sizes) 
            for _ in range(ensemble_size)
        ]).to(self.device)
        
        self.optimizers = [
            optim.Adam(policy.parameters(), lr=lr, eps=eps) for policy in self.policies
        ]

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
        """Gets a centralized value by averaging critic estimates."""
        state_tensor = torch.FloatTensor(global_state).to(self.device)
        
        all_values = []
        with torch.no_grad():
            for policy in self.policies:
                value = policy.get_value(state_tensor)
                all_values.append(value)
        
        ensemble_value = torch.stack(all_values).mean()
        return ensemble_value.cpu().numpy().flatten()[0]

    def update(self, data):
        """Updates each policy in the ensemble on the same batch of data."""
        num_transitions = len(data['returns'])
        num_agents = len(data['obs']) // num_transitions
        
        total_kl = 0
        total_loss = 0
        num_updates = 0

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

                for policy, optimizer in zip(self.policies, self.optimizers):
                    loss, _, _, _, kl = self._compute_and_apply_loss(policy, optimizer, batch_data)
                    total_kl += kl
                    total_loss += loss.item()
                    num_updates += 1

                    if self.target_kl > 0 and kl > 1.5 * self.target_kl:
                        break
                else:
                    continue
                break

        return {
            "total_loss": total_loss / num_updates if num_updates > 0 else 0,
            "kl_divergence": total_kl / num_updates if num_updates > 0 else 0
        }

    def _compute_and_apply_loss(self, policy, optimizer, data):
        """Helper to compute and apply loss for a single MAPPO policy."""
        obs = torch.FloatTensor(data['obs']).to(self.device)
        actions = torch.LongTensor(data['actions']).to(self.device)
        advantages = torch.FloatTensor(data['advantages']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        action_masks = torch.BoolTensor(data['action_masks']).to(self.device)
        global_states = torch.FloatTensor(data['global_states']).to(self.device)
        returns = torch.FloatTensor(data['returns']).to(self.device)

        new_log_probs, state_values, entropy = policy.evaluate_actions(obs, actions, global_state=global_states, action_mask=action_masks)
        state_values = state_values.squeeze(-1)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
        
        value_loss = F.mse_loss(state_values, returns)
        
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
        
        with torch.no_grad():
            kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()

        optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
        optimizer.step()
        
        return loss, policy_loss, value_loss, entropy, kl

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {f'policy_state_dict_{i}': p.state_dict() for i, p in enumerate(self.policies)}
        checkpoint.update({f'optimizer_state_dict_{i}': o.state_dict() for i, o in enumerate(self.optimizers)})
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        for i, p in enumerate(self.policies):
            p.load_state_dict(checkpoint[f'policy_state_dict_{i}'])
        for i, o in enumerate(self.optimizers):
            o.load_state_dict(checkpoint[f'optimizer_state_dict_{i}'])
        print(f"Ensemble MAPPO model loaded from {path}")
