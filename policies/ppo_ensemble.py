import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from .ppo import ActorCriticMLPNetwork 

class PPOEnsemble:
    """
    A training-time ensemble of PPO policies.
    It manages multiple ActorCritic networks, trains them on the same data,
    and makes decisions by combining their outputs.
    """
    def __init__(self, obs_space, action_space, ensemble_size=5, hidden_sizes=(64, 64), lr=3e-4, eps=1e-8, clip_ratio=0.2, gamma=0.99, gae_lambda=0.95, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, target_kl=0.01, update_epochs=10, batch_size=64, device=None):
        """
        Initializes the PPO Ensemble.

        Args:
            ensemble_size (int): The number of models in the ensemble.
            All other arguments are standard PPO hyperparameters.
        """
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
            ActorCriticMLPNetwork(obs_dim, action_dim, hidden_sizes) for _ in range(ensemble_size)
        ]).to(self.device)
        
        self.optimizers = [
            optim.Adam(policy.parameters(), lr=lr, eps=eps) for policy in self.policies
        ]

    def get_action(self, obs, deterministic=False):
        """
        Gets an action from the ensemble and also returns action probabilities.
        """
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        all_logits, all_values = [], []
        with torch.no_grad():
            for policy in self.policies:
                dist, value = policy(obs_tensor)
                all_logits.append(dist.logits)
                all_values.append(value)

            avg_logits = torch.stack(all_logits).mean(dim=0)
            ensemble_dist = Categorical(logits=avg_logits)
            action = torch.argmax(ensemble_dist.probs, dim=-1) if deterministic else ensemble_dist.sample()
            ensemble_value = torch.stack(all_values).mean(dim=0)
            log_prob = ensemble_dist.log_prob(action)
            
            action_probs = ensemble_dist.probs.cpu().numpy().flatten()

        return action.cpu().numpy(), log_prob.cpu().numpy(), ensemble_value.cpu().numpy().flatten(), action_probs

    def update(self, data):
        """
        Updates each policy in the ensemble using bootstrapped data to ensure diversity.
        This is the key to effective ensemble training.
        """
        data_tensors = {
            k: torch.from_numpy(v).to(self.device) for k, v in data.items()
        }
        num_transitions = len(data_tensors['returns'])
        
        total_kl_avg = 0
        total_loss_avg = 0
        
        for policy, optimizer in zip(self.policies, self.optimizers):
            
            policy_total_loss = 0
            policy_total_kl = 0
            policy_update_count = 0

            # --- BOOTSTRAPPING: Creates a unique, random sample for THIS policy ---
            # This is the most important part for making the ensemble work.
            bootstrap_indices = np.random.choice(num_transitions, size=num_transitions, replace=True)
            
            for _ in range(self.update_epochs):
                np.random.shuffle(bootstrap_indices)
                
                for start in range(0, num_transitions, self.batch_size):
                    end = start + self.batch_size
                    batch_indices = bootstrap_indices[start:end]
                    
                    batch_data = {k: v[batch_indices] for k, v in data_tensors.items()}

                    loss, _, _, _, kl = self._compute_and_apply_loss(policy, optimizer, batch_data)
                    
                    policy_total_loss += loss.item()
                    policy_total_kl += kl
                    policy_update_count += 1

                    if self.target_kl > 0 and kl > 1.5 * self.target_kl:
                        break
                else:
                    continue
                break

            # Aggregate the average loss and KL for this policy
            if policy_update_count > 0:
                total_loss_avg += (policy_total_loss / policy_update_count)
                total_kl_avg += (policy_total_kl / policy_update_count)

        return {
            "total_loss": total_loss_avg / self.ensemble_size,
            "kl_divergence": total_kl_avg / self.ensemble_size
        }
    def _compute_and_apply_loss(self, policy, optimizer, data):
        """Helper function to compute and apply loss for a single policy."""
        obs = data['obs']
        actions = data['actions']
        returns = data['returns']
        advantages = data['advantages']
        old_log_probs = data['log_probs']
        action_masks = data['action_masks']
        
        new_log_probs, state_values, entropy = policy.evaluate_actions(obs, actions, action_masks)
        state_values = state_values.squeeze(-1)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
        value_loss = nn.functional.mse_loss(state_values, returns)
        
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
        
        with torch.no_grad():
            kl = ((torch.exp(new_log_probs - old_log_probs) - 1) - (new_log_probs - old_log_probs)).mean().item()

        optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
        optimizer.step()
        
        return loss, policy_loss, value_loss, entropy, kl

    def save(self, path):
        """Saves the state of all policies and optimizers."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            f'policy_state_dict_{i}': policy.state_dict()
            for i, policy in enumerate(self.policies)
        }
        checkpoint.update({
            f'optimizer_state_dict_{i}': optimizer.state_dict()
            for i, optimizer in enumerate(self.optimizers)
        })
        torch.save(checkpoint, path)

    def load(self, path):
        """Loads the state for all policies and optimizers."""
        checkpoint = torch.load(path, map_location=self.device)
        for i, policy in enumerate(self.policies):
            policy.load_state_dict(checkpoint[f'policy_state_dict_{i}'])
        for i, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(checkpoint[f'optimizer_state_dict_{i}'])
        print(f"Ensemble model loaded from {path}")
