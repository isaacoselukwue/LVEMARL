import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import cv2
from gymnasium.core import ObservationWrapper
from pettingzoo.utils.wrappers import BaseWrapper
import scipy.signal

class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_sizes=(256,)):
        super(ActorCriticNetwork, self).__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(in_channels=obs_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_obs = torch.zeros(1, *obs_shape)
            conv_out_size = self.conv_base(dummy_obs).shape[1]
        self.shared_mlp = nn.Sequential(nn.Linear(conv_out_size, hidden_sizes[0]), nn.ReLU())
        self.policy_head = nn.Linear(hidden_sizes[0], action_dim)
        self.value_head = nn.Linear(hidden_sizes[0], 1)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, x, action_mask=None):
        if len(x.shape) == 3: x = x.unsqueeze(0)
        x = self.conv_base(x)
        x = self.shared_mlp(x)
        action_logits = self.policy_head(x)
        value = self.value_head(x)
        if action_mask is not None:
            if len(action_mask.shape) == 1: action_mask = action_mask.unsqueeze(0)
            action_logits[action_mask == 0] = -1e8
        return Categorical(logits=action_logits), value

    def get_action(self, x, action_mask=None, deterministic=False):
        dist, value = self.forward(x, action_mask)
        action = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, x, actions, action_mask=None):
        dist, value = self.forward(x, action_mask)
        return dist.log_prob(actions), value, dist.entropy()

class ActorCriticMLPNetwork(nn.Module):
    """ An Actor-Critic network that uses MLPs for both actor and critic. """
    def __init__(self, obs_dim, action_dim, hidden_sizes=(256,)):
        super().__init__()
        
        layers = []
        input_dim = obs_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        
        self.shared_mlp = nn.Sequential(*layers)
        
        self.policy_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, x, action_mask=None):
        if len(x.shape) == 1: x = x.unsqueeze(0)
        x = self.shared_mlp(x)
        action_logits = self.policy_head(x)
        value = self.value_head(x)
        if action_mask is not None:
            if len(action_mask.shape) == 1: action_mask = action_mask.unsqueeze(0)
            action_logits[action_mask == 0] = -1e8
        return Categorical(logits=action_logits), value

    def get_action(self, x, action_mask=None, deterministic=False):
        dist, value = self.forward(x, action_mask)
        action = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, x, actions, action_mask=None):
        dist, value = self.forward(x, action_mask)
        return dist.log_prob(actions), value, dist.entropy()


class PPOBuffer:
    def __init__(self, obs_shape, action_dim, buffer_size, num_agents, gamma=0.99, gae_lambda=0.95):
        self.obs_buffer = np.zeros((buffer_size, num_agents, *obs_shape), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, num_agents), dtype=np.int64)
        self.action_mask_buffer = np.zeros((buffer_size, num_agents, action_dim), dtype=np.bool_)
        self.reward_buffer = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.value_buffer = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.log_prob_buffer = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.done_buffer = np.zeros((buffer_size, num_agents), dtype=np.bool_)
        self.advantage_buffer = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.return_buffer = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ptr, self.max_size = 0, buffer_size

    def store(self, obs, action, reward, value, log_prob, done, action_mask):
        self.obs_buffer[self.ptr] = obs
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.value_buffer[self.ptr] = value
        self.log_prob_buffer[self.ptr] = log_prob
        self.done_buffer[self.ptr] = done
        self.action_mask_buffer[self.ptr] = action_mask
        self.ptr += 1

    def _discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_path(self, last_values):
        rewards = np.vstack([self.reward_buffer, last_values])
        values = np.vstack([self.value_buffer, last_values])
        dones = np.vstack([self.done_buffer, np.zeros_like(last_values)])
        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1]) - values[:-1]
        for agent in range(deltas.shape[1]):
            self.advantage_buffer[:, agent] = self._discount_cumsum(deltas[:, agent], self.gamma * self.gae_lambda)
        self.return_buffer = self.advantage_buffer + self.value_buffer

    def get_data(self):
        assert self.ptr == self.max_size, "Buffer must be full before getting data."
        self.ptr = 0
        
        adv_mean = np.mean(self.advantage_buffer)
        adv_std = np.std(self.advantage_buffer) + 1e-8
        self.advantage_buffer = (self.advantage_buffer - adv_mean) / adv_std
        
        ret_mean = np.mean(self.return_buffer)
        ret_std = np.std(self.return_buffer) + 1e-8
        self.return_buffer = (self.return_buffer - ret_mean) / ret_std

        return dict(
            obs=self.obs_buffer.reshape(-1, *self.obs_buffer.shape[2:]),
            actions=self.action_buffer.flatten(),
            action_masks=self.action_mask_buffer.reshape(-1, self.action_mask_buffer.shape[-1]),
            returns=self.return_buffer.flatten(), # Now normalised
            advantages=self.advantage_buffer.flatten(),
            log_probs=self.log_prob_buffer.flatten()
        )

class PPO:
    def __init__(self, obs_space, action_space, hidden_sizes=(64, 64), lr=3e-4, eps=1e-8, clip_ratio=0.2, gamma=0.99, gae_lambda=0.95, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, target_kl=0.01, update_epochs=10, batch_size=64, device=None, use_cnn=False):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_ratio, self.gamma, self.gae_lambda = clip_ratio, gamma, gae_lambda
        self.value_coef, self.entropy_coef, self.max_grad_norm = value_coef, entropy_coef, max_grad_norm
        self.target_kl, self.update_epochs, self.batch_size = target_kl, update_epochs, batch_size
        if use_cnn:
            self.policy = ActorCriticNetwork(obs_space.shape, action_space.n, hidden_sizes).to(self.device)
        else:
            obs_dim = obs_space.shape[0]
            self.policy = ActorCriticMLPNetwork(obs_dim, action_space.n, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=eps)

    def compute_loss(self, data):
        obs = torch.FloatTensor(data['obs']).to(self.device)
        actions = torch.LongTensor(data['actions']).to(self.device)
        returns = torch.FloatTensor(data['returns']).to(self.device)
        advantages = torch.FloatTensor(data['advantages']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        action_masks = torch.BoolTensor(data['action_masks']).to(self.device)
        new_log_probs, state_values, entropy = self.policy.evaluate_actions(obs, actions, action_masks)
        state_values = state_values.squeeze(-1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
        value_loss = F.mse_loss(state_values, returns)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
        with torch.no_grad():
            kl = ((torch.exp(new_log_probs - old_log_probs) - 1) - (new_log_probs - old_log_probs)).mean().item()
        return loss, policy_loss, value_loss, entropy, kl

    def update(self, data):
        indices = np.arange(len(data['obs']))
        
        for i in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_data = {k: v[batch_indices] for k, v in data.items()}
                loss, _, _, _, kl = self.compute_loss(batch_data)

                if kl > 1.5 * self.target_kl:
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
            dist, value = self.policy(obs_tensor, mask_tensor)
            action = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            action_probs = dist.probs.cpu().numpy().flatten()
        
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy(), action_probs

    def save(self, path):
        """Saves the policy and optimizer state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Loads the policy and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")


class PPOTrainer:
    """
    Trainer class for PPO, handling the training loop and logging.
    """
    def __init__(
        self,
        ppo_agent,
        env,
        steps_per_epoch=4000,
        epochs=100,
        log_dir='logs',
        save_dir='models',
        save_freq=10,
        eval_freq=5,
        num_eval_episodes=10,
        max_ep_len=1000,
        seed=0
    ):
        self.ppo_agent = ppo_agent
        self.env = env
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.max_ep_len = max_ep_len
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.log_dir = log_dir
        self.save_dir = save_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        
        self.buffer = PPOBuffer(
            obs_dim=ppo_agent.obs_dim,
            action_dim=1,
            buffer_size=steps_per_epoch,
            gamma=ppo_agent.gamma,
            gae_lambda=ppo_agent.gae_lambda
        )
    
    def train(self):
        """
        Main training loop.
        """
        start_time = time.time()
        
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        ep_return = 0
        ep_length = 0
        
        for epoch in range(1, self.epochs + 1):
            for t in range(self.steps_per_epoch):
                action, log_prob, value = self.ppo_agent.get_action(obs)
                
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = step_result
                
                self.buffer.store(obs, action, reward, value, log_prob, done)
                
                obs = next_obs
                
                ep_return += reward
                ep_length += 1
                
                timeout = ep_length == self.max_ep_len
                terminal = done or timeout
                epoch_ended = t == self.steps_per_epoch - 1
                
                if terminal or epoch_ended:
                    if epoch_ended and not terminal:
                        _, _, value = self.ppo_agent.get_action(obs)
                        self.buffer.finish_path(value)
                    else:
                        self.buffer.finish_path(0)
                    
                    if terminal:
                        self.writer.add_scalar('train/episode_return', ep_return, epoch)
                        self.writer.add_scalar('train/episode_length', ep_length, epoch)
                        
                        reset_result = self.env.reset()
                        if isinstance(reset_result, tuple):
                            obs = reset_result[0]
                        else:
                            obs = reset_result
                        ep_return = 0
                        ep_length = 0
            
            data = self.buffer.get_data()
            logs = self.ppo_agent.update(data)
            
            for key, value in logs.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            
            if epoch % self.eval_freq == 0:
                eval_returns = self.evaluate()
                self.writer.add_scalar('eval/mean_return', eval_returns.mean(), epoch)
                self.writer.add_scalar('eval/std_return', eval_returns.std(), epoch)
                print(f"Epoch {epoch}: Eval mean return = {eval_returns.mean():.2f} ± {eval_returns.std():.2f}")
            
            if epoch % self.save_freq == 0:
                save_path = os.path.join(self.save_dir, f'ppo_epoch_{epoch}.pt')
                self.ppo_agent.save(save_path)
            
            time_taken = time.time() - start_time
            self.writer.add_scalar('time/time_elapsed', time_taken, epoch)
            print(f"Epoch {epoch}/{self.epochs}: Loss = {logs['total_loss']:.4f}, Time = {time_taken:.2f}s")
        
        save_path = os.path.join(self.save_dir, 'ppo_final.pt')
        self.ppo_agent.save(save_path)
        
        self.writer.close()
    
    def evaluate(self):
        """
        Evaluate the policy.
        
        Returns:
            episode_returns: Array of episode returns.
        """
        episode_returns = np.zeros(self.num_eval_episodes)
        
        for i in range(self.num_eval_episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result
            done = False
            ep_return = 0
            
            while not done:
                action, _, _ = self.ppo_agent.get_action(obs, deterministic=True)
                
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, _ = step_result
                
                ep_return += reward
                
                if done:
                    episode_returns[i] = ep_return
                    break
        
        return episode_returns

class ParallelPPOTrainer:
    def __init__(self, ppo_agent, parallel_env, steps_per_epoch=2048, epochs=500, log_dir='logs_parallel', save_dir='models_parallel', save_name='ppo_agent.pt', eval_freq=10, num_eval_episodes=10):
        self.agent = ppo_agent
        self.env = parallel_env
        self.save_name = save_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.steps_per_epoch, self.epochs = steps_per_epoch, epochs
        self.agents = self.env.possible_agents
        self.num_agents = len(self.agents)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.total_steps = 0
        
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes

        if callable(self.env.observation_space):
            agent_obs_space = self.env.observation_space(self.agents[0])
        else:
            agent_obs_space = self.env.observation_space[self.agents[0]]
        self.is_dict_obs = isinstance(agent_obs_space, gym.spaces.Dict)
        
        obs_shape = agent_obs_space['observation'].shape if self.is_dict_obs else agent_obs_space.shape
        action_dim = self.env.action_space(self.agents[0]).n
        self.buffer = PPOBuffer(obs_shape, action_dim, self.steps_per_epoch, self.num_agents)

    def evaluate(self):
        """
        Evaluate the agent's performance with a deterministic policy.
        """
        print("\n--- Evaluating ---")
        total_returns = {agent: [] for agent in self.agents}

        for i in range(self.num_eval_episodes):
            obs, infos = self.env.reset()
            ep_returns = {agent: 0 for agent in self.agents}
            
            while self.env.agents:
                active_agents = list(obs.keys())
                
                observations_np = np.array([obs[agent] for agent in active_agents])
                
                actions_list, _, _ = self.agent.get_action(observations_np, deterministic=True)
                
                env_actions = {agent: act for agent, act in zip(active_agents, actions_list)}
                obs, rewards, _, _, _ = self.env.step(env_actions)

                for agent in active_agents:
                    ep_returns[agent] += rewards[agent]
            
            for agent in self.agents:
                total_returns[agent].append(ep_returns[agent])

        mean_returns = {agent: np.mean(returns) for agent, returns in total_returns.items()}
        for agent, mean_ret in mean_returns.items():
            print(f"  Agent {agent}: Mean Return = {mean_ret:.2f}")
            self.writer.add_scalar(f'eval/mean_return_{agent}', mean_ret, self.total_steps)
        print("--- Done Evaluating ---\n")
        return mean_returns


    def train(self):
        start_time = time.time()
        obs, infos = self.env.reset()
        ep_returns = {agent: 0 for agent in self.agents}
        last_obs_np = {agent: obs[agent] for agent in self.agents}

        for epoch in range(1, self.epochs + 1):
            for t in range(self.steps_per_epoch):
                observations_np = np.array([obs.get(agent, last_obs_np[agent]) for agent in self.agents])
                action_masks_np = np.array([infos.get(agent, {}).get('legal_moves', np.ones(self.env.action_space(agent).n, dtype=np.int8)) for agent in self.agents])
                actions, log_probs, values = self.agent.get_action(observations_np, action_masks_np)
                
                active_agents = list(obs.keys())
                agent_idx_map = {agent: i for i, agent in enumerate(self.agents)}
                env_actions = {agent: actions[agent_idx_map[agent]] for agent in active_agents}
                next_obs, rewards, dones, truncs, infos = self.env.step(env_actions)
                
                rewards_arr = np.array([rewards.get(agent, 0) for agent in self.agents])
                dones_arr = np.array([dones.get(agent, True) or truncs.get(agent, True) for agent in self.agents])
                self.buffer.store(observations_np, actions, rewards_arr, values.flatten(), log_probs, dones_arr, action_masks_np)
                
                obs = next_obs
                for agent_id in self.agents:
                    if agent_id in next_obs:
                        last_obs_np[agent_id] = next_obs[agent_id]
                    ep_returns[agent_id] += rewards.get(agent_id, 0)
                    if dones.get(agent_id, False) or truncs.get(agent_id, False):
                        self.writer.add_scalar(f'returns/{agent_id}', ep_returns[agent_id], self.total_steps)
                        ep_returns[agent_id] = 0
                self.total_steps += self.num_agents

            last_values_np = np.zeros(self.num_agents)
            if obs:
                final_obs_np = np.array([obs.get(agent, last_obs_np[agent]) for agent in self.agents])
                _, _, last_values = self.agent.get_action(final_obs_np, deterministic=True)
                last_values_np = last_values.flatten()
            self.buffer.finish_path(last_values_np)
            
            data = self.buffer.get_data()
            logs = self.agent.update(data)
            
            self.writer.add_scalar('train/total_loss', logs['total_loss'], epoch)
            self.writer.add_scalar('train/kl_divergence', logs['kl_divergence'], epoch)
            print(f"Epoch {epoch}/{self.epochs} | Loss: {logs['total_loss']:.3f} | KL: {logs['kl_divergence']:.4f} | Time: {time.time() - start_time:.1f}s")

            if epoch % self.eval_freq == 0:
                self.evaluate()

class PreprocessFrameSingleAgent(ObservationWrapper):
    """
    Downsamples, grayscales, and NORMALIZES the observation frame.
    """
    def __init__(self, env, new_shape):
        super().__init__(env)
        self.new_shape = new_shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1, *new_shape), dtype=np.float32
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.new_shape, interpolation=cv2.INTER_AREA)
        return (obs.reshape(1, *self.new_shape) / 255.0).astype(np.float32)

class PreprocessFrame(BaseWrapper):
    """
    Downsamples, grayscales, and NORMALIZES the observation frame.
    """
    def __init__(self, env, new_shape):
        super().__init__(env)
        self.new_shape = new_shape
        self.observation_space = {
            agent: gym.spaces.Box(low=0.0, high=1.0, shape=(1, *new_shape), dtype=np.float32)
            for agent in self.env.possible_agents
        }

    def _process_obs(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.new_shape, interpolation=cv2.INTER_AREA)
        return (obs.reshape(1, *self.new_shape) / 255.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed, options)
        processed_obs = {agent: self._process_obs(o) for agent, o in obs.items()}
        return processed_obs, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        processed_obs = {agent: self._process_obs(o) for agent, o in obs.items()}
        return processed_obs, reward, terminated, truncated, info

# Example usage:
# if __name__ == "__main__":
#     import gym
    
#     # Create environment
#     env = gym.make("CartPole-v1")
    
#     # Initialize PPO agent
#     ppo_agent = PPO(
#         observation_space=env.observation_space,
#         action_space=env.action_space,
#         hidden_sizes=(64, 64),
#         lr=3e-4,
#         gamma=0.99,
#         clip_ratio=0.2,
#         value_coef=0.5,
#         entropy_coef=0.01,
#         update_epochs=10,
#         batch_size=64
#     )
    
#     # Initialize trainer
#     trainer = PPOTrainer(
#         ppo_agent=ppo_agent,
#         env=env,
#         steps_per_epoch=4000,
#         epochs=100,
#         log_dir='logs/ppo_cartpole',
#         save_dir='models/ppo_cartpole',
#         save_freq=10,
#         eval_freq=5
#     )
    
#     # Train the agent
#     trainer.train()
    
    
from pettingzoo.butterfly import pistonball_v6
if __name__ == "__main__":
    env = pistonball_v6.parallel_env(n_pistons=4, continuous=False)
    
    env = PreprocessFrame(env, new_shape=(84, 84))
    
    agent_obs_space = env.observation_space[env.possible_agents[0]]
    agent_action_space = env.action_space(env.possible_agents[0])

    ppo_agent = PPO(
        obs_space=agent_obs_space,
        action_space=agent_action_space,
        hidden_sizes=(256,),
        lr=1e-4,
        entropy_coef=0.01,
    )
    
    trainer = ParallelPPOTrainer(
        ppo_agent=ppo_agent,
        parallel_env=env,
        steps_per_epoch=2048,
        epochs=500
    )
    
    trainer.train()
    final_save_path = os.path.join(trainer.save_dir, trainer.save_name)
    ppo_agent.save(final_save_path)
    print(f"\nTraining complete. Model saved to {final_save_path}")

# if __name__ == "__main__":
#     # --- NEW: Configurable Model Path ---
#     MODEL_PATH = "models_parallel/ppo_agent.pt" # Easily change this path

#     # 1. Set up the environment (must match training setup)
#     env = pistonball_v6.parallel_env(n_pistons=4, continuous=False)
#     env = PreprocessFrame(env, new_shape=(84, 84))
    
#     agent_obs_space = env.observation_space[env.possible_agents[0]]
#     agent_action_space = env.action_space(env.possible_agents[0])

#     # 2. Instantiate the PPO agent
#     ppo_agent = PPO(
#         obs_space=agent_obs_space,
#         action_space=agent_action_space,
#         hidden_sizes=(256,),
#     )
    
#     # 3. Load the saved weights from the specified path
#     if os.path.exists(MODEL_PATH):
#         ppo_agent.load(MODEL_PATH)
#     else:
#         print(f"Error: Model not found at {MODEL_PATH}")
#         exit()

#     # 4. Instantiate the trainer and run evaluation
#     trainer = ParallelPPOTrainer(
#         ppo_agent=ppo_agent,
#         parallel_env=env,
#         num_eval_episodes=10 
#     )

#     trainer.evaluate()