import os
import time
import yaml
import numpy as np
from gym import spaces
from pommerman import make, agents, constants
from collections import deque
import scipy.stats

from policies.ppo import PPO, PPOBuffer
from .llm_integration.log_utils import CsvLogger

class PPOAgent(agents.BaseAgent):
    """
    An agent that uses a single PPO policy, now with comprehensive logging.
    """
    def __init__(self, config_path=None, config=None, **kwargs):
        super(PPOAgent, self).__init__(**kwargs)

        if config: self.config = config
        elif config_path:
            with open(config_path, 'r') as f: self.config = yaml.safe_load(f)
        else: raise ValueError("Config must be provided.")

        env_config = self.config['env']
        self.agent_list = [self, agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()]
        self.env = make(env_config['scenario'], self.agent_list)
        self.training_agent_id = 0 
        self.env.set_training_agent(self.training_agent_id)

        ppo_config = self.config['policy']['ppo']
        dummy_obs = self.env.reset()
        featurized_obs = self.env.featurize(dummy_obs[0])
        obs_dim = featurized_obs.shape[0]
        observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        action_space = self.env.action_space

        self.policy = PPO(
            obs_space=observation_space, action_space=action_space, use_cnn=False,
            hidden_sizes=tuple(ppo_config['hidden_sizes']), lr=ppo_config['lr'], gamma=ppo_config['gamma'],
            clip_ratio=ppo_config['clip_ratio'], value_coef=ppo_config['value_coef'],
            entropy_coef=ppo_config['entropy_coef'], update_epochs=ppo_config['update_epochs'],
            batch_size=ppo_config['batch_size']
        )
        
        self.logger = CsvLogger(log_dir=self.config['trainer']['log_dir'], config=self.config)
        self.total_steps = 0
        self.episode_num = 0

    def act(self, obs, action_space):
        featurized_obs = self.env.featurize(obs)
        action, _, _, _ = self.policy.get_action(featurized_obs, deterministic=True)
        return action

    def train(self, is_tuning=False):
        if not is_tuning: print("Starting PPO training...")
        trainer_config = self.config['trainer']
        buffer = PPOBuffer(
            obs_shape=(self.policy.policy.shared_mlp[0].in_features,),
            action_dim=self.env.action_space.n, buffer_size=trainer_config['steps_per_epoch'],
            num_agents=1, gamma=self.policy.gamma, gae_lambda=self.policy.gae_lambda
        )

        obs = self.env.reset()
        start_time = time.time()
        recent_wins = deque(maxlen=50)
        ep_return, ep_len, ep_entropy, ep_prob_gap = 0, 0, 0, 0

        for epoch in range(trainer_config['epochs']):
            for t in range(trainer_config['steps_per_epoch']):
                self.total_steps += 1
                agent_obs = self.env.featurize(obs[self.training_agent_id])
                action, log_prob, value, action_probs = self.policy.get_action(agent_obs)
                
                ep_entropy += scipy.stats.entropy(action_probs)
                sorted_probs = np.sort(action_probs)[::-1]
                prob_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
                ep_prob_gap += prob_gap

                all_actions = self.env.act(obs) 
                full_actions = [0] * len(self.agent_list)
                full_actions[self.training_agent_id] = action
                other_agent_ids = [i for i in range(len(self.agent_list)) if i != self.training_agent_id]
                for i, agent_id in enumerate(other_agent_ids): full_actions[agent_id] = all_actions[i]

                next_obs, reward, done, info = self.env.step(full_actions)
                buffer.store(agent_obs[np.newaxis, :], action, reward[self.training_agent_id], value, log_prob, done, np.ones(self.env.action_space.n))
                obs = next_obs
                ep_return += reward[self.training_agent_id]; ep_len += 1

                if done:
                    self.episode_num += 1
                    is_win = 1.0 if info.get('result') == constants.Result.Win and self.training_agent_id in info.get('winners', []) else 0.0
                    recent_wins.append(is_win)
                    if not is_tuning:
                        print(f"Epoch {epoch}, Ep {self.episode_num}: Ret={ep_return:.2f}, Len={ep_len}, Res={info.get('result')}")
                        self.logger.log_training_episode({
                            "epoch": epoch, "episode_num": self.episode_num, "total_steps": self.total_steps,
                            "ep_return": ep_return, "ep_length": ep_len, "result": info.get('result').name,
                            "policy_entropy": ep_entropy / ep_len if ep_len > 0 else 0,
                            "action_prob_gap": ep_prob_gap / ep_len if ep_len > 0 else 0
                        })
                    obs = self.env.reset()
                    ep_return, ep_len, ep_entropy, ep_prob_gap = 0, 0, 0, 0

            if not done:
                agent_obs = self.env.featurize(obs[self.training_agent_id])
                _, _, last_val, _ = self.policy.get_action(agent_obs)
                buffer.finish_path(last_val)

            data = buffer.get_data(); logs = self.policy.update(data)
            
            if not is_tuning:
                self.logger.log_epoch({
                    "epoch": epoch + 1, "total_steps": self.total_steps,
                    "loss": logs['total_loss'], "kl_divergence": logs['kl_divergence']
                })
                print(f"Epoch {epoch+1}/{trainer_config['epochs']} | Loss: {logs['total_loss']:.3f} | KL: {logs['kl_divergence']:.4f} | Time: {time.time() - start_time:.1f}s")
                if (epoch + 1) % trainer_config['save_freq'] == 0:
                    self.policy.save(os.path.join(trainer_config['save_dir'], trainer_config['save_name']))

        if not is_tuning: print("Training finished.")
        self.env.close()
        if is_tuning: return sum(recent_wins) / len(recent_wins) if recent_wins else 0.0

    def run_evaluation_episodes(self, num_episodes, render=False):
        print(f"Running {num_episodes} evaluation episodes...")
        wins, total_return = 0, 0
        for i in range(num_episodes):
            obs = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                if render: self.env.render(); time.sleep(0.1)
                agent_obs = self.env.featurize(obs[self.training_agent_id])
                action, _, _, _ = self.policy.get_action(agent_obs, deterministic=True)
                other_actions = self.env.act(obs)
                full_actions = [0] * len(self.agent_list)
                full_actions[self.training_agent_id] = action
                other_agent_ids = [j for j in range(len(self.agent_list)) if j != self.training_agent_id]
                for j, agent_id in enumerate(other_agent_ids): full_actions[agent_id] = other_actions[j]
                obs, reward, done, info = self.env.step(full_actions)
                ep_reward += reward[self.training_agent_id]
            
            if info.get('result') == constants.Result.Win and self.training_agent_id in info.get('winners', []):
                wins += 1
            total_return += ep_reward
            print(f"Evaluation Episode {i+1}: Return={ep_reward:.2f}, Result={info.get('result')}")
            self.logger.log_eval_episode({
                "eval_run": "final",
                "episode_num": i+1,
                "result": info.get('result').name,
                "ep_return": ep_reward
            })        
        return wins / num_episodes, total_return / num_episodes

    def evaluate(self):
        print("Starting PPO evaluation...")
        eval_config = self.config.get('evaluation', {})
        model_path = eval_config.get('model_path')
        if not model_path or not os.path.exists(model_path):
            print(f"Error: Evaluation model_path not found: {model_path}"); return
        self.policy.load(model_path)
        
        win_rate, avg_return = self.run_evaluation_episodes(eval_config.get('num_episodes', 10), render=True)
        
        self.logger.log_evaluation({
            "epoch": "final", "total_steps": self.total_steps,
            "win_rate": win_rate, "avg_return": avg_return
        })
        
        print(f"\nEvaluation finished. Win Rate: {win_rate:.2%}, Average Return: {avg_return:.2f}")
        self.env.close()
