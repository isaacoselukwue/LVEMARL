import os
import time
import yaml
import numpy as np
import torch
from gym import spaces
from pommerman import make, agents, constants
from collections import deque
import scipy.stats

from policies.mappo import MAPPO, MAPPOBuffer
from .llm_integration.log_utils import CsvLogger

class MAPPOAgent(agents.BaseAgent):
    """
    An agent orchestrator that uses a shared MAPPO policy, now with comprehensive logging.
    """
    def __init__(self, config_path=None, config=None, use_simple_opponents=False, **kwargs):
        shared_policy = kwargs.pop('_shared_policy', None)
        featurizer = kwargs.pop('_featurizer', None)
        super(MAPPOAgent, self).__init__(**kwargs)

        if config: self.config = config
        elif config_path:
            with open(config_path, 'r') as f: self.config = yaml.safe_load(f)
        else:
            if not shared_policy: raise ValueError("Config must be provided.")
        
        if shared_policy and featurizer:
            self.policy = shared_policy
            self.featurizer = featurizer
            return

        env_config = self.config['env']
        mappo_config = self.config['policy']['mappo']
        
        temp_env = make(env_config['scenario'], [agents.SimpleAgent() for _ in range(4)])
        self.featurizer = temp_env.featurize
        obs_dim = self.featurizer(temp_env.reset()[0]).shape[0]
        action_space = temp_env.action_space
        temp_env.close()
        global_state_dim = obs_dim * 4 

        self.policy = MAPPO(
            obs_space=spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32),
            action_space=action_space, global_state_dim=global_state_dim,
            hidden_sizes=tuple(mappo_config['hidden_sizes']), lr=mappo_config['lr'],
            gamma=mappo_config['gamma'], clip_ratio=mappo_config['clip_ratio'],
            value_coef=mappo_config['value_coef'], entropy_coef=mappo_config['entropy_coef'],
            update_epochs=mappo_config['update_epochs'], batch_size=mappo_config['batch_size']
        )
        
        self.use_simple_opponents = use_simple_opponents
        if self.use_simple_opponents:
            self.agent_list = [self, agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()]
            self.training_agent_id = 0; self.num_learning_agents = 1
        else: 
            self.agent_list = [self] + [MAPPOAgent(config={}, _shared_policy=self.policy, _featurizer=self.featurizer) for _ in range(3)]
            self.training_agent_id = None; self.num_learning_agents = 4
        
        self.env = make(env_config['scenario'], self.agent_list)
        if self.training_agent_id is not None: self.env.set_training_agent(self.training_agent_id)
        
        self.logger = CsvLogger(log_dir=self.config['trainer']['log_dir'], config=self.config, llm_assisted=False)
        self.total_steps = 0; self.episode_num = 0
        self.state_history = [deque(maxlen=100) for _ in range(4)]

    def act(self, obs, action_space):
        featurized_obs = self.featurizer(obs)
        action_mask = np.array(obs.get('legal_actions', np.ones(action_space.n)))
        action, _, _ = self.policy.get_action(featurized_obs, action_mask=action_mask, deterministic=True)
        return action.item()

    def train(self, is_tuning=False):
        if not is_tuning: print("Starting MAPPO training...")
        trainer_config = self.config['trainer']
        mappo_config = self.config['policy']['mappo']
        buffer = MAPPOBuffer(
            obs_shape=(self.policy.policy.actor_mlp[0].in_features,),
            global_state_shape=(self.policy.policy.critic_mlp[0].in_features,),
            action_dim=self.env.action_space.n, buffer_size=trainer_config['steps_per_epoch'],
            num_agents=self.num_learning_agents, gamma=mappo_config['gamma'], gae_lambda=self.policy.gae_lambda
        )
        obs = self.env.reset()
        ep_returns, ep_len = np.zeros(self.num_learning_agents), 0
        ep_entropies, ep_prob_gaps = np.zeros(self.num_learning_agents), np.zeros(self.num_learning_agents)
        recent_wins = deque(maxlen=50)

        for epoch in range(trainer_config['epochs']):
            for t in range(trainer_config['steps_per_epoch']):
                self.total_steps += self.num_learning_agents
                featurized_obs = np.array([self.featurizer(o) for o in obs])
                global_state = featurized_obs.flatten()
                value = self.policy.get_value(global_state)
                
                learning_agent_ids = [self.training_agent_id] if self.use_simple_opponents else range(self.num_learning_agents)
                
                actions, log_probs, action_masks, action_probs_list = [], [], [], []
                for i, agent_id in enumerate(learning_agent_ids):
                    action_mask_i = np.array(obs[agent_id].get('legal_actions', np.ones(self.env.action_space.n)))
                    action_i, log_prob_i, action_probs_i = self.policy.get_action(featurized_obs[i], action_mask=action_mask_i)
                    actions.append(action_i.item()); log_probs.append(log_prob_i.item())
                    action_masks.append(action_mask_i); action_probs_list.append(action_probs_i)
                
                for i in range(self.num_learning_agents):
                    ep_entropies[i] += scipy.stats.entropy(action_probs_list[i])
                    sorted_probs = np.sort(action_probs_list[i])[::-1]
                    ep_prob_gaps[i] += sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

                if self.use_simple_opponents:
                    other_actions = self.env.act(obs)
                    all_actions = [0] * len(self.agent_list)
                    all_actions[self.training_agent_id] = actions[0]
                    other_agent_ids = [j for j in range(len(self.agent_list)) if j != self.training_agent_id]
                    for j, agent_id in enumerate(other_agent_ids): all_actions[agent_id] = other_actions[j]
                else:
                    all_actions = actions

                next_obs, rewards, done, info = self.env.step(all_actions)
                
                shaped_rewards = np.array([rewards[i] for i in learning_agent_ids], dtype=np.float32)

                buffer.store(featurized_obs[learning_agent_ids], global_state, np.array(actions), shaped_rewards, 
                             value, np.array(log_probs), [done]*self.num_learning_agents, np.array(action_masks))

                obs = next_obs
                ep_returns += np.array([rewards[i] for i in learning_agent_ids])
                ep_len += 1

                if done:
                    self.episode_num += 1
                    is_win = 1.0 if info.get('result') == constants.Result.Win and (not self.use_simple_opponents or self.training_agent_id in info.get('winners', [])) else 0.0
                    recent_wins.append(is_win)

                    if not is_tuning:
                        print(f"Epoch {epoch}, Ep {self.episode_num}: Avg Ret={ep_returns.mean():.2f}, Len={ep_len}, Res={info.get('result')}")
                        self.logger.log_training_episode({
                            "epoch": epoch, "episode_num": self.episode_num, "total_steps": self.total_steps,
                            "ep_return": ep_returns.mean(), "ep_length": ep_len, "result": info.get('result').name,
                            "policy_entropy": ep_entropies.mean() / ep_len if ep_len > 0 else 0,
                            "action_prob_gap": ep_prob_gaps.mean() / ep_len if ep_len > 0 else 0
                        })
                    obs = self.env.reset()
                    ep_returns, ep_len = np.zeros(self.num_learning_agents), 0
                    ep_entropies, ep_prob_gaps = np.zeros(self.num_learning_agents), np.zeros(self.num_learning_agents)

            last_val = self.policy.get_value(np.array([self.featurizer(o) for o in obs]).flatten())
            buffer.finish_path(last_val)
            data = buffer.get_data(); logs = self.policy.update(data)
            if not is_tuning:
                self.logger.log_epoch({
                    "epoch": epoch + 1, "total_steps": self.total_steps,
                    "loss": logs['total_loss'], "kl_divergence": logs['kl_divergence']
                })
                print(f"Epoch {epoch+1}/{trainer_config['epochs']} | Loss: {logs['total_loss']:.3f} | KL: {logs['kl_divergence']:.4f}")
                if (epoch + 1) % trainer_config.get('save_freq', 20) == 0:
                    self.policy.save(os.path.join(trainer_config['save_dir'], trainer_config.get('save_name', 'mappo_agent.pt')))
                    
        if not is_tuning: print("Training finished.")
        self.env.close()
        if is_tuning: return sum(recent_wins) / len(recent_wins) if recent_wins else 0.0

    def run_evaluation_episodes(self, num_episodes, render=False):
        print(f"Running {num_episodes} evaluation episodes...")
        wins, total_return = 0, 0
        for i in range(num_episodes):
            obs = self.env.reset()
            done = False
            ep_rewards = np.zeros(self.num_learning_agents)
            while not done:
                if render: self.env.render(); time.sleep(0.1)
                if self.use_simple_opponents:
                    agent_obs = self.featurizer(obs[self.training_agent_id])
                    action_mask = np.array(obs[self.training_agent_id].get('legal_actions', np.ones(self.env.action_space.n)))
                    action, _, _ = self.policy.get_action(agent_obs, action_mask=action_mask, deterministic=True)
                    other_actions = self.env.act(obs)
                    all_actions = [0] * len(self.agent_list)
                    all_actions[self.training_agent_id] = action.item()
                    other_agent_ids = [j for j in range(len(self.agent_list)) if j != self.training_agent_id]
                    for j, agent_id in enumerate(other_agent_ids): all_actions[agent_id] = other_actions[j]
                else:
                    all_actions = []
                    for j in range(self.num_learning_agents):
                        agent_obs = self.featurizer(obs[j])
                        action_mask = np.array(obs[j].get('legal_actions', np.ones(self.env.action_space.n)))
                        action, _, _ = self.policy.get_action(agent_obs, action_mask=action_mask, deterministic=True)
                        all_actions.append(action.item())
                obs, rewards, done, info = self.env.step(all_actions)
                learning_agent_ids = [self.training_agent_id] if self.use_simple_opponents else range(self.num_learning_agents)
                ep_rewards += np.array([rewards[k] for k in learning_agent_ids])
            if info.get('result') == constants.Result.Win:
                if not self.use_simple_opponents or self.training_agent_id in info.get('winners', []): wins += 1
            total_return += ep_rewards.mean()
            print(f"Evaluation Episode {i+1}: Return={ep_rewards.mean():.2f}, Result={info.get('result')}")
            self.logger.log_eval_episode({
                "eval_run": "final",
                "episode_num": i+1,
                "result": info.get('result').name,
                "ep_return": ep_rewards.mean()
            })
        return wins / num_episodes, total_return / num_episodes

    def evaluate(self):
        eval_config = self.config.get('evaluation', {})
        model_path = eval_config.get('model_path')
        if not model_path or not os.path.exists(model_path):
            print(f"Error: model_path not found: {model_path}"); return
        self.policy.load(model_path)
        eval_agent = MAPPOAgent(config={}, _shared_policy=self.policy, _featurizer=self.featurizer)
        eval_agent.logger = self.logger
        self.use_simple_opponents = True
        self.agent_list = [eval_agent, agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()]
        self.env = make(self.config['env']['scenario'], self.agent_list)
        self.training_agent_id = 0
        self.env.set_training_agent(self.training_agent_id)
        win_rate, avg_return = self.run_evaluation_episodes(eval_config.get('num_episodes', 10), render=True)
        self.logger.log_evaluation({
            "epoch": "final", "total_steps": self.total_steps,
            "win_rate": win_rate, "avg_return": avg_return
        })
        print(f"\nEvaluation finished. Win Rate vs SimpleAgents: {win_rate:.2%}, Avg Return: {avg_return:.2f}")
        self.env.close()
