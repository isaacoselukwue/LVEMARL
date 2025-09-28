import os
import time
import yaml
import numpy as np
import torch
from pommerman import make, agents, constants
from collections import deque

from policies.qmix import QMIX, EpisodeReplayBuffer

class QMIXAgent(agents.BaseAgent):
    """
    An agent orchestrator that uses a shared QMIX policy for self-play.
    This version is architected for robust training and Optuna tuning.
    """
    def __init__(self, config_path=None, config=None, **kwargs):
        shared_policy = kwargs.pop('_shared_policy', None)
        featurizer = kwargs.pop('_featurizer', None)
        super(QMIXAgent, self).__init__(**kwargs)

        if config:
            self.config = config
        elif config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            if not shared_policy:
                raise ValueError("Either 'config_path' or 'config' must be provided.")
        
        if shared_policy and featurizer:
            self.policy = shared_policy
            self.featurizer = featurizer
            return

        env_config = self.config['env']
        qmix_config = self.config['policy']['qmix']
        
        temp_agents = [agents.SimpleAgent() for _ in range(4)]
        temp_env = make(env_config['scenario'], temp_agents)
        self.num_agents = len(temp_env._agents)
        self.featurizer = temp_env.featurize
        
        featurized_obs = self.featurizer(temp_env.reset()[0])
        obs_dim = featurized_obs.shape[0]
        state_dim = obs_dim * self.num_agents
        self.action_space = temp_env.action_space
        temp_env.close()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = QMIX(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=self.action_space.n,
            num_agents=self.num_agents,
            hidden_sizes=tuple(qmix_config['agent_hidden_sizes']),
            mixer_hidden_size=qmix_config['mixer_hidden_sizes'][0],
            lr=qmix_config['lr'],
            gamma=qmix_config['gamma'],
            device=self.device
        )

        print("Initializing QMIX agent for self-play.")
        self.agent_list = [self] + [
            QMIXAgent(config={}, _shared_policy=self.policy, _featurizer=self.featurizer) 
            for _ in range(self.num_agents - 1)
        ]
        self.num_learning_agents = self.num_agents
        self.env = make(env_config['scenario'], self.agent_list)

    def act(self, obs, action_space):
        self.policy.reset_hidden_states(batch_size=1)
        featurized_obs = self.featurizer(obs)
        actions = self.policy.choose_actions(featurized_obs[np.newaxis, :], epsilon=0.0, deterministic=True)
        return actions.item()

    def train(self, is_tuning=False):
        print("Starting QMIX training...")
        trainer_config = self.config['trainer']
        qmix_config = self.config['policy']['qmix']
        
        buffer = EpisodeReplayBuffer(
            capacity=qmix_config['buffer_size'],
            max_episode_len=trainer_config.get('max_episode_len', 800)
        )
        
        epsilon = qmix_config['epsilon_start']
        epsilon_decay_rate = (qmix_config['epsilon_start'] - qmix_config['epsilon_finish']) / qmix_config['epsilon_anneal_time']
        
        recent_returns = deque(maxlen=50)
        timestep = 0
        start_time = time.time()
        
        num_episodes = trainer_config.get('episodes', 5000)
        if is_tuning:
            num_episodes = trainer_config.get('tuning_episodes', 150)

        for episode in range(num_episodes):
            episode_buffer = {'obs': [], 'actions': [], 'rewards': [], 'dones': []}
            obs = self.env.reset()
            self.policy.reset_hidden_states(batch_size=self.num_learning_agents)
            done = False
            ep_returns = np.zeros(self.num_learning_agents)

            while not done:
                featurized_obs = np.array([self.featurizer(o) for o in obs])
                agent_actions = self.policy.choose_actions(featurized_obs, epsilon)
                
                next_obs, rewards, done, info = self.env.step(agent_actions)
                
                shaped_rewards = np.array(rewards, dtype=np.float32)

                # --- Reward Shaping Logic ---
                if done and info.get('result') == constants.Result.Win:
                    winners = info.get('winners', [])
                    for winner_id in winners:
                        shaped_rewards[winner_id] += 1.0 # Win bonus
                else: 
                    for i in range(self.num_agents):
                        if obs[i]['blast_strength'] < next_obs[i]['blast_strength'] or \
                           obs[i]['can_kick'] < next_obs[i]['can_kick'] or \
                           obs[i]['ammo'] < next_obs[i]['ammo']:
                            shaped_rewards[i] += 0.1 # Power-up bonus
                        if agent_actions[i] == constants.Action.Bomb.value: shaped_rewards[i] += 0.01 # Bombing bonus
                        if agent_actions[i] == constants.Action.Stop.value: shaped_rewards[i] -= 0.001 # Inaction penalty

                # NOTE: Using the mean reward is a simplification for FFA. I did this just to check the compatibility
                # QMIX is fundamentally designed for cooperative tasks with a true team reward.
                team_reward = shaped_rewards.mean()

                episode_buffer['obs'].append(featurized_obs)
                episode_buffer['actions'].append(agent_actions)
                episode_buffer['rewards'].append([team_reward])
                episode_buffer['dones'].append([done])

                obs = next_obs
                ep_returns += np.array(rewards)
                timestep += 1
                epsilon = max(qmix_config['epsilon_finish'], epsilon - epsilon_decay_rate)
                
                if len(buffer) >= qmix_config['batch_size']:
                    batch, max_len = buffer.sample(qmix_config['batch_size'])
                    self.policy.update(batch, max_len)

                if timestep % qmix_config['target_update_freq'] == 0:
                    self.policy.update_targets()
            
            buffer.store_episode(episode_buffer)
            recent_returns.append(ep_returns.mean())

            if not is_tuning and (episode + 1) % 10 == 0:
                print(f"Eps {episode+1} | Timestep {timestep}: Avg Return={np.mean(recent_returns):.2f}, Epsilon={epsilon:.3f}, Time={time.time() - start_time:.1f}s")
        
        print("Training finished.")
        
        if is_tuning:
            return np.mean(recent_returns) if recent_returns else -10.0

    def evaluate(self):
        eval_config = self.config.get('evaluation', {})
        model_path = eval_config.get('model_path')
        if not model_path or not os.path.exists(model_path):
            print(f"Error: model_path not specified or not found: {model_path}")
            return
        
        self.policy.load(model_path)
        
        eval_agent = QMIXAgent(config={}, _shared_policy=self.policy, _featurizer=self.featurizer)
        agent_list = [eval_agent] + [agents.SimpleAgent() for _ in range(self.num_agents - 1)]
        eval_env = make(self.config['env']['scenario'], agent_list)
        
        wins = 0
        print(f"Running {eval_config.get('num_episodes', 10)} evaluation episodes...")
        for i in range(eval_config.get('num_episodes', 10)):
            obs = eval_env.reset()
            self.policy.reset_hidden_states(batch_size=1)
            done = False
            ep_reward = 0
            while not done:
                eval_env.render()
                time.sleep(0.1)

                agent_obs = self.featurizer(obs[0])
                action = self.policy.choose_actions(agent_obs[np.newaxis, :], epsilon=0.0, deterministic=True)[0]
                
                simple_agent_actions = eval_env.act(obs)
                all_actions = [0] * self.num_agents
                all_actions[0] = action
                for agent_id in range(1, self.num_agents):
                    all_actions[agent_id] = simple_agent_actions[agent_id-1]

                obs, rewards, done, info = eval_env.step(all_actions)
                ep_reward += rewards[0]
            
            if info.get('result') == constants.Result.Win and 0 in info.get('winners', []):
                wins += 1
            
            print(f"Evaluation Episode {i+1}: Return={ep_reward:.2f}, Result={info.get('result')}")

        print(f"\nEvaluation finished. Win Rate: {wins / eval_config.get('num_episodes', 10):.2%}")
        eval_env.close()
