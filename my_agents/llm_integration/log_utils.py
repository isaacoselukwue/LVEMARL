import csv
import os
import time
import yaml

class CsvLogger:
    """A utility class to handle logging experiment data to CSV files."""
    def __init__(self, log_dir, config, llm_assisted=False):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.llm_assisted = llm_assisted

        self.ep_log_path = os.path.join(self.log_dir, "training_episodes.csv")
        self.epoch_log_path = os.path.join(self.log_dir, "epoch_summary.csv")
        self.eval_log_path = os.path.join(self.log_dir, "evaluation_runs.csv")
        self.meta_log_path = os.path.join(self.log_dir, "metadata.csv")
        self.eval_episode_log_path = os.path.join(self.log_dir, "evaluation_episodes.csv")

        self._setup_files()
        self._log_metadata(config)

    def _setup_files(self):
        """Creates CSV files and writes headers if they don't exist."""
        if not os.path.exists(self.ep_log_path):
            with open(self.ep_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "epoch", "episode_num", "total_steps", 
                    "ep_return", "ep_length", "result", "policy_entropy", 
                    "action_prob_gap", "llm_assisted"
                ])
        
        if not os.path.exists(self.epoch_log_path):
            with open(self.epoch_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "epoch", "total_steps", "loss", 
                    "kl_divergence", "llm_assisted"
                ])

        if not os.path.exists(self.eval_log_path):
            with open(self.eval_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "epoch", "total_steps", "win_rate", "avg_return"])

        if not os.path.exists(self.eval_episode_log_path):
            with open(self.eval_episode_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp",  "eval_run", "episode_num", "result", "ep_return"])

    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flattens a nested dictionary for metadata logging."""
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _log_metadata(self, config):
        """Logs the flattened experiment configuration."""
        flat_config = self._flatten_dict(config)
        with open(self.meta_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(flat_config.keys())
            writer.writerow(flat_config.values())

    def log_training_episode(self, data):
        """Logs a single training episode's stats."""
        with open(self.ep_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), data['epoch'], data['episode_num'], data['total_steps'],
                data['ep_return'], data['ep_length'], data['result'],
                data['policy_entropy'], data['action_prob_gap'],
                self.llm_assisted
            ])

    def log_epoch(self, data):
        """Logs the summary of a training epoch."""
        with open(self.epoch_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), data['epoch'], data['total_steps'],
                data['loss'], data['kl_divergence'],
                self.llm_assisted
            ])

    def log_evaluation(self, data):
        """Logs the results of an evaluation run."""
        with open(self.eval_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), data['epoch'], data['total_steps'],
                data['win_rate'], data['avg_return']
            ])

    def log_eval_episode(self, data):
        """Logs the results of an evaluation episode."""
        with open(self.eval_episode_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), data['eval_run'], data['episode_num'],
                data['result'], data['ep_return']
            ])
