import os
import argparse
import yaml
import optuna
from copy import deepcopy
import gc

from my_agents.ppo_agent import PPOAgent

BASE_CONFIG = {
    'env': {
        'scenario': 'PommeFFACompetition-v0'
    },
    'policy': {
        'ppo': {
            'hidden_sizes': [512],
            'lr': 0.00025,
            'gamma': 0.99,
            'clip_ratio': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'update_epochs': 10,
            'batch_size': 256
        }
    },
    'trainer': {
        'steps_per_epoch': 2048,
        'epochs': 30,
        'log_dir': 'logs/optuna_ppo',
        'save_dir': 'models/optuna_ppo',
        'save_name': 'tuned_ppo_agent.pt',
        'save_freq': 100,
        'eval_freq': 100
    }
}

def objective(trial: optuna.Trial):
    """
    The objective function for Optuna. A "trial" consists of:
    1. Suggesting a set of hyperparameters.
    2. Running a short training session with them against SimpleAgents.
    3. Returning the final win rate.
    """
    trial_config = deepcopy(BASE_CONFIG)
    ppo_config = trial_config['policy']['ppo']

    ppo_config['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    ppo_config['gamma'] = trial.suggest_float('gamma', 0.98, 0.999)
    ppo_config['entropy_coef'] = trial.suggest_float('entropy_coef', 1e-3, 0.05, log=True)
    ppo_config['clip_ratio'] = trial.suggest_float('clip_ratio', 0.1, 0.3)
    ppo_config['value_coef'] = trial.suggest_float('value_coef', 0.3, 0.7)
    ppo_config['update_epochs'] = trial.suggest_int('update_epochs', 5, 20)
    layer_size = trial.suggest_categorical('layer_size', [128, 256, 512])
    ppo_config['hidden_sizes'] = [layer_size]

    print(f"\n--- Starting Trial {trial.number}: {trial.params} ---")

    agent = None
    try:
        agent = PPOAgent(config=trial_config)
        win_rate = agent.train(is_tuning=True)
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        if agent and hasattr(agent, 'env'):
            agent.env.close()
        del agent
        gc.collect()

    print(f"--- Trial {trial.number} Finished | Win Rate: {win_rate:.4f} ---")
    
    return win_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=50, help='Number of optimization trials.')
    parser.add_argument('--study-name', type=str, default='ppo_pommerman_study', help='Name for the Optuna study.')
    parser.add_argument('--storage-db', type=str, default='sqlite:///trials/ppo_study.db', help='Optuna storage database URL.')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.storage_db.replace("sqlite:///", "")), exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage_db,
        direction='maximize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )

    study.optimize(objective, n_trials=args.n_trials)

    print("\n--- Optimization Finished ---")
    
    results_df = study.trials_dataframe()
    csv_path = os.path.join(os.path.dirname(args.storage_db.replace("sqlite:///", "")), f"{args.study_name}_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Full study results saved to {csv_path}")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("\nNo trials completed successfully. Cannot determine best parameters.")
    else:
        best_trial = study.best_trial
        print(f"\nBest trial value (win rate): {best_trial.value:.4f}")
        print("Best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")

        best_params_config = deepcopy(BASE_CONFIG)
        ppo_params = best_params_config['policy']['ppo']
        for key, value in best_trial.params.items():
            if key in ppo_params:
                ppo_params[key] = value
        
        layer_size = best_trial.params['layer_size']
        ppo_params['hidden_sizes'] = [layer_size]

        yaml_path = os.path.join(os.path.dirname(args.storage_db.replace("sqlite:///", "")), "best_ppo_params.yml")
        with open(yaml_path, 'w') as f:
            yaml.dump(best_params_config, f, default_flow_style=False)
        
        print(f"\nBest hyperparameters saved to {yaml_path}")
