import gc
import os
import argparse
import yaml
import optuna
from copy import deepcopy

from my_agents.mappo_agent import MAPPOAgent

BASE_CONFIG = {
    'env': {
        'scenario': 'PommeFFACompetition-v0'
    },
    'policy': {
        'mappo': {
            'hidden_sizes': [256, 256],
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
        'log_dir': 'logs/optuna_trials',
        'save_dir': 'models/optuna_trials'
    },
    'evaluation': {
        'num_episodes': 50 
    }
}

def objective(trial: optuna.Trial):
    """
    The objective function for Optuna.
    1. Suggests a set of hyperparameters.
    2. Runs a training session against fixed SimpleAgents.
    3. Runs a deterministic evaluation session against SimpleAgents.
    4. Returns the true win rate from the evaluation.
    """
    trial_config = deepcopy(BASE_CONFIG)
    mappo_config = trial_config['policy']['mappo']

    mappo_config['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    mappo_config['gamma'] = trial.suggest_float('gamma', 0.98, 0.999)
    mappo_config['entropy_coef'] = trial.suggest_float('entropy_coef', 1e-3, 0.05, log=True)
    mappo_config['clip_ratio'] = trial.suggest_float('clip_ratio', 0.1, 0.3)
    mappo_config['value_coef'] = trial.suggest_float('value_coef', 0.3, 0.7)
    mappo_config['update_epochs'] = trial.suggest_int('update_epochs', 5, 20)
    layer_size = trial.suggest_categorical('layer_size', [128, 256, 512])
    mappo_config['hidden_sizes'] = [layer_size, layer_size]

    print(f"\n--- Starting Trial {trial.number}: {trial.params} ---")

    try:
        agent = MAPPOAgent(config=trial_config, use_simple_opponents=True)
        agent.train(is_tuning=True)

        win_rate = agent.run_evaluation_episodes(num_episodes=trial_config['evaluation']['num_episodes'])
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()
    
    finally:
        if agent and hasattr(agent, 'env'):
            agent.env.close()
        del agent
        gc.collect()

    print(f"--- Trial {trial.number} Finished | True Win Rate: {win_rate:.4f} ---")
    
    return win_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=100, help='Number of optimization trials.')
    parser.add_argument('--study-name', type=str, default='mappo_pommerman_study_v2', help='Name for the Optuna study.')
    parser.add_argument('--storage-db', type=str, default='sqlite:///trials/mappo_study_v2.db', help='Optuna storage URL.')
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


    best_trial = study.best_trial
    print(f"\nBest trial value (win rate): {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    best_params_config = deepcopy(BASE_CONFIG)
    best_params_config['policy']['mappo']['lr'] = best_trial.params['lr']
    best_params_config['policy']['mappo']['gamma'] = best_trial.params['gamma']
    best_params_config['policy']['mappo']['entropy_coef'] = best_trial.params['entropy_coef']
    best_params_config['policy']['mappo']['clip_ratio'] = best_trial.params['clip_ratio']
    best_params_config['policy']['mappo']['value_coef'] = best_trial.params['value_coef']
    best_params_config['policy']['mappo']['update_epochs'] = best_trial.params['update_epochs']
    layer_size = best_trial.params['layer_size']
    best_params_config['policy']['mappo']['hidden_sizes'] = [layer_size, layer_size]

    yaml_path = os.path.join(os.path.dirname(args.storage_db.replace("sqlite:///", "")), "best_mappo_params.yml")
    with open(yaml_path, 'w') as f:
        yaml.dump(best_params_config, f, default_flow_style=False)
    
    print(f"\nBest hyperparameters saved to {yaml_path}")
    print("You can now use this file for a full training run with train.py")
