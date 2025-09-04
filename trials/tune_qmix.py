import os
import argparse
import yaml
import optuna
from copy import deepcopy
import gc

from my_agents.qmix_agent import QMIXAgent

BASE_CONFIG = {
    'env': {
        'scenario': 'PommeFFACompetition-v0'
    },
    'policy': {
        'qmix': {
            'agent_hidden_sizes': [64, 64],
            'mixer_hidden_sizes': [64],
            'lr': 0.0005,
            'gamma': 0.99,
            'buffer_size': 5000,
            'batch_size': 32,
            'target_update_freq': 200,
            'epsilon_start': 1.0,
            'epsilon_finish': 0.05,
            'epsilon_anneal_time': 50000
        }
    },
    'trainer': {
        'tuning_episodes': 150,
        'log_dir': 'logs/optuna_qmix',
        'save_dir': 'models/optuna_qmix'
    }
}

def objective(trial: optuna.Trial):
    """
    Objective function for QMIX tuning.
    1. Suggests hyperparameters.
    2. Runs a short self-play training session.
    3. Returns the final average episode return as the metric to maximize.
    """
    trial_config = deepcopy(BASE_CONFIG)
    qmix_config = trial_config['policy']['qmix']

    qmix_config['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    qmix_config['gamma'] = trial.suggest_float('gamma', 0.95, 0.999)
    qmix_config['target_update_freq'] = trial.suggest_int('target_update_freq', 100, 1000)
    layer_size = trial.suggest_categorical('layer_size', [32, 64, 128])
    qmix_config['agent_hidden_sizes'] = [layer_size, layer_size]
    qmix_config['mixer_hidden_sizes'] = [layer_size]

    print(f"\n--- Starting Trial {trial.number}: {trial.params} ---")

    agent = None
    try:
        agent = QMIXAgent(config=trial_config)
        avg_return = agent.train(is_tuning=True)

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        if agent and hasattr(agent, 'env'):
            agent.env.close()
        del agent
        gc.collect()

    print(f"--- Trial {trial.number} Finished | Avg Return: {avg_return:.4f} ---")
    return avg_return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=100, help='Number of optimization trials.')
    parser.add_argument('--study-name', type=str, default='qmix_pommerman_study', help='Name for the Optuna study.')
    parser.add_argument('--storage-db', type=str, default='sqlite:///trials/qmix_study.db', help='Optuna storage URL.')
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
        print(f"\nBest trial value (avg return): {best_trial.value:.4f}")
        print("Best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")

        best_params_config = deepcopy(BASE_CONFIG)
        qmix_params = best_params_config['policy']['qmix']
        qmix_params['lr'] = best_trial.params['lr']
        qmix_params['gamma'] = best_trial.params['gamma']
        qmix_params['target_update_freq'] = best_trial.params['target_update_freq']
        layer_size = best_trial.params['layer_size']
        qmix_params['agent_hidden_sizes'] = [layer_size, layer_size]
        qmix_params['mixer_hidden_sizes'] = [layer_size]

        yaml_path = os.path.join(os.path.dirname(args.storage_db.replace("sqlite:///", "")), "best_qmix_params.yml")
        with open(yaml_path, 'w') as f:
            yaml.dump(best_params_config, f, default_flow_style=False)
        
        print(f"\nBest hyperparameters saved to {yaml_path}")
