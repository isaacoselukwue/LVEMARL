import gc
import os
import argparse
import yaml
import optuna
from copy import deepcopy

from my_agents.happo_ensemble_agent import HAPPOEnsembleAgent

BASE_CONFIG = {
    'env': { 'scenario': 'PommeFFACompetition-v0' },
    'policy': {
        'happo_ensemble': {
            'ensemble_size': 5, 'hidden_sizes': [256, 256], 'lr': 0.0003, 
            'gamma': 0.99, 'clip_ratio': 0.2, 'value_coef': 1.0, 
            'entropy_coef': 0.01, 'update_epochs': 10, 'batch_size': 256, 'tau': 0.005
        }
    },
    'trainer': {
        'steps_per_epoch': 2048, 'epochs': 30,
        'log_dir': 'logs/optuna_happo_ensemble', 'save_dir': 'models/optuna_happo_ensemble'
    },
    'evaluation': { 'num_episodes': 50 }
}

def objective(trial: optuna.Trial):
    """Optuna objective function for HAPPO Ensemble."""
    trial_config = deepcopy(BASE_CONFIG)
    happo_config = trial_config['policy']['happo_ensemble']

    happo_config['ensemble_size'] = trial.suggest_int('ensemble_size', 3, 5)
    happo_config['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    happo_config['gamma'] = trial.suggest_float('gamma', 0.98, 0.999)
    happo_config['entropy_coef'] = trial.suggest_float('entropy_coef', 1e-3, 0.05, log=True)
    happo_config['clip_ratio'] = trial.suggest_float('clip_ratio', 0.1, 0.3)
    happo_config['tau'] = trial.suggest_float('tau', 1e-3, 1e-1, log=True)
    layer_size = trial.suggest_categorical('layer_size', [128, 256])
    happo_config['hidden_sizes'] = [layer_size, layer_size]

    print(f"\n--- Starting Trial {trial.number}: {trial.params} ---")
    agent = None
    try:
        agent = HAPPOEnsembleAgent(config=trial_config, use_simple_opponents=True)
        agent.train(is_tuning=True)
        win_rate = agent.run_evaluation_episodes(num_episodes=trial_config['evaluation']['num_episodes'])
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        if agent and hasattr(agent, 'env'): agent.env.close()
        del agent; gc.collect()

    print(f"--- Trial {trial.number} Finished | Win Rate: {win_rate:.4f} ---")
    return win_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--study-name', type=str, default='happo_ensemble_study')
    parser.add_argument('--storage-db', type=str, default='sqlite:///trials/happo_ensemble_study.db')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.storage_db.replace("sqlite:///", "")), exist_ok=True)
    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage_db,
        direction='maximize', load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\n--- Optimization Finished ---")
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        print("\nNo trials completed successfully.")
    else:
        best_trial = study.best_trial
        print(f"\nBest trial win rate: {best_trial.value:.4f}")
        print("Best hyperparameters:")
        for key, value in best_trial.params.items(): print(f"  {key}: {value}")
        best_params_config = deepcopy(BASE_CONFIG)
        best_params_config['policy']['happo_ensemble'].update(best_trial.params)
        if 'layer_size' in best_trial.params:
            best_params_config['policy']['happo_ensemble']['hidden_sizes'] = [best_trial.params['layer_size']] * 2
            del best_params_config['policy']['happo_ensemble']['layer_size']
        yaml_path = os.path.join(os.path.dirname(args.storage_db.replace("sqlite:///", "")), "best_happo_ensemble_params.yml")
        with open(yaml_path, 'w') as f: yaml.dump(best_params_config, f, default_flow_style=False)
        print(f"\nBest hyperparameters saved to {yaml_path}")
