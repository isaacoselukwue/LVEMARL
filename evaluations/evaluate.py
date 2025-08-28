import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import arviz as az
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random

# --- Configuration ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14

# --- 1. Data Loading ---
def _find_col(df, candidates, default=None):
    """Return first column name from candidates that exists in df, else default."""
    if df is None or df.empty:
        return default
    for c in candidates:
        if c in df.columns:
            return c
    return default
def load_experiment_data(base_dir, experiment_name):
    """
    Loads all relevant CSV files for a single experiment.

    Args:
        base_dir (str): The base directory containing all experiment folders.
        experiment_name (str): The name of the specific experiment folder.

    Returns:
        dict: A dictionary of pandas DataFrames, with keys for each CSV type
              ('epoch_summary', 'evaluation_runs', 'metadata', 'training_episodes').
              Returns None if the directory is not found.
    """
    experiment_path = os.path.join(base_dir, experiment_name)
    if not os.path.isdir(experiment_path):
        print(f"Warning: Directory not found for experiment '{experiment_name}'. Skipping.")
        return None

    data_files = {
        'epoch_summary': 'epoch_summary.csv',
        'evaluation_runs': 'evaluation_runs.csv',
        'metadata': 'metadata.csv',
        'training_episodes': 'training_episodes.csv'
    }

    data = {}
    print(f"Loading data for: {experiment_name}")
    for key, filename in data_files.items():
        file_path = os.path.join(experiment_path, filename)
        try:
            data[key] = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"  - Warning: Could not find '{filename}' in '{experiment_path}'")
            data[key] = pd.DataFrame()
    return data

# --- 2. Single-Experiment Vis Functions ---

def plot_optimizer_learning(epoch_df, save_path=None):
    """Plots Loss and KL Divergence vs. Epoch."""
    if epoch_df.empty or 'loss' not in epoch_df.columns or 'kl_divergence' not in epoch_df.columns:
        print("  - Skipping optimizer plot: missing data.")
        return

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epoch_df['epoch'], epoch_df['loss'], color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('KL Divergence', color='tab:blue')
    ax2.plot(epoch_df['epoch'], epoch_df['kl_divergence'], color='tab:blue', label='KL Divergence')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Optimizer Learning: Loss and KL Divergence vs. Epoch')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_learning_curve(training_df, save_path=None):
    """Plots the learning curve with EMA and a 95% bootstrap CI."""
    if training_df.empty or 'ep_return' not in training_df.columns:
        print("  - Skipping learning curve plot: missing data.")
        return

    training_df['return_ema'] = training_df['ep_return'].ewm(span=100, adjust=False).mean()

    n_bootstraps = 500
    bootstrapped_means = np.zeros((n_bootstraps, len(training_df)))
    returns_array = training_df['ep_return'].values
    for i in range(n_bootstraps):
        resampled_indices = np.random.randint(0, len(returns_array), len(returns_array))
        resampled_returns = returns_array[resampled_indices]
        bootstrapped_means[i, :] = pd.Series(resampled_returns).expanding().mean().values

    ci_lower = np.percentile(bootstrapped_means, 2.5, axis=0)
    ci_upper = np.percentile(bootstrapped_means, 97.5, axis=0)

    plt.figure()
    plt.plot(training_df['episode_num'], training_df['ep_return'], 'k-', alpha=0.1, label='Raw Episode Return')
    plt.plot(training_df['episode_num'], training_df['return_ema'], 'r-', linewidth=2, label='100-Episode EMA')
    plt.fill_between(training_df['episode_num'], ci_lower, ci_upper, color='blue', alpha=0.2, label='95% Bootstrap CI')
    plt.title('Performance Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_outcome_variability(training_df, save_path=None):
    """Creates a violin plot of returns at different epochs."""
    if training_df.empty or 'ep_return' not in training_df.columns or 'epoch' not in training_df.columns:
        print("  - Skipping outcome variability plot: missing data.")
        return

    unique_epochs = training_df['epoch'].unique()
    if len(unique_epochs) > 10:
        selected_epochs = unique_epochs[::len(unique_epochs) // 10]
    else:
        selected_epochs = unique_epochs

    violin_df = training_df[training_df['epoch'].isin(selected_epochs)]

    plt.figure()
    sns.violinplot(x='epoch', y='ep_return', data=violin_df, inner='quartile', cut=0)
    plt.title('Distribution of Episode Returns Across Training')
    plt.xlabel('Epoch')
    plt.ylabel('Episode Return')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_win_tie_loss(training_df, save_path=None):
    """Creates a stacked bar plot of Win/Tie/Loss counts per epoch."""
    if training_df.empty or 'result' not in training_df.columns or 'epoch' not in training_df.columns:
        print("  - Skipping win/tie/loss plot: missing data.")
        return

    bin_size = 50
    training_df['epoch_bin'] = (training_df['epoch'] // bin_size) * bin_size
    result_counts = training_df.groupby(['epoch_bin', 'result']).size().unstack(fill_value=0)
    for outcome in ['Win', 'Tie', 'Loss']:
        if outcome not in result_counts.columns:
            result_counts[outcome] = 0
    result_counts = result_counts[['Win', 'Tie', 'Loss']]

    result_counts.plot(kind='bar', stacked=True, figsize=(14, 7),
                      color=['#2ca02c', '#ff7f0e', '#d62728'])
    plt.title('Game Outcomes per Epoch (Binned)')
    plt.xlabel('Epoch (Binned)')
    plt.ylabel('Number of Episodes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_win_speed(training_df, save_path=None):
    """Creates a histogram of episode lengths for winning games."""
    if training_df.empty or 'result' not in training_df.columns or 'ep_length' not in training_df.columns:
        print("  - Skipping win speed plot: missing data.")
        return

    winning_episodes = training_df[training_df['result'] == 'Win']
    if winning_episodes.empty:
        print("  - No winning episodes found to plot win speed.")
        return

    plt.figure()
    sns.histplot(winning_episodes['ep_length'], bins=30, kde=True)
    plt.title('Distribution of Episode Length for Winning Games')
    plt.xlabel('Episode Length (Steps)')
    plt.ylabel('Frequency')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_combined_learning_curves(policy_name, data_dict, save_path=None):
    """Plots learning curves for baseline, ensemble, and LLM variants of a policy."""
    plt.figure(figsize=(15, 8))
    plt.title(f'Learning Curve Comparison for {policy_name.upper()}', fontsize=20)
    
    variants = {
        'Baseline': f'{policy_name}_pommerman',
        'Ensemble': f'{policy_name}_ensemble_pommerman',
        'Ensemble+LLM': f'{policy_name}_ensemble_pommerman_llm'
    }
    
    colors = {'Baseline': 'blue', 'Ensemble': 'green', 'Ensemble+LLM': 'red'}

    for label, exp_name in variants.items():
        if exp_name in data_dict and not data_dict[exp_name]['training_episodes'].empty:
            df = data_dict[exp_name]['training_episodes']
            # Smooth the curve for better visualization
            ema = df['ep_return'].ewm(span=200, adjust=False).mean()
            plt.plot(df['total_steps'], ema, label=label, color=colors[label], linewidth=2.5)

    plt.xlabel('Total Steps')
    plt.ylabel('Exponential Moving Average of Episode Return (span=200)')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_population_comparison(all_data, save_path=None):
    """Creates violin plots comparing return distributions across all models."""
    
    plot_data = []
    for exp_name, data in all_data.items():
        if 'training_episodes' in data and not data['training_episodes'].empty:
            df = data['training_episodes']
            stable_returns = df[df['episode_num'] > df['episode_num'].max() / 2]['ep_return']
            
            if 'ensemble' in exp_name and 'llm' in exp_name:
                model_type = 'Ensemble+LLM'
            elif 'ensemble' in exp_name:
                model_type = 'Ensemble'
            else:
                model_type = 'Baseline'
            
            policy = exp_name.split('_')[0]
            
            for ret in stable_returns:
                plot_data.append({'policy': policy.upper(), 'model_type': model_type, 'return': ret})

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(18, 10))
    sns.violinplot(x='policy', y='return', hue='model_type', data=plot_df,
                   palette='muted', split=True, inner='quartile', cut=0)
    plt.title('Population Comparison of Final Return Distributions', fontsize=20)
    plt.xlabel('Policy')
    plt.ylabel('Episode Return (Second Half of Training)')
    plt.legend(title='Model Type')
    if save_path:
        plt.savefig(save_path)
    plt.show()

# --- 3. Inferential and Comparative Statistics Functions ---

def analyze_win_rate_posterior(eval_df, meta_df):
    """Calculates and prints the Bayesian posterior for the win-rate."""
    if eval_df.empty or meta_df.empty:
        return None, None
    try:
        n = meta_df['evaluation_num_episodes'].iloc[0]
        win_rate = eval_df['win_rate'].iloc[0]
        w = int(win_rate * n)

        alpha_post = 1 + w
        beta_post = 1 + n - w

        map_estimate = w / n
        ci_low, ci_high = stats.beta.interval(0.95, alpha_post, beta_post)

        print("\n--- Bayesian Win-Rate Analysis ---")
        print(f"Evaluation Games (n): {n}, Wins (w): {w}")
        print(f"Posterior Distribution: Beta(α={alpha_post}, β={beta_post})")
        print(f"MAP Win-Rate: {map_estimate:.2%}")
        print(f"95% Credible Interval: ({ci_low:.2%} - {ci_high:.2%})")
        return alpha_post, beta_post
    except (IndexError, KeyError):
        print("  - Could not perform win-rate analysis: missing evaluation data.")
        return None, None

def compare_win_rates(alpha1, beta1, name1, alpha2, beta2, name2):
    """Compares two agents' win-rates using their Beta posteriors."""
    if alpha1 is None or alpha2 is None:
        print("\n--- Skipping Win-Rate Comparison: missing posterior data ---")
        return

    print(f"\n--- Comparing Win-Rates: {name1} vs {name2} ---")
    n_samples = 50000
    samples1 = stats.beta.rvs(alpha1, beta1, size=n_samples)
    samples2 = stats.beta.rvs(alpha2, beta2, size=n_samples)

    delta = samples2 - samples1
    p_better = (delta > 0).mean()

    plt.figure()
    sns.histplot(delta, kde=True)
    plt.axvline(0, color='k', linestyle='--')
    plt.title(f'Posterior Distribution of Win-Rate Difference ({name2} - {name1})')
    plt.xlabel('Difference in Win-Rate')
    plt.ylabel('Density')
    plt.show()

    print(f"Posterior probability that '{name2}' is better than '{name1}': {p_better:.2%}")

def compare_returns_bootstrap(returns1, name1, returns2, name2):
    """Compares two agents' mean returns using a bootstrap test."""
    if returns1.empty or returns2.empty:
        print("\n--- Skipping Return Comparison (Bootstrap): missing return data ---")
        return
    print(f"\n--- Comparing Mean Returns (Bootstrap): {name1} vs {name2} ---")
    n_samples = 20000
    boot_diffs = np.random.choice(returns2, (n_samples, len(returns2))).mean(1) - \
                 np.random.choice(returns1, (n_samples, len(returns1))).mean(1)

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    mean_diff = np.mean(boot_diffs)

    print(f"Mean Return Difference ({name2} - {name1}): {mean_diff:.3f}")
    print(f"95% Bootstrap CI for the Difference: [{ci_low:.3f}, {ci_high:.3f}]")
    if ci_low > 0 or ci_high < 0:
        print("  - Result is statistically significant at p < 0.05 (CI does not contain zero).")
    else:
        print("  - Result is not statistically significant (CI contains zero).")

def compare_returns_ttest(returns1, name1, returns2, name2):
    """Compares mean returns using Welch's t-test."""
    if returns1.empty or returns2.empty:
        print(f"\n--- Comparing Mean Returns (Welch's t-test): {name1} vs {name2} ---")
        t_stat, p_val = stats.ttest_ind(returns2, returns1, equal_var=False, nan_policy='omit')
        print(f"T-statistic: {t_stat:.3f}, P-value: {p_val:.4f}")
        if p_val < 0.05:
            print(f"  - Significant difference found. Mean return of {name2} is likely different from {name1}.")
        else:
            print("  - No significant difference found.")

def compare_returns_mannwhitneyu(returns1, name1, returns2, name2):
    """Compares return distributions using Mann-Whitney U test."""
    if returns1.empty or returns2.empty:
        print(f"\n--- Comparing Return Distributions (Mann-Whitney U): {name1} vs {name2} ---")
        u_stat, p_val = stats.mannwhitneyu(returns2, returns1, alternative='two-sided')
        print(f"U-statistic: {u_stat}, P-value: {p_val:.4f}")
        if p_val < 0.05:
            print(f"  - Significant difference found. The return distributions of {name1} and {name2} are likely different.")
        else:
            print("  - No significant difference found.")

def compare_outcomes_chisquared(results1, name1, results2, name2):
    """Compares Win/Tie/Loss distributions using a Chi-Squared test."""
    if results1.empty or results2.empty:
        print(f"\n--- Comparing Game Outcomes (Chi-Squared): {name1} vs {name2} ---")
        counts1 = results1.value_counts()
        counts2 = results2.value_counts()
        
        contingency_table = pd.DataFrame({'Agent1': counts1, 'Agent2': counts2}).fillna(0)
        print("Contingency Table:")
        print(contingency_table)

        try:
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            print(f"\nChi-Squared: {chi2:.3f}, P-value: {p:.4f}, Degrees of Freedom: {dof}")
            if p < 0.05:
                print("  - Significant difference found. The distribution of outcomes (Win/Tie/Loss) is different between the agents.")
            else:
                print("  - No significant difference found in outcome distributions.")
        except ValueError as e:
            print(f"  - Could not perform Chi-Squared test: {e}")

def perform_anova_and_tukey(all_data):
    """Performs a two-way ANOVA on episode returns and a Tukey HSD post-hoc test."""
    print("\n\n" + "="*50)
    print("  PERFORMING TWO-WAY ANOVA & TUKEY'S HSD TEST  ")
    print("="*50)
    
    anova_data = []
    for exp_name, data in all_data.items():
        if 'training_episodes' in data and not data['training_episodes'].empty:
            df = data['training_episodes']
            stable_returns = df[df['episode_num'] > df['episode_num'].max() / 2]['ep_return']

            if 'ensemble' in exp_name and 'llm' in exp_name:
                model_type = 'EnsembleLLM'
            elif 'ensemble' in exp_name:
                model_type = 'Ensemble'
            else:
                model_type = 'Baseline'
            
            policy = exp_name.split('_')[0]
            
            for ret in stable_returns:
                anova_data.append({'policy': policy, 'model_type': model_type, 'ep_return': ret})
    
    anova_df = pd.DataFrame(anova_data)
    
    if anova_df.empty:
        print("No data available for ANOVA.")
        return

    model = ols('ep_return ~ C(policy) + C(model_type) + C(policy):C(model_type)', data=anova_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    print("\nTwo-Way ANOVA Results for Episode Return:")
    print(anova_table)
    print("\nInterpretation:")
    print("- C(policy): Effect of the algorithm (PPO, HAPPO, MAPPO).")
    print("- C(model_type): Effect of the model architecture (Baseline, Ensemble, Ensemble+LLM).")
    print("- C(policy):C(model_type): Interaction effect between algorithm and architecture.")
    print("A low p-value (PR(>F) < 0.05) indicates a statistically significant effect.")

    print("\n--- Tukey's HSD Post-Hoc Test ---")
    anova_df['combination'] = anova_df.policy + " / " + anova_df.model_type
    tukey = pairwise_tukeyhsd(endog=anova_df['ep_return'], groups=anova_df['combination'], alpha=0.05)
    print(tukey)
    print("\nInterpretation:")
    print("- The table shows pairwise comparisons between all experimental groups.")
    print("- 'reject=True' indicates a statistically significant difference between the pair.")


def perform_loo_comparison(all_data):
    """Performs Bayesian model comparison using LOO-CV and PSIS."""
    print("\n\n" + "="*50)
    print("  PERFORMING BAYESIAN MODEL COMPARISON (LOO-CV)  ")
    print("="*50)
    print("Comparing models based on their expected log predictive density (ELPD).")
    print("Higher ELPD_loo is better. The best model will have a score of 0 in the comparison table.")

    mcmc_results = {}
    
    def bernoulli_beta(trials, successes=None):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 10))
        beta = numpyro.sample('beta', dist.Uniform(0, 10))
        theta = numpyro.sample('theta', dist.Beta(alpha, beta))
        numpyro.sample('obs', dist.Binomial(total_count=trials, probs=theta), obs=successes)

    for exp_name, data in all_data.items():
        if 'evaluation_runs' in data and not data['evaluation_runs'].empty:
            try:
                n = data['metadata']['evaluation_num_episodes'].iloc[0]
                win_rate = data['evaluation_runs']['win_rate'].iloc[0]
                w = int(win_rate * n)
                
                kernel = NUTS(bernoulli_beta)
                mcmc = MCMC(kernel, num_warmup=1000, num_samples=5000, num_chains=4)
                mcmc.run(random.PRNGKey(0), trials=n, successes=w)
                
                mcmc_results[exp_name] = az.from_numpyro(mcmc)
            except (IndexError, KeyError):
                print(f"  - Skipping LOO for {exp_name}: missing evaluation data.")

    if not mcmc_results:
        print("No models available for LOO comparison.")
        return

    loo_compare = az.compare(mcmc_results, ic="loo")
    print("\nLOO Comparison Table:")
    print(loo_compare)
    az.plot_compare(loo_compare, insample_dev=False)
    plt.title("LOO-CV Model Comparison")
    plt.show()
    
def perform_pairwise_bayesian(all_data, comparison_pairs):
    """Performs direct pairwise Bayesian comparisons of win rates."""
    print("\n\n" + "="*50)
    print("  PERFORMING PAIRWISE BAYESIAN COMPARISONS  ")
    print("="*50)

    for agent1_name, agent2_name in comparison_pairs:
        print(f"\n--- Comparing Win-Rates: {agent1_name} vs {agent2_name} ---")
        
        if agent1_name in all_data and agent2_name in all_data:
            data1 = all_data[agent1_name]
            data2 = all_data[agent2_name]

            alpha1, beta1 = data1.get('posterior_alpha'), data1.get('posterior_beta')
            alpha2, beta2 = data2.get('posterior_alpha'), data2.get('posterior_beta')

            if alpha1 is None or alpha2 is None:
                print("  - Skipping: missing posterior data for one or both agents.")
                continue

            n_samples = 50000
            samples1 = stats.beta.rvs(alpha1, beta1, size=n_samples)
            samples2 = stats.beta.rvs(alpha2, beta2, size=n_samples)

            delta = samples2 - samples1
            p_better = (delta > 0).mean()
            
            print(f"Posterior probability that '{agent2_name}' is better than '{agent1_name}': {p_better:.2%}")
        else:
            print(f"  - Skipping: Data for '{agent1_name}' or '{agent2_name}' not found.")


# --- 4. Grouped / aggregated plots (policy x method) ---
def _parse_policy_variant_from_name(exp_name: str):
    policy = exp_name.split('_')[0]
    if 'ensemble' in exp_name and 'llm' in exp_name:
        variant = 'Ensemble+LLM'
    elif 'ensemble' in exp_name:
        variant = 'Ensemble'
    else:
        variant = 'Baseline'
    return policy, variant

def build_grouped_epoch_df(all_data):
    """Return DataFrame with canonical columns: epoch, loss, policy, variant (local copy only)."""
    epoch_candidates = ['epoch', 'total_steps', 'timestamp', 'step', 'iteration']
    loss_candidates = ['loss', 'ep_return', 'value', 'train_loss', 'metric']

    rows = []
    for exp_name, data in all_data.items():
        df = data.get('epoch_summary')
        if df is None or df.empty:
            continue
        epoch_col = _find_col(df, epoch_candidates)
        loss_col = _find_col(df, loss_candidates)
        if epoch_col is None or loss_col is None:
            # skip if this experiment doesn't provide the necessary columns
            continue
        policy, variant = _parse_policy_variant_from_name(exp_name)
        tmp = df[[epoch_col, loss_col]].copy()
        tmp = tmp.rename(columns={epoch_col: 'epoch', loss_col: 'loss'})
        tmp['policy'] = policy
        tmp['variant'] = variant
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=['epoch','loss','policy','variant'])
    return pd.concat(rows, ignore_index=True)

def build_grouped_training_df(all_data):
    """Return DataFrame with canonical columns where possible (local copy only)."""
    epnum_candidates = ['episode_num', 'ep', 'episode']
    epret_candidates = ['ep_return', 'return', 'avg_return']
    length_candidates = ['ep_length', 'episode_length', 'length']
    result_candidates = ['result', 'outcome', 'game_result']
    rows = []
    for exp_name, data in all_data.items():
        df = data.get('training_episodes')
        if df is None or df.empty:
            continue
        tmp = df.copy()
        epoch_col = _find_col(tmp, ['epoch','total_steps','timestamp'])
        epnum_col = _find_col(tmp, epnum_candidates)
        epret_col = _find_col(tmp, epret_candidates)
        length_col = _find_col(tmp, length_candidates)
        result_col = _find_col(tmp, result_candidates)

        if epoch_col is None and epnum_col is None and epret_col is None:
            continue

        rename_map = {}
        if epoch_col and epoch_col != 'epoch':
            rename_map[epoch_col] = 'epoch'
        if epnum_col and epnum_col != 'episode_num':
            rename_map[epnum_col] = 'episode_num'
        if epret_col and epret_col != 'ep_return':
            rename_map[epret_col] = 'ep_return'
        if length_col and length_col != 'ep_length':
            rename_map[length_col] = 'ep_length'
        if result_col and result_col != 'result':
            rename_map[result_col] = 'result'
        if rename_map:
            tmp = tmp.rename(columns=rename_map)

        policy, variant = _parse_policy_variant_from_name(exp_name)
        tmp['policy'] = policy
        tmp['variant'] = variant
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def _ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def plot_grouped_epoch_loss(all_data, results_dir='analysis_results'):
    """Aggregate epoch→loss across experiments and save comparisons by policy/variant."""
    df = build_grouped_epoch_df(all_data)
    if df.empty:
        print("  - No epoch_summary data available for grouped loss plots.")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='whitegrid')

    for policy in df['policy'].unique():
        out_dir = os.path.join(results_dir, policy)
        _ensure_dir(out_dir)
        sub = df[df['policy'] == policy]
        plt.figure(figsize=(10,6))
        sns.lineplot(data=sub, x='epoch', y='loss', hue='variant', estimator='mean', errorbar=('ci', 95))
        plt.title(f'{policy.upper()} — Loss by Variant (pooled)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(title='Variant')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'loss_by_variant.png'), dpi=200)
        plt.show()
        plt.close()

    # Plot: for each variant, compare policies
    for variant in df['variant'].unique():
        out_dir = os.path.join(results_dir, variant.replace('+','_'))
        _ensure_dir(out_dir)
        sub = df[df['variant'] == variant]
        plt.figure(figsize=(10,6))
        sns.lineplot(data=sub, x='epoch', y='loss', hue='policy', estimator='mean', errorbar=('ci', 95))
        plt.title(f'{variant} — Loss by Policy (pooled)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(title='Policy')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'loss_by_policy.png'), dpi=200)
        plt.show()
        plt.close()

def plot_grouped_outcome_variability(all_data, results_dir='analysis_results'):
    """Violin plots of returns per variant within each policy, saved per-policy."""
    df = build_grouped_training_df(all_data)
    if df.empty or 'ep_return' not in df.columns:
        print("  - No training_episodes ep_return data available for grouped outcome variability.")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='whitegrid')

    for policy in df['policy'].unique():
        out_dir = os.path.join(results_dir, policy)
        _ensure_dir(out_dir)
        sub = df[df['policy'] == policy].copy()
        if 'episode_num' in sub.columns:
            sub = sub[sub['episode_num'] > (sub['episode_num'].max() / 2)]
        plt.figure(figsize=(10,6))
        sns.violinplot(data=sub, x='variant', y='ep_return', inner='quartile', cut=0, palette='muted', hue='variant')
        plt.title(f'{policy.upper()} — Return Distribution by Variant (stable window)')
        plt.xlabel('Variant')
        plt.ylabel('Episode Return')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'outcome_variability_by_variant.png'), dpi=200)
        plt.show()
        plt.close()

def plot_grouped_win_tie_loss(all_data, results_dir='analysis_results', bin_size=50):
    """Stacked bar of Win/Tie/Loss counts per variant, saved per-policy."""
    df = build_grouped_training_df(all_data)
    if df.empty or 'result' not in df.columns:
        print("  - No training_episodes result data available for grouped win/tie/loss.")
        return

    import matplotlib.pyplot as plt
    for policy in df['policy'].unique():
        out_dir = os.path.join(results_dir, policy)
        _ensure_dir(out_dir)
        sub = df[df['policy'] == policy].copy()
        if 'epoch' in sub.columns:
            sub['epoch_bin'] = (sub['epoch'] // bin_size) * bin_size
            counts = sub.groupby(['variant','epoch_bin','result']).size().unstack(fill_value=0).reset_index()
            for variant in counts['variant'].unique():
                vdf = counts[counts['variant'] == variant].set_index('epoch_bin')
                for col in ['Win','Tie','Loss']:
                    if col not in vdf.columns:
                        vdf[col] = 0
                vdf[['Win','Tie','Loss']].plot(kind='bar', stacked=True, figsize=(12,6),
                                              color=['#2ca02c', '#ff7f0e', '#d62728'])
                plt.title(f'{policy} — {variant} Outcomes per Epoch Bin')
                plt.xlabel('Epoch (Binned)')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'win_tie_loss_{variant.replace("+","_")}.png'), dpi=200)
                plt.show()
                plt.close()
        else:
            counts = sub.groupby(['variant','result']).size().unstack(fill_value=0)
            for variant in counts.index:
                row = counts.loc[variant]
                fig, ax = plt.subplots(figsize=(6,4))
                row[['Win','Tie','Loss']].plot(kind='bar', stacked=False, color=['#2ca02c','#ff7f0e','#d62728'], ax=ax)
                ax.set_title(f'{policy} — {variant} Outcome Counts (aggregate)')
                ax.set_ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'win_tie_loss_{variant.replace("+","_")}.png'), dpi=200)
                plt.show()
                plt.close()

def plot_grouped_win_speed(all_data, results_dir='analysis_results'):
    """Histogram of episode lengths for winning games by variant, saved per-policy."""
    df = build_grouped_training_df(all_data)
    if df.empty or 'ep_length' not in df.columns or 'result' not in df.columns:
        print("  - No ep_length/result data available for grouped win speed.")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='whitegrid')

    for policy in df['policy'].unique():
        out_dir = os.path.join(results_dir, policy)
        _ensure_dir(out_dir)
        sub = df[(df['policy'] == policy) & (df['result'] == 'Win')].copy()
        if sub.empty:
            continue
        plt.figure(figsize=(10,6))
        sns.histplot(data=sub, x='ep_length', hue='variant', kde=True, element='step', stat='count')
        plt.title(f'{policy.upper()} — Episode Length (Wins) by Variant')
        plt.xlabel('Episode Length (Steps)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'win_speed_by_variant.png'), dpi=200)
        plt.show()
        plt.close()
