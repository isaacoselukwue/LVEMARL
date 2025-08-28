import itertools
import math
from typing import Dict, List, Tuple, Optional
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pygam import LinearGAM, s
# ------------------------------
# Metric registry & aggregation
# ------------------------------

# Where to find each metric and how to turn episode-level into epoch-level
# source ∈ {'epoch_summary','evaluation_runs','training_episodes'}
# agg is used only when source == 'training_episodes'
METRIC_REGISTRY = {
    # epoch_summary (already per-epoch)
    'loss':             {'source': 'epoch_summary',      'col': 'loss'},
    'kl_divergence':    {'source': 'epoch_summary',      'col': 'kl_divergence'},
    'total_steps':      {'source': 'epoch_summary',      'col': 'total_steps'},

    # evaluation_runs (already per-epoch evaluation)
    'win_rate':         {'source': 'evaluation_runs',    'col': 'win_rate'},
    'avg_return':       {'source': 'evaluation_runs',    'col': 'avg_return'},

    # training_episodes (per-episode → aggregate to epoch mean)
    'policy_entropy':   {'source': 'training_episodes',  'col': 'policy_entropy', 'agg': 'mean'},
    'action_prob_gap':  {'source': 'training_episodes',  'col': 'action_prob_gap', 'agg': 'mean'},
    'ep_return':        {'source': 'training_episodes',  'col': 'ep_return', 'agg': 'mean'},
}

# ---------------------------
# Helpers for data wrangling
# ---------------------------

def _infer_cols(df: pd.DataFrame,
                time_candidates: List[str] = ["step", "t", "iteration", "episode", "epoch", "timesteps", "frame"],
                loss_candidates: List[str] = ["loss", "value", "reward", "returns", "score", "metric"]) -> Tuple[str, str]:
    """
    Try to guess the time and loss columns.
    Raises if nothing matches.
    """
    time_col = None
    for c in time_candidates:
        if c in df.columns:
            time_col = c
            break
    loss_col = None
    for c in loss_candidates:
        if c in df.columns:
            loss_col = c
            break
    if time_col is None:
        raise ValueError(f"Could not infer time column. Tried: {time_candidates}. Columns present: {list(df.columns)}")
    if loss_col is None:
        raise ValueError(f"Could not infer loss/metric column. Tried: {loss_candidates}. Columns present: {list(df.columns)}")
    return time_col, loss_col


def parse_experiment_columns(df: pd.DataFrame,
                             exp_col: str = "experiment") -> pd.DataFrame:
    """
    Expect df to have 'experiment' naming like:
      happo_pommerman, mappo_ensemble_pommerman_llm, etc.
    We derive:
      - algo: happo/mappo/ppo
      - variant: baseline | ensemble | ensemble_llm
      - env: pommerman (best-effort)
    """
    if exp_col not in df.columns:
        raise ValueError(f"Expected '{exp_col}' column in df.")
    out = df.copy()
    def _parse_one(s: str) -> Tuple[str,str,str]:
        s = str(s)
        parts = s.split("_")
        algo = None
        variant = "baseline"
        env = None
        # best-effort parse
        if len(parts) > 0:
            algo = parts[0]
        if "ensemble" in parts and "llm" in parts:
            variant = "ensemble_llm"
        elif "ensemble" in parts:
            variant = "ensemble"
        if parts:
            env = parts[-1]
            # handle trailing 'llm'
            if env == "llm" and len(parts) >= 2:
                env = parts[-2]
        return algo or "unknown", variant, env or "unknown"
    parsed = out[exp_col].map(_parse_one)
    out["algo"] = parsed.map(lambda x: x[0])
    out["variant"] = parsed.map(lambda x: x[1])
    out["env"] = parsed.map(lambda x: x[2])
    return out

def run_full_metric_suite(all_data: dict,
                          outdir: str = "analysis_results",
                          n_boot: int = 5000):
    os.makedirs(outdir, exist_ok=True)

    metrics = [
        'loss',
        'kl_divergence',
        'policy_entropy',
        'win_rate',
        'avg_return',
        'total_steps',
        'action_prob_gap',
        'ep_return'
    ]

    longdfs = {m: build_metric_long_df(all_data, m) for m in metrics}

    for m, dfm in longdfs.items():
        if dfm.empty:
            print(f"[warn] No data for {m}")
            continue
        plot_metric_by_group(dfm, m, hue='variant',
                             title_suffix='(pooled)',
                             save_path=os.path.join(outdir, f"{m}_by_variant.png"))
        plot_metric_by_group(dfm, m, hue='algo',
                             title_suffix='(pooled)',
                             save_path=os.path.join(outdir, f"{m}_by_algo.png"))

    for m in ['loss','kl_divergence','policy_entropy','win_rate','avg_return']:
        dfm = longdfs.get(m, pd.DataFrame())
        if dfm.empty:
            print(f"[warn] Skipping contrasts for {m} (no data)")
            continue

        for variant in ['baseline','ensemble','ensemble_llm']:
            png = os.path.join(outdir, f"contr_{m}_algo_{variant}.png")
            csv = os.path.join(outdir, f"contr_{m}_algo_{variant}.csv")
            run_bootstrap_contrasts_for_metric(
                dfm, m, group_col='algo',
                within={'variant': variant},
                n_boot=n_boot,
                save_csv=csv,
                save_png=png
            )

        for algo in ['happo','mappo','ppo']:
            png = os.path.join(outdir, f"contr_{m}_variant_{algo}.png")
            csv = os.path.join(outdir, f"contr_{m}_variant_{algo}.csv")
            run_bootstrap_contrasts_for_metric(
                dfm, m, group_col='variant',
                within={'algo': algo},
                n_boot=n_boot,
                save_csv=csv,
                save_png=png
            )

    print(f"Done. Outputs in: {outdir}")


# -----------------------------------------
# Summary metrics for a single loss curve
# -----------------------------------------

def final_window_mean(df: pd.DataFrame,
                      time_col: Optional[str] = None,
                      loss_col: Optional[str] = None,
                      window_frac: float = 0.1,
                      min_window: int = 100) -> float:
    """
    Mean loss over the final window of training.
    """
    if time_col is None or loss_col is None:
        time_col, loss_col = _infer_cols(df)
    tmp = df.sort_values(time_col, kind="mergesort")
    n = len(tmp)
    w = max(min_window, int(max(1, math.floor(n * window_frac))))
    tail = tmp.iloc[-w:][loss_col].to_numpy()
    return float(np.nanmean(tail))


def area_under_curve(df: pd.DataFrame,
                     time_col: Optional[str] = None,
                     loss_col: Optional[str] = None,
                     normalize_time: bool = True) -> float:
    """
    Simple trapezoidal AUC over training. If normalize_time, divide by (t_max - t_min) to scale to ~mean level.
    """
    if time_col is None or loss_col is None:
        time_col, loss_col = _infer_cols(df)
    tmp = df.sort_values(time_col, kind="mergesort")
    t = tmp[time_col].to_numpy()
    y = tmp[loss_col].to_numpy()
    if len(t) < 2:
        return float(y.mean()) if len(y) else np.nan
    auc = np.trapz(y, t)
    if normalize_time:
        denom = (t[-1] - t[0])
        if denom <= 0:
            return float(y.mean())
        return float(auc / denom)
    return float(auc)


# --------------------------------------
# Moving-block bootstrap for time series
# --------------------------------------

def _moving_block_bootstrap_1d(x: np.ndarray,
                               block_size: Optional[int] = None,
                               n_samples: int = 1000,
                               random_state: Optional[int] = None) -> np.ndarray:
    """
    Simple circular moving-block bootstrap for a 1D array.
    Returns an array of shape (n_samples, len(x)).
    """
    rng = np.random.default_rng(random_state)
    n = len(x)
    if n == 0:
        raise ValueError("Empty series provided to bootstrap.")
    if block_size is None:
        block_size = max(10, int(math.sqrt(n)))
    n_blocks = int(math.ceil(n / block_size))
    # circular indices
    xb = np.pad(x, (0, block_size), mode="wrap")
    out = np.empty((n_samples, n), dtype=float)
    for i in range(n_samples):
        start_idx = rng.integers(0, n)  # choose starting point for each block
        blocks = []
        for _ in range(n_blocks):
            s = rng.integers(0, n)  # random starting index for each block
            blocks.append(xb[s:s+block_size])
        resampled = np.concatenate(blocks)[:n]
        out[i] = resampled
    return out


def bootstrap_mean_ci(x: np.ndarray,
                      block_size: Optional[int] = None,
                      n_samples: int = 5000,
                      ci: float = 0.95,
                      random_state: Optional[int] = None) -> Tuple[float, Tuple[float,float]]:
    """
    Block-bootstrap the mean and return (mean_estimate, (lo, hi)).
    """
    bs = _moving_block_bootstrap_1d(x, block_size=block_size, n_samples=n_samples, random_state=random_state)
    means = bs.mean(axis=1)
    lo, hi = np.quantile(means, [(1-ci)/2, 1-(1-ci)/2])
    return float(np.mean(means)), (float(lo), float(hi))


def bootstrap_mean_diff_ci(x: np.ndarray, y: np.ndarray,
                           block_size: Optional[int] = None,
                           n_samples: int = 5000,
                           ci: float = 0.95,
                           random_state: Optional[int] = None) -> Tuple[float, Tuple[float,float], float]:
    """
    Block-bootstrap the difference in means: mean(x) - mean(y).
    Returns (diff_estimate, (lo, hi), p_two_sided),
    where p is the proportion of bootstrap diffs that cross zero (two-sided).
    """
    bsx = _moving_block_bootstrap_1d(x, block_size=block_size, n_samples=n_samples, random_state=random_state)
    bsy = _moving_block_bootstrap_1d(y, block_size=block_size, n_samples=n_samples, random_state=random_state)
    dx = bsx.mean(axis=1) - bsy.mean(axis=1)
    diff_est = float(np.mean(dx))
    lo, hi = np.quantile(dx, [(1-ci)/2, 1-(1-ci)/2])
    p_left = np.mean(dx <= 0)
    p_right = np.mean(dx >= 0)
    p_two = 2*min(p_left, p_right)
    p_two = min(1.0, max(0.0, p_two))
    return diff_est, (float(lo), float(hi)), float(p_two)


# ---------------------------------------------------
# Pairwise contrasts built from per-experiment curves
# ---------------------------------------------------

def _ensure_columns(df: pd.DataFrame, required: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Present: {list(df.columns)}")


def _series_from_df(df: pd.DataFrame,
                    time_col: str,
                    loss_col: str,
                    use_final_window: bool = True,
                    window_frac: float = 0.1,
                    min_window: int = 100) -> np.ndarray:
    """Return the vector of losses to feed the bootstrap.
       If use_final_window=True, restrict to last window to emphasize end performance."""
    tmp = df.sort_values(time_col, kind="mergesort")
    y = tmp[loss_col].to_numpy()
    if not use_final_window:
        return y
    n = len(y)
    w = max(min_window, int(max(1, math.floor(n * window_frac))))
    return y[-w:]


def pairwise_contrasts(df: pd.DataFrame,
                       group_col: str,
                       exp_col: str = "experiment",
                       time_col: Optional[str] = None,
                       loss_col: Optional[str] = None,
                       within: Optional[Dict[str, str]] = None,
                       n_boot: int = 5000,
                       block_size: Optional[int] = None,
                       use_final_window: bool = True,
                       window_frac: float = 0.1,
                       min_window: int = 100,
                       random_state: Optional[int] = 42) -> pd.DataFrame:
    """
    Build all pairwise contrasts for levels of `group_col` (e.g., algo) within optional constraints (e.g., variant=baseline).
    Returns a tidy DataFrame with mean diffs and CI.
    """
    if time_col is None or loss_col is None:
        time_col, loss_col = _infer_cols(df)
    work = df.copy()
    if within:
        for k, v in within.items():
            work = work[work[k] == v]
    _ensure_columns(work, [group_col, time_col, loss_col, exp_col])
    results = []
    for a, b in itertools.combinations(sorted(work[group_col].dropna().unique()), 2):
        A = work[work[group_col] == a]
        B = work[work[group_col] == b]
        if A.empty or B.empty:
            continue
        xa = _series_from_df(A, time_col, loss_col, use_final_window, window_frac, min_window)
        xb = _series_from_df(B, time_col, loss_col, use_final_window, window_frac, min_window)
        diff, (lo, hi), p = bootstrap_mean_diff_ci(xa, xb, block_size=block_size, n_samples=n_boot, random_state=random_state)
        results.append({
            "contrast": f"{a} - {b}",
            "group": group_col,
            "within": within if within else {},
            "mean_diff": diff,
            "ci_lo": lo,
            "ci_hi": hi,
            "p_two_sided": p,
            "n_a": len(xa),
            "n_b": len(xb),
            "block_size": block_size if block_size is not None else int(max(10, math.sqrt(max(len(xa), len(xb))))),
            "used_final_window": use_final_window,
            "window_frac": window_frac,
            "min_window": min_window,
        })
    out = pd.DataFrame(results).sort_values("p_two_sided")
    return out


# -------------------
# Simple CI plotting
# -------------------

def plot_contrasts_ci(contrast_df: pd.DataFrame, title: str = "Pairwise contrasts (mean diff with 95% CI)"):
    """
    Matplotlib plot: mean difference with 95% CI. Positive means group A > group B.
    Robust to NaNs and swapped CI bounds.
    """
    if contrast_df.empty:
        raise ValueError("Empty contrasts DataFrame.")

    df = contrast_df.copy()
    # drop bad rows
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean_diff", "ci_lo", "ci_hi"])
    if df.empty:
        raise ValueError("All contrast rows have invalid CI values.")

    # ensure lo <= hi
    lo = df[["ci_lo", "ci_hi"]].min(axis=1)
    hi = df[["ci_lo", "ci_hi"]].max(axis=1)

    # non-negative error bars
    err_left  = (df["mean_diff"] - lo).clip(lower=0)
    err_right = (hi - df["mean_diff"]).clip(lower=0)
    xerr = np.vstack([err_left.to_numpy(), err_right.to_numpy()])

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5*len(df))))
    y = np.arange(len(df))[::-1]

    ax.errorbar(
        x=df["mean_diff"].to_numpy(),
        y=y,
        xerr=xerr,
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df["contrast"].tolist())
    ax.set_xlabel("Mean difference (A - B)")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# -----------------------------------------
# Convenience: end-to-end typical workflow
# -----------------------------------------

def build_and_plot_all(df: pd.DataFrame,
                       exp_col: str = "experiment",
                       groupings: List[Tuple[str, Optional[Dict[str,str]]]] = None,
                       time_col: Optional[str] = None,
                       loss_col: Optional[str] = None,
                       n_boot: int = 5000,
                       random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Example workflow that:
      - Ensures (algo, variant, env) exist from experiment
      - Builds pairwise contrasts for each (group_col, within) config
      - Plots CI for each
    Returns dict of {key: contrast_df}
    """
    if groupings is None:
        groupings = [
            ("algo", {"variant": "baseline"}),
            ("algo", {"variant": "ensemble"}),
            ("algo", {"variant": "ensemble_llm"}),
            ("variant", {"algo": "happo"}),
            ("variant", {"algo": "mappo"}),
            ("variant", {"algo": "ppo"}),
        ]
    dfp = parse_experiment_columns(df, exp_col=exp_col) if "algo" not in df.columns else df
    out = {}
    for group_col, within in groupings:
        key = f"{group_col}|{within}" if within else group_col
        contr = pairwise_contrasts(
            dfp,
            group_col=group_col,
            exp_col=exp_col,
            time_col=time_col,
            loss_col=loss_col,
            within=within,
            n_boot=n_boot,
            random_state=random_state
        )
        if not contr.empty:
            plot_contrasts_ci(contr, title=f"Pairwise contrasts for {group_col} within {within}")
        out[key] = contr
    return out

def aggregate_training_to_epoch(training_df):
    """Aggregate episode-level metrics (policy_entropy, ep_return, etc.) into epoch-level means."""
    if training_df.empty:
        return pd.DataFrame()
    agg = training_df.groupby('epoch').agg({
        'policy_entropy': 'mean',
        'ep_return': 'mean',
        'ep_length': 'mean',
        'action_prob_gap': 'mean'
    }).reset_index()
    return agg

def _exp_to_algo_variant_env(exp_name: str):
    parts = exp_name.split('_')
    algo = parts[0] if parts else 'unknown'
    if 'ensemble' in parts and 'llm' in parts:
        variant = 'ensemble_llm'
    elif 'ensemble' in parts:
        variant = 'ensemble'
    else:
        variant = 'baseline'
    env = parts[-1] if parts else 'unknown'
    if env == 'llm' and len(parts) >= 2:
        env = parts[-2]
    return algo, variant, env

def _aggregate_training_to_epoch(training_df: pd.DataFrame, col: str, agg: str = 'mean') -> pd.DataFrame:
    if training_df is None or training_df.empty or col not in training_df.columns:
        return pd.DataFrame(columns=['epoch', col])
    # Note: training_episodes.epoch starts at 0 in your data. That's OK; we don’t shift it.
    out = training_df.groupby('epoch', as_index=False)[col].agg(agg)
    return out

def build_metric_long_df(all_data: dict, metric_name: str) -> pd.DataFrame:
    """
    Returns a tidy long DF with columns:
      ['experiment','epoch','value','algo','variant','env']
    for the requested metric_name (per METRIC_REGISTRY).
    """
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric '{metric_name}'. Add it to METRIC_REGISTRY.")

    spec = METRIC_REGISTRY[metric_name]
    source = spec['source']
    col    = spec['col']
    agg    = spec.get('agg', None)

    rows = []
    for exp_name, bundles in all_data.items():
        df = bundles.get(source, pd.DataFrame())
        sub = pd.DataFrame()

        if source in ('epoch_summary',):
            if not df.empty and {'epoch', col}.issubset(df.columns):
                sub = df[['epoch', col]].dropna().copy()

        elif source in ('evaluation_runs',):
            if not df.empty and {'epoch', col}.issubset(df.columns):
                sub = df[['epoch', col]].dropna().copy()
            else:
                tr = bundles.get('training_episodes', pd.DataFrame())
                synth = _fallback_eval_from_training(tr, want=col)
                if not synth.empty:
                    sub = synth.rename(columns={col: col})

        elif source == 'training_episodes':
            sub = _aggregate_training_to_epoch(df, col=col, agg=agg)

        if sub is None or sub.empty:
            continue

        algo, variant, env = _exp_to_algo_variant_env(exp_name)
        sub['experiment'] = exp_name
        sub['algo'] = algo
        sub['variant'] = variant
        sub['env'] = env
        sub = sub.rename(columns={col: 'value'})
        rows.append(sub[['experiment','epoch','value','algo','variant','env']])

    if not rows:
        return pd.DataFrame(columns=['experiment','epoch','value','algo','variant','env'])
    out = pd.concat(rows, ignore_index=True)
    out = out[np.isfinite(out['value'])]
    return out

def _fallback_eval_from_training(training_df: pd.DataFrame, want: str) -> pd.DataFrame:
    """
    Build per-epoch win_rate or avg_return from training_episodes.
    Returns df[['epoch', '<metric>']] or empty if impossible.
    """
    if training_df is None or training_df.empty or 'epoch' not in training_df.columns:
        return pd.DataFrame()

    if want == 'win_rate':
        if 'result' not in training_df.columns:
            return pd.DataFrame()
        g = training_df.groupby('epoch')['result'].apply(lambda s: (s == 'Win').mean())
        return g.reset_index().rename(columns={'result': 'win_rate'})

    if want == 'avg_return':
        if 'ep_return' not in training_df.columns:
            return pd.DataFrame()
        g = training_df.groupby('epoch')['ep_return'].mean()
        return g.reset_index().rename(columns={'ep_return': 'avg_return'})

    return pd.DataFrame()

def plot_metric_by_group(metric_df: pd.DataFrame, metric_name: str,
                         hue: str = 'variant', title_suffix: str = '', save_path: str = None):
    """
    Line plot of metric vs epoch, averaged across experiments, with 95% CI,
    grouped by 'hue' (e.g., 'variant' or 'algo').
    """
    if metric_df.empty:
        print(f"  - No data to plot for metric '{metric_name}'.")
        return

    import seaborn as sns
    sns.set(style='whitegrid')

    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=metric_df,
        x='epoch', y='value', hue=hue,
        estimator='mean', errorbar=('ci', 95)
    )
    plt.title(f"{metric_name.replace('_',' ').title()} by {hue.title()} {title_suffix}".strip())
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.replace('_',' ').title())
    plt.legend(title=hue.title())
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220)
    plt.show()
    plt.close()

def run_bootstrap_contrasts_for_metric(metric_df: pd.DataFrame,
                                       metric_name: str,
                                       group_col: str,
                                       within: dict = None,
                                       n_boot: int = 5000,
                                       save_csv: str = None,
                                       save_png: str = None):
    """
    Wrapper that reuses the generic contrasts machinery for any metric.
    Assumes metric_df has columns: ['experiment','epoch','value', group_col, ...]
    """
    if metric_df.empty:
        print(f"  - No data for contrasts on '{metric_name}'.")
        return pd.DataFrame()

    df = metric_df.rename(columns={'value': metric_name}).copy()
    df['experiment'] = df['experiment'].astype(str)

    contr = pairwise_contrasts(
        df,
        group_col=group_col,
        exp_col='experiment',
        time_col='epoch',
        loss_col=metric_name,          # the thing we compare
        within=within,
        n_boot=n_boot,
        random_state=42
    )
    if contr.empty:
        print(f"  - No contrasts produced for '{metric_name}'.")
        return contr

    # Mark 95% CI significance
    contr['significant_95'] = (contr['ci_lo'] > 0) | (contr['ci_hi'] < 0)

    # Plot
    try:
        fig, ax = plot_contrasts_ci(contr, title=f"Pairwise contrasts: {group_col} within {within} — {metric_name}")
        if save_png:
            fig.savefig(save_png, dpi=220, bbox_inches='tight')
        plt.show()
        plt.close()
    except Exception as e:
        print(f"  - Plotting failed: {e}")

    if save_csv:
        contr.to_csv(save_csv, index=False)

    return contr

def _ensure_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}. Present: {list(df.columns)}")


def _prep_metric_df(metric_df: pd.DataFrame, time_col: str, group_col: str, value_col: str="value") -> pd.DataFrame:
    """Clean & sort a long metric DF so GAM can run reliably."""
    if metric_df is None or metric_df.empty:
        return pd.DataFrame(columns=[time_col, group_col, value_col, "experiment", "algo", "variant", "env"])
    df = metric_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[time_col, group_col, value_col])
    df = df.sort_values([group_col, time_col], kind="mergesort")
    return df


def fit_gam_per_group(metric_df: pd.DataFrame,
                      time_col: str = "epoch",
                      group_col: str = "variant",
                      value_col: str = "value",
                      lam: float = 0.6,
                      n_splines: int = 25) -> Dict[str, Dict]:
    """
    Fit a separate LinearGAM per level of `group_col` using a smooth of `time_col`.
    Returns: {group_level: {"gam": model, "Xgrid": Xgrid, "yhat": yhat, "ci": (lo, hi), "n": n}}
    """
    
    _ensure_cols(metric_df, [time_col, group_col, value_col], "fit_gam_per_group")

    out = {}
    for level, sub in metric_df.groupby(group_col):
        if sub[value_col].notna().sum() < 5:
            continue
        X = sub[[time_col]].to_numpy(dtype=float)
        y = sub[value_col].to_numpy(dtype=float)

        gam = LinearGAM(s(0, n_splines=n_splines), lam=lam).fit(X, y)

        x_min, x_max = float(np.nanmin(X)), float(np.nanmax(X))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
            continue
        Xgrid = np.linspace(x_min, x_max, 200).reshape(-1, 1)

        yhat = gam.predict(Xgrid)
        intervals = gam.prediction_intervals(Xgrid, width=0.95)
        lo = intervals[:, 0]
        hi = intervals[:, 1]
        out[level] = {"gam": gam, "Xgrid": Xgrid, "yhat": yhat, "ci": (lo, hi), "n": int(len(y))}
    return out


def summarize_gams(gams: Dict[str, Dict]) -> pd.DataFrame:
    """Return a tidy summary: group, n, edof, pseudo_r2, aic, gcv, scale."""
    rows = []
    for level, obj in gams.items():
        gam = obj["gam"]
        stats = getattr(gam, "statistics_", None)
        def _get_stat(key, default=np.nan):
            if stats is None:
                return default
            if isinstance(stats, dict):
                return stats.get(key, default)
            return getattr(stats, key, default)

        edof = _get_stat("edof", np.nan)
        pseudo_r2 = _get_stat("pseudo_r2", np.nan)
        aic = _get_stat("AIC", _get_stat("aic", np.nan))
        gcv = _get_stat("GCV", _get_stat("gcv", np.nan))
        scale = _get_stat("scale", np.nan)

        def _as_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        rows.append({
            "group": level,
            "n": obj.get("n", np.nan),
            "edof": _as_float(edof),
            "pseudo_r2": _as_float(pseudo_r2),
            "aic": _as_float(aic),
            "gcv": _as_float(gcv),
            "scale": _as_float(scale),
        })
    return (pd.DataFrame(rows)
              .sort_values("aic")
              if rows else pd.DataFrame(columns=["group","n","edof","pseudo_r2","aic","gcv","scale"]))


def plot_gam_groups(gams: Dict[str, Dict],
                    title: str,
                    xlabel: str = "Epoch",
                    ylabel: str = "Metric",
                    save_path: Optional[str] = None):
    """
    One matplotlib figure: each group's smooth with 95% CI ribbon.
    (Single chart per figure, no manual colors.)
    """
    if not gams:
        raise ValueError("No GAMs to plot.")
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for level, obj in gams.items():
        Xgrid = obj["Xgrid"]
        yhat = obj["yhat"]
        lo, hi = obj["ci"]
        plt.plot(Xgrid[:, 0], yhat, label=str(level))
        plt.fill_between(Xgrid[:, 0], lo, hi, alpha=0.15)

    plt.legend(title="Group")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close()


def run_gam_suite_for_metric(metric_df: pd.DataFrame,
                             metric_name: str,
                             outdir: str,
                             time_col: str = "epoch",
                             group_cols: List[str] = ("variant", "algo"),
                             lam: float = 0.6,
                             n_splines: int = 25) -> Dict[str, pd.DataFrame]:
    """
    For a given metric long-DF, fit/plot GAMs twice:
      - grouped by group_cols[0] (default: 'variant')
      - grouped by group_cols[1] (default: 'algo')
    Save plots and CSV summaries into outdir.
    Returns dict of summaries: {"by_variant": df, "by_algo": df}
    """
    os.makedirs(outdir, exist_ok=True)
    results = {}

    clean = _prep_metric_df(metric_df, time_col=time_col, group_col=group_cols[0], value_col="value")
    if not clean.empty:
        gams_v = fit_gam_per_group(clean, time_col=time_col, group_col=group_cols[0],
                                   value_col="value", lam=lam, n_splines=n_splines)
        if gams_v:
            plot_gam_groups(
                gams_v,
                title=f"{metric_name.replace('_', ' ').title()} — by {group_cols[0]}",
                xlabel=time_col.title(),
                ylabel=metric_name.replace('_', ' ').title(),
                save_path=os.path.join(outdir, f"{metric_name}_gam_by_{group_cols[0]}.png")
            )
            summ_v = summarize_gams(gams_v)
            summ_v.to_csv(os.path.join(outdir, f"{metric_name}_gam_by_{group_cols[0]}.csv"), index=False)
            results["by_"+group_cols[0]] = summ_v

    clean = _prep_metric_df(metric_df, time_col=time_col, group_col=group_cols[1], value_col="value")
    if not clean.empty:
        gams_a = fit_gam_per_group(clean, time_col=time_col, group_col=group_cols[1],
                                   value_col="value", lam=lam, n_splines=n_splines)
        if gams_a:
            plot_gam_groups(
                gams_a,
                title=f"{metric_name.replace('_', ' ').title()} — by {group_cols[1]}",
                xlabel=time_col.title(),
                ylabel=metric_name.replace('_', ' ').title(),
                save_path=os.path.join(outdir, f"{metric_name}_gam_by_{group_cols[1]}.png")
            )
            summ_a = summarize_gams(gams_a)
            summ_a.to_csv(os.path.join(outdir, f"{metric_name}_gam_by_{group_cols[1]}.csv"), index=False)
            results["by_"+group_cols[1]] = summ_a

    return results


def run_full_gam_suite(longdfs: Dict[str, pd.DataFrame],
                       outdir: str = "analysis_results_gam",
                       time_col: str = "epoch",
                       lam: float = 0.6,
                       n_splines: int = 25) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Loop over a dict of metric_name -> long DF,
    and fit/plot GAMs per metric by variant and by algo.
    Saves plots and tidy CSVs in outdir.
    Returns nested dict of summaries: {metric_name: {"by_variant": df, "by_algo": df}}
    """
    
    os.makedirs(outdir, exist_ok=True)
    out = {}
    for metric_name, df in longdfs.items():
        if df is None or df.empty:
            print(f"[GAM] Skipping '{metric_name}': empty data.")
            continue
        print(f"[GAM] Fitting {metric_name} ...")
        res = run_gam_suite_for_metric(
            df, metric_name, outdir=outdir, time_col=time_col,
            group_cols=("variant", "algo"), lam=lam, n_splines=n_splines
        )
        out[metric_name] = res
    print(f"[GAM] Done. Outputs in: {outdir}")
    return out

def build_all_longdfs(all_data: dict, metric_names=None) -> dict:
    """
    Returns a dict: {metric_name: long_df with columns
       ['experiment','epoch','value','algo','variant','env']}
    Uses your existing build_metric_long_df (and its evaluation fallbacks).
    """
    if metric_names is None:
        metric_names = list(METRIC_REGISTRY.keys())

    out = {}
    for m in metric_names:
        try:
            dfm = build_metric_long_df(all_data, m) 
        except Exception as e:
            print(f"[longdfs] {m}: failed to build ({e})")
            dfm = pd.DataFrame(columns=['experiment','epoch','value','algo','variant','env'])
        out[m] = dfm
    return out
