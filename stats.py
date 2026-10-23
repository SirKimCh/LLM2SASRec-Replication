"""Statistics pre-registered in ho_so/09_exp_plan.md. PRIMARY: paired differences across runs (same seed), t CI, two-sided
one-sample t, Holm–Bonferroni (m = 4) over C1–C4. SECONDARY: per-session Wilcoxon signed-rank + bootstrap 95 % CI (10 000, seed 42)."""
import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(diff, n=10000, seed=42, alpha=0.05):
    d = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired(a, b):
    """a, b: per-session values (pd.Series indexed by SessionId). Difference a − b on the common sessions."""
    a, b = a.align(b, join="inner")
    d = (a - b).to_numpy()
    p = 1.0 if not np.any(d) else float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    lo, hi = bootstrap_ci(d)
    return {"n": len(d), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p}


def seed_paired(a, b, alpha=0.05):
    """PRIMARY test (G1.5, Gusak et al. RecSys'26): paired differences ACROSS RUNS. a, b: pd.Series of per-seed NDCG@20 indexed
    by seed (same seeds for both configurations). Returns mean difference, t-based 95 % CI over the seed pairs and the two-sided
    one-sample t p-value. With 5 seeds the power is low: a non-significant result is "khong ket luan duoc", not "no effect"."""
    from scipy.stats import t as tdist
    a, b = a.align(b, join="inner")
    d = (a - b).to_numpy(dtype=float)
    n = len(d)
    if n < 2:
        return {"n_pairs": n, "seed_mean_diff": float(d.mean()) if n else float("nan"), "seed_ci_lo": float("nan"), "seed_ci_hi": float("nan"), "seed_p": float("nan")}
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        p = 0.0 if m != 0 else 1.0
        return {"n_pairs": n, "seed_mean_diff": float(m), "seed_ci_lo": float(m), "seed_ci_hi": float(m), "seed_p": p}
    q = tdist.ppf(1 - alpha / 2, n - 1)
    p = 2 * tdist.sf(abs(m / se), n - 1)
    return {"n_pairs": n, "seed_mean_diff": float(m), "seed_ci_lo": float(m - q * se), "seed_ci_hi": float(m + q * se), "seed_p": float(p)}


def welch(a, b):
    """Seed-level test (G2): Welch t-test on the per-seed means of two configurations (two-sided). Needs >= 2 seeds each."""
    from scipy.stats import ttest_ind
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(ttest_ind(a, b, equal_var=False).pvalue)


def holm(pvals):
    """{name: p} -> {name: Holm-adjusted p} (step-down, monotone, capped at 1)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out
