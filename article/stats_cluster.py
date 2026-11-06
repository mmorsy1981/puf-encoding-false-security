"""Cluster-level statistics for the secret-table experiments.
Restarts inside one instance (or one device) are correlated, so every interval here resamples whole
instances (devices) with replacement and never individual restarts. Fixed seed, B = 4000 resamples.

Conventions: a 'matrix' has one row per cluster (instance or device) and one column per N; an entry is the
fraction of that cluster's restarts that succeeded at that N.
"""
import math
import numpy as np

B = 4000
_RNG = np.random.default_rng(20261024)


def boot_idx(n, b=B):
    return _RNG.integers(0, n, (b, n))


def mean_ci(vals, alpha=0.05):
    v = np.asarray(vals, float)
    idx = boot_idx(len(v))
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def band(mat, alpha=0.05):
    """Per-column mean over clusters with a cluster-bootstrap percentile band. Returns (mean, lo, hi) arrays."""
    mat = np.asarray(mat, float)
    idx = boot_idx(mat.shape[0])
    boots = mat[idx].mean(1)                       # (B, ncols)
    return mat.mean(0), np.percentile(boots, 100 * alpha / 2, axis=0), np.percentile(boots, 100 * (1 - alpha / 2), axis=0)


def n50(Ns, rates):
    """Log-interpolated N at which the rate first crosses 0.5 (NaN if it never does or starts above)."""
    Ns, rates = list(Ns), list(rates)
    if rates[0] >= 0.5:
        return float('nan')
    for i in range(1, len(Ns)):
        if rates[i - 1] < 0.5 <= rates[i]:
            t = (0.5 - rates[i - 1]) / (rates[i] - rates[i - 1])
            return math.exp(math.log(Ns[i - 1]) + t * (math.log(Ns[i]) - math.log(Ns[i - 1])))
    return float('nan')


def n50_ci(Ns, mat, known=None, alpha=0.05):
    """N50 of the cluster-mean curve with a cluster-bootstrap interval.
    known: optional per-cluster known-table N-to-0.95; the ratio N50 / median(known) is bootstrapped with the same
    resamples. Returns dict(n50, lo, hi, ratio, rlo, rhi, nan)."""
    mat = np.asarray(mat, float)
    pt = n50(Ns, mat.mean(0))
    idx = boot_idx(mat.shape[0])
    vals, rat = [], []
    for row in idx:
        v = n50(Ns, mat[row].mean(0))
        vals.append(v)
        if known is not None:
            rat.append(v / float(np.nanmedian(np.asarray(known, float)[row])))
    vals = np.array(vals)
    ok = vals[~np.isnan(vals)]
    out = {'n50': pt, 'nan': int(np.isnan(vals).sum()),
           'lo': float(np.percentile(ok, 100 * alpha / 2)) if len(ok) else float('nan'),
           'hi': float(np.percentile(ok, 100 * (1 - alpha / 2))) if len(ok) else float('nan')}
    if known is not None:
        rat = np.array(rat)
        rok = rat[~np.isnan(rat)]
        out["ratio"] = pt / float(np.nanmedian(known))
        out['rlo'] = float(np.percentile(rok, 100 * alpha / 2)) if len(rok) else float('nan')
        out['rhi'] = float(np.percentile(rok, 100 * (1 - alpha / 2))) if len(rok) else float('nan')
    return out


def median_ci(vals, alpha=0.05):
    v = np.asarray(vals, float)
    idx = boot_idx(len(v))
    m = np.median(v[idx], axis=1)
    return float(np.median(v)), float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def ratio_of_ratios(Ns_a, mat_a, kn_a, Ns_b, mat_b, kn_b, alpha=0.05):
    """(N50_b / median known_b) / (N50_a / median known_a) with a joint instance bootstrap (same instance ids in a and b)."""
    mat_a, mat_b = np.asarray(mat_a, float), np.asarray(mat_b, float)
    kn_a, kn_b = np.asarray(kn_a, float), np.asarray(kn_b, float)
    k = min(len(mat_a), len(mat_b))
    mat_a, mat_b, kn_a, kn_b = mat_a[:k], mat_b[:k], kn_a[:k], kn_b[:k]
    pt = (n50(Ns_b, mat_b.mean(0)) / np.median(kn_b)) / (n50(Ns_a, mat_a.mean(0)) / np.median(kn_a))
    out = []
    for row in boot_idx(k):
        v = (n50(Ns_b, mat_b[row].mean(0)) / np.median(kn_b[row])) / (n50(Ns_a, mat_a[row].mean(0)) / np.median(kn_a[row]))
        out.append(v)
    out = np.array(out)
    out = out[~np.isnan(out)]
    return float(pt), float(np.percentile(out, 100 * alpha / 2)), float(np.percentile(out, 100 * (1 - alpha / 2)))


def interp95(pairs, thr=0.95):
    """Log-linear interpolation of the smallest N at which a known-table accuracy curve first reaches thr.
    pairs: iterable of (N, accuracy). Returns NaN if the curve never reaches thr; the first N if it starts above."""
    pairs = sorted(pairs)
    if pairs[0][1] >= thr:
        return float(pairs[0][0])
    for (n0, a0), (n1, a1) in zip(pairs, pairs[1:]):
        if a0 < thr <= a1:
            t = (thr - a0) / (a1 - a0)
            return math.exp(math.log(n0) + t * (math.log(n1) - math.log(n0)))
    return float('nan')
