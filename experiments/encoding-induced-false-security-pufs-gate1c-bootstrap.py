"""Bootstrap CIs for Gate1 interpolated N-to-threshold ratios (Review-3 Q1/W1).
Resamples the 10 configs (5 inst x 2 seeds) with replacement, 20000 replicates;
ratio-of-medians per replicate; 95% percentile CI. Writes CSV alongside.
Reads experiments/encoding-induced-false-security-pufs-gate1c-finegrid.csv
Writes experiments/encoding-induced-false-security-pufs-gate1c-bootstrap.csv
cols: threshold,ratio_median,ci_lo,ci_hi
"""
import csv, numpy as np
from collections import defaultdict
d = defaultdict(dict)
for r in csv.DictReader(open('experiments/encoding-induced-false-security-pufs-gate1c-finegrid.csv', encoding='utf-8')):
    if r['tier'] in ('plain-parity', 'sbox-piparity-T1'):
        d[(r['tier'], int(r['instance']), int(r['aseed']))][int(r['N'])] = float(r['acc'])
def nto(curve, thr):
    xs = sorted(curve)
    for i, x in enumerate(xs):
        if curve[x] >= thr:
            return float(x) if i == 0 else xs[i-1] + (thr - curve[xs[i-1]]) / (curve[x] - curve[xs[i-1]]) * (x - xs[i-1])
    return None
rng = np.random.default_rng(7)
rows = []
for thr in (0.90, 0.95):
    P = np.array([nto(d[('plain-parity', i, a)], thr) for i in range(5) for a in range(2)])
    T = np.array([nto(d[('sbox-piparity-T1', i, a)], thr) for i in range(5) for a in range(2)])
    rmed = float(np.median(T)) / float(np.median(P))
    bs = np.array([float(np.median(rng.choice(T, 10, replace=True))) / float(np.median(rng.choice(P, 10, replace=True))) for _ in range(20000)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    rows.append((thr, round(rmed, 2), round(lo, 2), round(hi, 2)))
    print('thr', thr, 'ratio', round(rmed, 2), 'CI', (round(lo, 2), round(hi, 2)), flush=True)
with open('experiments/encoding-induced-false-security-pufs-gate1c-bootstrap.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['threshold', 'ratio_median', 'ci_lo', 'ci_hi']); w.writerows(rows)
