"""Noisy ceiling, corrected (Review v2 §4.1): same-object repeated evals.
pypuf replays identical noise for fresh same-seed objects (fresh-vs-fresh agree=1.0),
so noise draws must come from repeated eval() calls on ONE object.
Plain 64-bit APUF, fixed 10k test set, 5 instances. Metrics: clean-vs-noisy agreement
(mean of 5 same-object repeats) + same-object repeat agreement (intra-run noise floor).
Writes experiments/encoding-induced-false-security-pufs-noisy-ceiling.csv
cols: instance,metric,value
"""
import csv, numpy as np
from pypuf.simulation import ArbiterPUF
def labels(puf, C01): return ((puf.eval((2*C01.astype(np.int8) - 1)).reshape(-1) + 1)//2).astype(int)
rows = []
for inst in range(5):
    r = np.random.default_rng(8100+inst)
    Cte = r.integers(0, 2, size=(10000, 64)).astype(np.uint8)
    y0 = labels(ArbiterPUF(n=64, seed=500+inst, noisiness=0.0), Cte)
    pun = ArbiterPUF(n=64, seed=500+inst, noisiness=0.1)
    ys = [labels(pun, Cte) for _ in range(5)]
    rows.append((inst, 'clean-vs-noisy-mean5', round(float(np.mean([float((y == y0).mean()) for y in ys])), 4)))
    rows.append((inst, 'sameobj-repeat-agree', round(float((ys[0] == ys[1]).mean()), 4)))
    print('inst', inst, 'ceiling', rows[-2][2], 'repeat', rows[-1][2], flush=True)
with open('experiments/encoding-induced-false-security-pufs-noisy-ceiling.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['instance', 'metric', 'value']); w.writerows(rows)
import collections
d = collections.defaultdict(list)
for i, m, v in rows: d[m].append(v)
print({m: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) for m, v in d.items()})
