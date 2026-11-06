"""ROC3 composite noisy ceiling (Review v2 §4.1): same-object repeated evals.
For each of 2 instances: fixed Cte (10k, rng 3000+inst, matches roc3-noisy.py);
clean labels once (nz=0); noisy labels 5x from the SAME nz=0.1 object (fresh objects
replay identical noise, so new objects would fake agreement=1.0).
Metrics: composite R agreement mean, R* agreement mean.
Writes experiments/encoding-induced-false-security-pufs-roc3-noisy-ceiling.csv
cols: instance,metric,value
"""
import csv, numpy as np
src = open('experiments/encoding-induced-false-security-pufs-roc3.py').read().split('rows = []')[0]
ns = {}; exec(src, ns)
from pypuf.simulation import ArbiterPUF
rows = []
for inst in [0, 1]:
    r = np.random.default_rng(3000+inst)
    Cte = r.integers(0, 2, size=(10000, 64)).astype(np.uint8)
    puf0 = ArbiterPUF(n=64, seed=400+inst, noisiness=0.0)
    Rs0 = ns['labels'](puf0, Cte)
    y0 = ns['labels'](puf0, ns['roc3_internal'](Cte, Rs0))
    puns = ArbiterPUF(n=64, seed=400+inst, noisiness=0.1)
    ca, ra = [], []
    for _ in range(5):
        Rs = ns['labels'](puns, Cte)
        ra.append(float((Rs == Rs0).mean()))
        ca.append(float((ns['labels'](puns, ns['roc3_internal'](Cte, Rs)) == y0).mean()))
    rows.append((inst, 'composite-agreement-mean5', round(float(np.mean(ca)), 4)))
    rows.append((inst, 'rstar-agreement-mean5', round(float(np.mean(ra)), 4)))
    print('inst', inst, 'composite', round(float(np.mean(ca)), 4), 'rstar', round(float(np.mean(ra)), 4), flush=True)
with open('experiments/encoding-induced-false-security-pufs-roc3-noisy-ceiling.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['instance', 'metric', 'value']); w.writerows(rows)
