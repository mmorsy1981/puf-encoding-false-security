"""Plain parity-input big-MLP arm (V5 §3.1 requirement): the competent-uninformed
control missing from the E5 comparison. Identical parity-input big-MLP, N-grid
subset, torch seeds as pilot-e5parmlp.py; labels = plain APUF f(C).
Seeds: 0 = probe4 exact (puf 0, rng 11/12, expect 0.9797/0.9921 repro at 20k/200k);
1 = puf 1, rng 111/112; 2 = puf 2, rng 211/212. N in {20k, 200k, 1M}.
Writes experiments/encoding-induced-false-security-pufs-pilot-e5plainparmlp.csv
cols [test,config,instance,acc,train_acc].
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF

t0 = time.time()
src = open('experiments/encoding-induced-false-security-pufs-pilot-e5parmlp.py').read()
ns = {}
exec(src.split('t0 = time.time()')[0], ns)
bigmlp_par, parity_feats, labels, gen_labels_plain = ns['bigmlp_par'], ns['parity_feats'], ns['labels'], None

def plain_labels(puf, C):
    n = len(C); y = np.zeros(n, dtype=int)
    for s in range(0, n, 100000):
        y[s:s+100000] = labels(puf, C[s:s+100000])
    return y

rows = []
SEEDS = {0: (0, 11, 12), 1: (1, 111, 112), 2: (2, 211, 212)}
for s, (pseed, rtr, rte) in SEEDS.items():
    puf = ArbiterPUF(n=64, seed=pseed, noisiness=0.0)
    Ctr_full = np.random.default_rng(rtr).integers(0, 2, size=(1000000, 64)).astype(np.uint8)
    Cte = np.random.default_rng(rte).integers(0, 2, size=(10000, 64)).astype(np.uint8)
    ytr_full = plain_labels(puf, Ctr_full); yte = plain_labels(puf, Cte)
    Xte = parity_feats(Cte).astype(np.float32)
    for N in [20000, 200000, 1000000]:
        tr, te = bigmlp_par(parity_feats(Ctr_full[:N]).astype(np.float32), ytr_full[:N], Xte, yte, s)
        rows.append(('E5plainpar', f'parMLP-plain-N={N}', s, round(te, 4), round(tr, 4)))
        print(f'E5plainpar seed={s} N={N} test={te:.4f} train={tr:.4f}', flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-e5plainparmlp.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'instance', 'acc', 'train_acc']); w.writerows(rows)
