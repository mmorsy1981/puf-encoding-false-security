"""E1 missing-cell controls — vetted pilot.
m=0 RSO control in the SAME harness (raw-LR T0 + parity-LR T0), m in {0,1,8,32},
N=100k/10k, 2 instances. m=0: plain APUF (no mask, no flip). m=1: fixed mask+flip.
m=8/32: paper sweep points (gate2 continuity).
Vetted bar: a T0 learner must reach ~95% at the limit (m=0/1) for the sweep to be
interpretable. Both learners reported; raw-LR expected weak (~0.6), parity-LR
expected ~0.99 at limit and ~0.5 at m=32 (paper: <65%@2, <60%@8, <55%@16, ~50%@32).
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def parity_feats(C):
    S = 1 - 2*C.astype(float); X = np.ones((len(C), 65))
    for i in range(64): X[:, i] = np.prod(S[:, i:], axis=1)
    return X

def fit_score(Xtr, ytr, Xte, yte, seed):
    if len(np.unique(ytr)) < 2:
        return float((yte == np.bincount(yte).argmax()).mean())
    c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
    c.fit(Xtr, ytr); return float(c.score(Xte, yte))

def rso_gen(puf, K, b, N, r):
    m = len(K)
    C = r.integers(0, 2, size=(N, 64)).astype(np.uint8)
    if m == 0:
        return C, labels(puf, C)
    ii, jj = r.integers(0, m, size=N), r.integers(0, m, size=N)
    obs = np.zeros(N, dtype=int)
    for t in range(N):
        obs[t] = labels(puf, (C[t] ^ K[ii[t]]).reshape(1, -1))[0] ^ b[jj[t]]
    return C, obs

rows = []; t0 = time.time(); NTR, NTE = 100000, 10000
for inst in [0, 1]:
    puf = ArbiterPUF(n=64, seed=100+inst, noisiness=0.0)
    for m in [0, 1, 8, 32]:
        rng = np.random.default_rng(5000+inst)
        K = rng.integers(0, 2, size=(m, 64)).astype(np.uint8) if m else np.zeros((0, 64), np.uint8)
        b = rng.integers(0, 2, size=(m,)) if m else np.zeros(0, dtype=int)
        Ctr, ytr = rso_gen(puf, K, b, NTR, np.random.default_rng(6000+inst*10+m))
        Cte, yte = rso_gen(puf, K, b, NTE, np.random.default_rng(7000+inst*10+m))
        a_raw = fit_score(Ctr.astype(float), ytr, Cte.astype(float), yte, inst)
        a_par = fit_score(parity_feats(Ctr), ytr, parity_feats(Cte), yte, inst)
        rows.append(('E1', f'm={m}-N=100k-rawLR', inst, round(a_raw, 4)))
        rows.append(('E1', f'm={m}-N=100k-parLR', inst, round(a_par, 4)))
        print(f'E1 inst {inst} m={m} raw={a_raw:.4f} par={a_par:.4f}', flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-e1m0.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'instance', 'acc']); w.writerows(rows)
