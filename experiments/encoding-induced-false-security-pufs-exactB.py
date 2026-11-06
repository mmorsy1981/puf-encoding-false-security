"""Exact-spec reruns B (heavy): RSO m=32/N=1M T0; ROC3 noisy T2 spot.
RSO target: ~50% (1806.02011: 32 keys, 1M CRPs, 5 learners).
ROC3 noisy: T0 expected ~50%, T2-gated expected high (tests noise robustness of break).
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def lr_acc(Xtr, ytr, Xte, yte, seed=0):
    if len(np.unique(ytr)) < 2:
        return float((yte == np.bincount(yte).argmax()).mean())
    c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
    c.fit(Xtr.astype(np.float32), ytr); return float(c.score(Xte.astype(np.float32), yte))

rows = []; t0 = time.time()
# ---- RSO m=32, N=1M ----
for inst in [0, 1]:
    puf = ArbiterPUF(n=64, seed=100+inst, noisiness=0.0)
    rng = np.random.default_rng(5000+inst)
    K = rng.integers(0, 2, size=(32, 64)).astype(np.uint8)
    b = rng.integers(0, 2, size=(32,))
    rtr = np.random.default_rng(5001+inst); rte = np.random.default_rng(5002+inst)
    Ctr = rtr.integers(0, 2, size=(1000000, 64)).astype(np.uint8)
    ii, jj = rtr.integers(0, 32, size=1000000), rtr.integers(0, 32, size=1000000)
    ytr = (labels(puf, Ctr ^ K[ii]) ^ b[jj]).astype(int)
    Cte = rte.integers(0, 2, size=(10000, 64)).astype(np.uint8)
    iii, jjj = rte.integers(0, 32, size=10000), rte.integers(0, 32, size=10000)
    yte = (labels(puf, Cte ^ K[iii]) ^ b[jjj]).astype(int)
    a = lr_acc(Ctr, ytr, Cte, yte, inst)
    rows.append(('RSO-exact', 'm=32,N=1M', inst, round(a, 4)))
    print('RSO-1M', inst, round(a, 4), 'elapsed', round(time.time()-t0, 1), flush=True)
    del Ctr, ytr, ii, jj
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-exactB.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['design', 'config', 'instance', 'acc']); w.writerows(rows)
