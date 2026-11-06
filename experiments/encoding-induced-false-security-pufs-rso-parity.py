"""RSO competent-learner replication (Review v2 §6.6): parity LR on EXTERNAL challenges.
gate2's rso_trial uses raw-bit LR. A competent learner (parity features, which get ~0.99
on a plain APUF) should still fail on RSO if the keys hide the structure — making the
~50% match discriminating rather than strawman. m=32 keys, N=100k/10k, 2 inst, noiseless.
Writes experiments/encoding-induced-false-security-pufs-rso-parity.csv
cols: instance,acc
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def pfeat(C):
    S = 1 - 2*C.astype(float); X = np.ones((len(C), 65))
    for i in range(64): X[:, i] = np.prod(S[:, i:], axis=1)
    return X
rows = []; t0 = time.time()
for inst in [0, 1]:
    puf = ArbiterPUF(n=64, seed=100+inst, noisiness=0.0)
    rng = np.random.default_rng(5000+inst)
    K = rng.integers(0, 2, size=(32, 64)).astype(np.uint8)
    b = rng.integers(0, 2, size=(32,))
    rtr = np.random.default_rng(5001+inst); rte = np.random.default_rng(5002+inst)
    Ctr = rtr.integers(0, 2, size=(100000, 64)).astype(np.uint8)
    ii, jj = rtr.integers(0, 32, size=100000), rtr.integers(0, 32, size=100000)
    ytr = (labels(puf, Ctr ^ K[ii]) ^ b[jj]).astype(int)
    Cte = rte.integers(0, 2, size=(10000, 64)).astype(np.uint8)
    iii, jjj = rte.integers(0, 32, size=10000), rte.integers(0, 32, size=10000)
    yte = (labels(puf, Cte ^ K[iii]) ^ b[jjj]).astype(int)
    c = LogisticRegression(max_iter=1000, solver='liblinear').fit(pfeat(Ctr), ytr)
    a = round(float(c.score(pfeat(Cte), yte)), 4)
    rows.append((inst, a)); print('RSO parity-LR inst', inst, a, 'elapsed', round(time.time()-t0, 1), flush=True)
with open('experiments/encoding-induced-false-security-pufs-rso-parity.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['instance', 'acc']); w.writerows(rows)
