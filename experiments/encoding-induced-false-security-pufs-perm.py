"""Leverage-3: public fixed bit-permutation layer. Tiers: raw LR (T0), naive parity LR
(their 0.525 case), T1 apply-known-permutation + parity. Expect: T0~0.52, naive-phi~0.52, T1~plain."""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def pfeat(C):
    S = 1 - 2*C.astype(float); X = np.ones((len(C), 65))
    for i in range(64): X[:, i] = np.prod(S[:, i:], axis=1)
    return X
def lr_acc(Xtr, ytr, Xte, yte, seed=0):
    if len(np.unique(ytr)) < 2:
        return float((yte == np.bincount(yte).argmax()).mean())
    c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
    c.fit(Xtr.astype(float), ytr); return float(c.score(Xte.astype(float), yte))

rngp = np.random.default_rng(12345)
PERM = rngp.permutation(64)
INV = np.argsort(PERM)
t0 = time.time()
rows = []
for inst in [0, 1, 2]:
    puf = ArbiterPUF(n=64, seed=inst, noisiness=0.0)
    rt = np.random.default_rng(41000+inst)
    Cte = rt.integers(0, 2, size=(5000, 64)).astype(np.uint8)
    yte = labels(puf, Cte[:, PERM])
    r = np.random.default_rng(42000+inst)
    Ctr = r.integers(0, 2, size=(5000, 64)).astype(np.uint8)
    ytr = labels(puf, Ctr[:, PERM])
    for N in [1000, 5000]:
        a = lr_acc(Ctr[:N].astype(float), ytr[:N], Cte.astype(float), yte, 0)
        b = lr_acc(pfeat(Ctr[:N]), ytr[:N], pfeat(Cte), yte, 0)
        c = lr_acc(pfeat(Ctr[:N][:, PERM]), ytr[:N], pfeat(Cte[:, PERM]), yte, 0)
        rows.append(('perm-raw-T0', N, inst, round(a, 4)))
        rows.append(('perm-naivephi', N, inst, round(b, 4)))
        rows.append(('perm-T1-applyperm', N, inst, round(c, 4)))
        print(f'inst{inst} N={N} raw={a:.3f} naivephi={b:.3f} T1={c:.3f}', flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-perm.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['tier', 'N', 'instance', 'acc']); w.writerows(rows)
