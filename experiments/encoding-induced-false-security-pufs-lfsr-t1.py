"""LFSR-APUF proper T1 (audit close-out): replaces orphan lfsr-t1.csv with a
checked-in generator. Per-sample: evolve C1 by Galois LFSR Shift=C2*Base (Base=10,
taps (0,1,3,4) stand-in, documented), then parity-LR. N grid {1000,2000,10000},
instances 0-2, attack seed 0 (matches prior runs' convention)."""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
t0 = time.time()
def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def pfeat(C):
    S = 1-2*C.astype(float); X = np.ones((len(C), 65))
    for i in range(64): X[:, i] = np.prod(S[:, i:], axis=1)
    return X
def galois(s, cl, taps=(0, 1, 3, 4)):
    s = s.copy()
    for _ in range(cl):
        fb = s[-1]
        for t in taps: s[t] ^= fb
        s = np.roll(s, 1); s[0] = fb
    return s
def lfsr_int(C68, base=10):
    N = len(C68); out = np.zeros((N, 64), dtype=np.uint8)
    for t in range(N):
        k = int(C68[t, 64:].dot([8, 4, 2, 1])) * base
        out[t] = galois(C68[t, :64], k)
    return out
rows = []
for inst in [0, 1, 2]:
    puf = ArbiterPUF(n=64, seed=200+inst, noisiness=0.0)
    rng = np.random.default_rng(6000+inst)
    Ctr = rng.integers(0, 2, size=(20000, 68)).astype(np.uint8)
    Cte = rng.integers(0, 2, size=(5000, 68)).astype(np.uint8)
    ytr = labels(puf, lfsr_int(Ctr)); yte = labels(puf, lfsr_int(Cte))
    Ptr, Pte = pfeat(lfsr_int(Ctr)), pfeat(lfsr_int(Cte))
    for N in [1000, 2000, 10000]:
        c = LogisticRegression(max_iter=1000, solver='liblinear').fit(Ptr[:N], ytr[:N])
        a = round(float(c.score(Pte, yte)), 4)
        rows.append(('LFSR-T1-proper', N, inst, a)); print(inst, N, a, flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-lfsr-t1.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['tier', 'N', 'instance', 'acc']); w.writerows(rows)
