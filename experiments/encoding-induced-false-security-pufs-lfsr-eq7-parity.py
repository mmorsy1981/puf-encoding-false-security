"""LFSR Eq.7 + external-parity learner (Review v2 §4.2/§6.2): discriminating replication.
Model: Galois LFSR map from lfsr-t1ms.py (taps (0,1,3,4) stand-in, shift = C2*Base,
seeds 200+inst / rng 6000+inst+1000*aseed). Learner: parity LR on EXTERNAL C1 bits
(competent uninformed: succeeds on plain APUF, ~0.99). Bases {0,1,3,10,20} x 5 inst x 2 seeds,
N=10k train / 5k test, noiseless. Compares vs paper Fig.9 (Base=0 ~93%, 1 -> 54%, 10 -> 51.79%).
Writes experiments/encoding-induced-false-security-pufs-lfsr-eq7-parity.csv
cols: base,instance,aseed,acc
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
rows = []; t0 = time.time()
for base in [0, 1, 3, 10, 20]:
    for inst in [0, 1, 2, 3, 4]:
        for asd in [0, 1]:
            puf = ArbiterPUF(n=64, seed=200+inst, noisiness=0.0)
            rng = np.random.default_rng(6000+inst+1000*asd)
            Ctr = rng.integers(0, 2, size=(10000, 68)).astype(np.uint8)
            Cte = rng.integers(0, 2, size=(5000, 68)).astype(np.uint8)
            ytr = labels(puf, lfsr_int(Ctr, base)); yte = labels(puf, lfsr_int(Cte, base))
            c = LogisticRegression(max_iter=1000, solver='liblinear').fit(pfeat(Ctr[:, :64]), ytr)
            a = round(float(c.score(pfeat(Cte[:, :64]), yte)), 4)
            rows.append((base, inst, asd, a)); print('base', base, 'inst', inst, 'asd', asd, a, flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-lfsr-eq7-parity.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['base', 'instance', 'aseed', 'acc']); w.writerows(rows)
from collections import defaultdict
d = defaultdict(list)
for b, i, a_, v in rows: d[b].append(v)
for b in sorted(d): print('Base', b, 'mean', round(float(np.mean(d[b])), 4), 'range', (min(d[b]), max(d[b])))
