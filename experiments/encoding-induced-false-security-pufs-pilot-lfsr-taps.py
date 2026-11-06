"""LFSR tap-agnosticism sweep (audit R1): Eq.7 external-parity replication across tap sets.
Same harness/seeds as lfsr-eq7-parity.py; taps vary over 5 stand-in 4-tap sets
(paper omits taps). Bases {0,1,3,10,20} x 5 inst x 2 seeds, N=10k/5k, noiseless.
Writes experiments/encoding-induced-false-security-pufs-pilot-lfsr-taps.csv
cols: tapset,base,instance,aseed,acc
"""
import time, csv, os, sys, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
TAPSETS = {
    't014': (0, 1, 3, 4),
    't0249': (0, 2, 4, 9),
    't03511': (0, 3, 5, 11),
    't01712': (0, 1, 7, 12),
    't05915': (0, 5, 9, 15),
}
def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def pfeat(C):
    S = 1 - 2*C.astype(float); X = np.ones((len(C), 65))
    for i in range(64): X[:, i] = np.prod(S[:, i:], axis=1)
    return X
def galois(s, cl, taps):
    s = s.copy()
    for _ in range(cl):
        fb = s[-1]
        for t in taps: s[t] ^= fb
        s = np.roll(s, 1); s[0] = fb
    return s
def lfsr_int(C68, base, taps):
    N = len(C68); out = np.zeros((N, 64), dtype=np.uint8)
    for t in range(N):
        k = int(C68[t, 64:].dot([8, 4, 2, 1])) * base
        out[t] = galois(C68[t, :64], k, taps)
    return out
rows = []; t0 = time.time()
want = sys.argv[1:] or list(TAPSETS)
for ts in want:
    taps = TAPSETS[ts]
    for base in [0, 1, 3, 10, 20]:
        for inst in [0, 1, 2, 3, 4]:
            for asd in [0, 1]:
                puf = ArbiterPUF(n=64, seed=200+inst, noisiness=0.0)
                rng = np.random.default_rng(6000+inst+1000*asd)
                Ctr = rng.integers(0, 2, size=(10000, 68)).astype(np.uint8)
                Cte = rng.integers(0, 2, size=(5000, 68)).astype(np.uint8)
                ytr = labels(puf, lfsr_int(Ctr, base, taps)); yte = labels(puf, lfsr_int(Cte, base, taps))
                c = LogisticRegression(max_iter=1000, solver='liblinear').fit(pfeat(Ctr[:, :64]), ytr)
                a = round(float(c.score(pfeat(Cte[:, :64]), yte)), 4)
                rows.append((ts, base, inst, asd, a))
    print('tapset', ts, 'done', round(time.time()-t0, 1), flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-lfsr-taps.csv', 'a', newline='') as f:
    w = csv.writer(f)
    if os.path.getsize('experiments/encoding-induced-false-security-pufs-pilot-lfsr-taps.csv') == 0:
        w.writerow(['tapset', 'base', 'instance', 'aseed', 'acc'])
    w.writerows(rows)
from collections import defaultdict
d = defaultdict(list)
for ts, b, i, a_, v in rows: d[(ts, b)].append(v)
for k in sorted(d): print(k, 'mean', round(float(np.mean(d[k])), 4), 'range', (min(d[k]), max(d[k])))
