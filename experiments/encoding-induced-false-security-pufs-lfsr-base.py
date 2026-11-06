"""LFSR-APUF Base sweep (Review-3 issue F): discriminating replication of Fig.9.
Model: gate2.py lfsr_internal (64-bit Fibonacci, taps x^64+x^4+x^3+x+1,
clocks=Base+popcount(C)%16, state->APUF). Paper Fig.9 (LR, 10k CRPs):
Base=0 -> ~93%, Base=1 -> ~54%, Base=10 -> 51.79%.
Sweep bases {0,1,3,5,8,10,20} x 3 inst, raw-bit LR, Ntr=10k, Nte=5k.
Writes experiments/encoding-induced-false-security-pufs-lfsr-base.csv
cols: base,instance,acc
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
TAPS = (64, 4, 3, 1, 0)
def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def lfsr_internal(C, base):
    N = len(C); out = np.zeros_like(C)
    for t in range(N):
        s = C[t].copy(); k = base + int(C[t].sum()) % 16
        for _ in range(k):
            fb = s[0] ^ s[TAPS[0]-TAPS[1]] ^ s[TAPS[0]-TAPS[2]] ^ s[TAPS[0]-TAPS[3]]
            s = np.roll(s, -1); s[-1] = fb
        out[t] = s
    return out
rows = []; t0 = time.time()
for base in [0, 1, 3, 5, 8, 10, 20]:
    for inst in [0, 1, 2]:
        puf = ArbiterPUF(n=64, seed=200+inst, noisiness=0.0)
        rng = np.random.default_rng(6100+inst)
        Ctr = rng.integers(0, 2, size=(10000, 64)).astype(np.uint8)
        Cte = rng.integers(0, 2, size=(5000, 64)).astype(np.uint8)
        ytr = labels(puf, lfsr_internal(Ctr, base)); yte = labels(puf, lfsr_internal(Cte, base))
        if len(np.unique(ytr)) < 2:
            a = float((yte == np.bincount(yte).argmax()).mean())
        else:
            c = LogisticRegression(max_iter=1000, solver='liblinear').fit(Ctr.astype(float), ytr)
            a = float(c.score(Cte.astype(float), yte))
        rows.append((base, inst, round(a, 4))); print('base', base, 'inst', inst, round(a, 4), flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-lfsr-base.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['base', 'instance', 'acc']); w.writerows(rows)
from collections import defaultdict
d = defaultdict(list)
for b, i, a in rows: d[b].append(a)
for b in sorted(d): print('Base', b, 'mean', round(float(np.mean(d[b])), 4), d[b])
