"""LFSR slice-trained control (V4-fix slice 6).
Parity LR trained ONLY on C2=0 challenges (Base=10, 5 inst): train on C2=0
subset of 10k, test on C2=0 slice of 5k; plus per-instance split reporting of
the slice-trained model on overall / C2=0 / C2!=0 test partitions.
Model/data match lfsr-eq7-parity.py: Galois taps (0,1,3,4), shift k=C2*Base,
puf seeds 200+inst, rng 6000+inst, external-parity feats on C1 (64->65 dims),
liblinear. Does NOT modify lfsr-eq7-parity.csv.
Writes experiments/encoding-induced-false-security-pufs-pilot-lfsr-slice.csv
cols: base,instance,n_train_slice,n_test_c2eq0,acc_overall,acc_c2eq0,acc_c2ne0
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

CSV = 'experiments/encoding-induced-false-security-pufs-pilot-lfsr-slice.csv'

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
for inst in [0, 1, 2, 3, 4]:
    puf = ArbiterPUF(n=64, seed=200+inst, noisiness=0.0)
    rng = np.random.default_rng(6000+inst)
    Ctr = rng.integers(0, 2, size=(10000, 68)).astype(np.uint8)
    Cte = rng.integers(0, 2, size=(5000, 68)).astype(np.uint8)
    ytr = labels(puf, lfsr_int(Ctr, 10)); yte = labels(puf, lfsr_int(Cte, 10))
    c2tr = Ctr[:, 64:].dot([8, 4, 2, 1]); c2te = Cte[:, 64:].dot([8, 4, 2, 1])
    mtr = (c2tr == 0); mte0 = (c2te == 0); mte1 = ~mte0
    n_tr = int(mtr.sum()); n_te0 = int(mte0.sum())
    c = LogisticRegression(max_iter=1000, solver='liblinear').fit(pfeat(Ctr[mtr][:, :64]), ytr[mtr])
    pred = c.predict(pfeat(Cte[:, :64]))
    acc_all = round(float((pred == yte).mean()), 4)
    acc0 = round(float((pred[mte0] == yte[mte0]).mean()) if n_te0 else float('nan'), 4)
    acc1 = round(float((pred[mte1] == yte[mte1]).mean()), 4)
    rows.append((10, inst, n_tr, n_te0, acc_all, acc0, acc1))
    print(f'inst{inst} n_tr0={n_tr} n_te0={n_te0} overall={acc_all} c2eq0={acc0} c2ne0={acc1}', flush=True)
print('elapsed', round(time.time()-t0, 1))
with open(CSV, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['base', 'instance', 'n_train_slice', 'n_test_c2eq0', 'acc_overall', 'acc_c2eq0', 'acc_c2ne0']); w.writerows(rows)
