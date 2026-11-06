"""RQ2 extensions: OB-PUF 4-expert EM mixture; LFSR-APUF C2-grouped divide-and-conquer.
OB: pattern unknown per sample -> EM over 4 experts (each = 4 parity-LRs, one per
output bit, trained on samples assigned to that pattern); 4-class gating LR for decode.
LFSR: C2 (shift count) is part of the EXTERNAL challenge (block segmentation),
so group by observed C2 (16 groups); within a group the C1->internal map is a
FIXED public transform -> parity LR per group. N=100k (~6k/group).
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def pfeat(C, n=64):
    S = 1-2*C.astype(float); X = np.ones((len(C), n+1))
    for i in range(n): X[:, i] = np.prod(S[:, i:], axis=1)
    return X
def fit_lr(Xtr, ytr, seed=0):
    c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
    c.fit(Xtr.astype(float), ytr); return c

OB_POS = [[0,1,2],[0,1,63],[0,62,63],[61,62,63]]
OB_VAL = np.array([[0,1,0],[1,0,1],[0,1,0],[1,0,1]], dtype=np.uint8)
OB_SR = np.array([[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,1,0]], dtype=np.uint8)
def ob_full(Cob, i):
    N = len(Cob); C = np.zeros((N, 61+3), dtype=np.uint8)
    for t in range(N):
        pos = sorted(OB_POS[i]); vals = OB_VAL[i]; src = 0; p = 0
        for dst in range(64):
            if p < 3 and dst == pos[p]: C[t, dst] = vals[p]; p += 1
            else: C[t, dst] = Cob[t, src]; src += 1
    return C
def ob_gen(inst, N, seed, nz=0.0):
    pufs = [ArbiterPUF(n=64, seed=300+inst*10+j, noisiness=nz) for j in range(4)]
    r = np.random.default_rng(seed)
    Cob = r.integers(0, 2, size=(N, 61)).astype(np.uint8)
    ii = r.integers(0, 4, size=N)
    Rob = np.zeros((N, 4), dtype=int)
    for i in range(4):
        m = ii == i
        if m.sum(): Rob[m] = np.column_stack([labels(p, ob_full(Cob[m], i)) for p in pufs]) ^ OB_SR[i]
    return Cob, Rob

def ob_em(Ctr, Ytr, Cte, Yte, iters=5, seed=0):
    rng = np.random.default_rng(seed); asg = rng.integers(0, 4, size=len(Ctr)); hist = []
    for it in range(iters):
        experts = []
        for i in range(4):
            m = (asg == i)
            if m.sum() < 200: experts.append(None); continue
            experts.append([fit_lr(pfeat(ob_full(Ctr[m], i)), Ytr[m, j], seed) for j in range(4)])
        # E-step: agreement count across 4 bits
        agree = np.zeros((len(Ctr), 4))
        for i in range(4):
            if experts[i] is None: continue
            Fi = ob_full(Ctr, i)
            for j in range(4):
                agree[:, i] += (experts[i][j].predict(pfeat(Fi)) == Ytr[:, j])
        new = np.argmax(agree, axis=1)
        new[agree.max(axis=1) == 0] = asg[agree.max(axis=1) == 0]
        asg = new
        # gated decode
        gate = fit_lr(pfeat(Ctr, 61), asg, seed)
        from sklearn.linear_model import LogisticRegression as L2
        gr = gate.predict(pfeat(Cte, 61))
        pred = np.zeros_like(Yte)
        for i in range(4):
            m = (gr == i)
            if m.sum() and experts[i] is not None:
                Fi = ob_full(Cte[m], i)
                for j in range(4): pred[m, j] = experts[i][j].predict(pfeat(Fi))
        acc = float((pred == Yte).mean())
        hist.append(round(acc, 4))
    return hist

# LFSR exact pieces (reuse Eq.7 Galois)
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
for inst in [0, 1, 2]:
    Ctr, Ytr = ob_gen(inst, 20000, 7000+inst)[:2]
    Cte, Yte = ob_gen(inst, 5000, 7100+inst)[:2]
    # T0 baseline: raw partial LR per bit
    base = float(np.mean([fit_lr(Ctr, Ytr[:, j], 0).score(Cte, Yte[:, j]) for j in range(4)])) if False else None
    b0 = [fit_lr(Ctr, Ytr[:, j], 0).score(Cte, Yte[:, j]) for j in range(4)]
    rows.append(('OB-T0-raw', inst, round(float(np.mean(b0)), 4)))
    print('OB T0 inst', inst, [round(float(x), 3) for x in b0], flush=True)
    hist = ob_em(Ctr, Ytr, Cte, Yte)
    rows.append(('OB-T2-EMgated', inst, hist[-1]))
    print('OB T2 inst', inst, hist, flush=True)
for inst in [0, 1, 2]:
    puf = ArbiterPUF(n=64, seed=200+inst, noisiness=0.0)
    rng = np.random.default_rng(6000+inst)
    Ctr = rng.integers(0, 2, size=(100000, 68)).astype(np.uint8)
    Cte = rng.integers(0, 2, size=(10000, 68)).astype(np.uint8)
    ytr = labels(puf, lfsr_int(Ctr)); yte = labels(puf, lfsr_int(Cte))
    vtr, vte = Ctr[:, 64:].dot([8, 4, 2, 1]), Cte[:, 64:].dot([8, 4, 2, 1])
    tot, n = 0, 0
    for v in range(16):
        m = (vtr == v); mt = (vte == v)
        if m.sum() < 500 or mt.sum() == 0: n += mt.sum(); continue
        c = fit_lr(pfeat(Ctr[m][:, :64]), ytr[m], 0)
        tot += (c.predict(pfeat(Cte[mt][:, :64])) == yte[mt]).sum(); n += mt.sum()
    acc = round(float(tot / n), 4)
    rows.append(('LFSR-T2-C2grouped', inst, acc))
    print('LFSR T2 inst', inst, acc, 'coverage', n, flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-t2ext.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['tier', 'instance', 'acc']); w.writerows(rows)
