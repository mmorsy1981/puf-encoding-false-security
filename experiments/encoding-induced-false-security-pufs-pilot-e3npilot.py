"""E3 single-seed N-pilot — vetted scope (NOT the 10x3 grid).
Schemes (MEJO Fig.8, §3.1, verified from on-file PDF 2026-10-18):
  3-3: C1=flip(C,R*); C2=Sbox(C1); C*=shift(C2,R*); R=APUF(C*).
  3-2: C1=flip(C,R*); C2=shift(C,R*); C12=C1^C2; C*=Sbox(C12); R=APUF(C*).
Conventions: odd  = R*=0 flips 0-indexed even positions {0,2,4..} (roc3.py choice);
             even = R*=0 flips {1,3,5..}. Shift: R*=0 -> left(-1), else right(+1).
Grid: scheme{3-2,3-3} x conv{odd,even} x noise{0.0,0.1} x N{10k,20k,50k nested};
T0 raw-LR floor at 50k; T2 EM gated+ungated per N; noisy ceiling = clean-vs-noisy
label agreement on shared 10k C (same weight seed, noisiness 0.0 vs 0.1).
Single seed throughout (puf 400, train rng 2000, test rng 3000).
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

def load_gate1():
    src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
    ns = {}
    exec(src.split(chr(10) + "if __name__ == '__main__':")[0], ns)
    return ns
G = load_gate1()
SBOX_ARR = G['SBOX_ARR']
parity_feats, to_pm, labels = G['parity_feats'], G['to_pm'], G['labels']
bits_to_bytes, bytes_to_bits = G['bits_to_bytes'], G['bytes_to_bits']

def flip_oe(C, r, conv):
    C = C.copy()
    if conv == 'odd':
        idx = np.arange(0, 64, 2) if r == 0 else np.arange(1, 64, 2)
    else:
        idx = np.arange(1, 64, 2) if r == 0 else np.arange(0, 64, 2)
    C[:, idx] ^= 1
    return C

def shift_lr(C, r):
    return np.roll(C, -1 if r == 0 else 1, axis=1)

def sbox_map(C):
    return bytes_to_bits(SBOX_ARR[bits_to_bytes(C)])

def internal_33(C, Rstar, conv):
    N = len(C); out = np.zeros_like(C)
    for r in (0, 1):
        m = (Rstar == r)
        if m.sum() == 0: continue
        out[m] = shift_lr(sbox_map(flip_oe(C[m], r, conv)), r)
    return out

def internal_32(C, Rstar, conv):
    N = len(C); out = np.zeros_like(C)
    for r in (0, 1):
        m = (Rstar == r)
        if m.sum() == 0: continue
        C1 = flip_oe(C[m], r, conv)
        C2 = shift_lr(C[m], r)
        out[m] = sbox_map(C1 ^ C2)
    return out

MAPS = {'3-3': internal_33, '3-2': internal_32}

def gen(puf, N, rng, scheme, conv):
    C = rng.integers(0, 2, size=(N, 64)).astype(np.uint8)
    Rs = labels(puf, C)
    Ci = MAPS[scheme](C, Rs, conv)
    return C, labels(puf, Ci), Rs

def t2_em(mapfn, conv, Ctr, ytr, Cte, yte, iters=8, seed=0):
    rng = np.random.default_rng(seed)
    asg = rng.integers(0, 2, size=len(Ctr))
    for it in range(iters):
        models = []
        for r in (0, 1):
            m = (asg == r)
            if m.sum() < 100:
                models.append(None); continue
            clf = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
            clf.fit(parity_feats(mapfn(Ctr[m], np.full(m.sum(), r), conv)), ytr[m])
            models.append(clf)
        margins = np.zeros((len(Ctr), 2))
        for r in (0, 1):
            if models[r] is None: margins[:, r] = -np.inf; continue
            margins[:, r] = models[r].decision_function(parity_feats(mapfn(Ctr, np.full(len(Ctr), r), conv)))
        agree = np.sign(margins) == (2*ytr - 1)[:, None]
        better = np.argmax(np.where(agree, np.abs(margins), -1), axis=1)
        neither = ~agree.any(axis=1)
        better[neither] = asg[neither]
        asg = better
    tm = np.zeros((len(Cte), 2))
    for r in (0, 1):
        if models[r] is None: tm[:, r] = -np.inf; continue
        tm[:, r] = models[r].decision_function(parity_feats(mapfn(Cte, np.full(len(Cte), r), conv)))
    pred_u = (tm[np.arange(len(Cte)), np.argmax(np.abs(tm), axis=1)] > 0).astype(int)
    ung = float((pred_u == yte).mean())
    gate = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
    try:
        gate.fit(parity_feats(Ctr), asg)
        gr = gate.predict(parity_feats(Cte))
        gp = np.zeros(len(Cte), dtype=int)
        for r in (0, 1):
            m = (gr == r)
            if m.sum() and models[r] is not None:
                gp[m] = (models[r].decision_function(parity_feats(mapfn(Cte[m], np.full(m.sum(), r), conv))) > 0).astype(int)
        gat = float((gp == yte).mean())
    except Exception:
        gat = ung
    return round(ung, 4), round(gat, 4)

rows = []; t0 = time.time()
NTR, NTE = 50000, 10000
for scheme in ['3-3', '3-2']:
    for conv in ['odd', 'even']:
        for nz in [0.0, 0.1]:
            puf = ArbiterPUF(n=64, seed=400, noisiness=nz)
            Ctr, ytr, _ = gen(puf, NTR, np.random.default_rng(2000), scheme, conv)
            Cte, yte, _ = gen(puf, NTE, np.random.default_rng(3000), scheme, conv)
            c = LogisticRegression(max_iter=1000, solver='liblinear')
            c.fit(Ctr.astype(float), ytr)
            fl = round(float(c.score(Cte.astype(float), yte)), 4)
            rows.append(('E3', f'{scheme}-{conv}-nz={nz}-T0-50k', 0, fl))
            for N in [10000, 20000, 50000]:
                u, g = t2_em(MAPS[scheme], conv, Ctr[:N], ytr[:N], Cte, yte)
                rows.append(('E3', f'{scheme}-{conv}-nz={nz}-EM-{N//1000}k-ung', 0, u))
                rows.append(('E3', f'{scheme}-{conv}-nz={nz}-EM-{N//1000}k-gat', 0, g))
            print(f'E3 {scheme} {conv} nz={nz} T0={fl}', flush=True)
        # noisy ceiling on shared challenges (same weights, clean vs noisy labels)
        pc = ArbiterPUF(n=64, seed=400, noisiness=0.0)
        pn = ArbiterPUF(n=64, seed=400, noisiness=0.1)
        Cc = np.random.default_rng(3000).integers(0, 2, size=(NTE, 64)).astype(np.uint8)
        _, yc, _ = gen(pc, NTE, np.random.default_rng(3000), scheme, conv)
        _, yn, _ = gen(pn, NTE, np.random.default_rng(3000), scheme, conv)
        assert (Cc == np.random.default_rng(3000).integers(0, 2, size=(NTE, 64)).astype(np.uint8)).all()
        ceil = round(float((yc == yn).mean()), 4)
        rows.append(('E3', f'{scheme}-{conv}-ceiling', 0, ceil))
        print(f'E3 {scheme} {conv} ceiling={ceil}', flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-e3npilot.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'instance', 'acc']); w.writerows(rows)
