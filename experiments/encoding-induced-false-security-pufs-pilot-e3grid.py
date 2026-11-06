"""E3 10x3 grid (V4 submission gate): EM-20k gated+ungated for all 8
scheme x convention x noise configs, 10 instances x 3 data seeds.
Schemes/conventions identical to pilot-e3npilot.py (3-2 extracted first-hand
from MEJO p.5 S3.1; 3-3 as roc3.py). Nested data: puf seed 400+inst (roc3
convention); train rng 2000+100*inst+10*s; test rng 3000+100*inst+10*s, so
(inst,seed=0) reproduces the pilot cells exactly (rig check).
T0 floors already established (0.495-0.512 + XGB) and not rerun here.
Usage: pilot-e3grid.py STARTINST ENDINST  (append-mode CSV).
"""
import time, csv, os, sys, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

CSV = 'experiments/encoding-induced-false-security-pufs-pilot-e3grid.csv'

def load_gate1():
    src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
    ns = {}
    exec(src.split(chr(10) + "if __name__ == '__main__':")[0], ns)
    return ns
G = load_gate1()
SBOX_ARR, parity_feats, to_pm, labels = G['SBOX_ARR'], G['parity_feats'], G['to_pm'], G['labels']
bits_to_bytes, bytes_to_bits = G['bits_to_bytes'], G['bytes_to_bits']
def sbox_map(C): return bytes_to_bits(SBOX_ARR[bits_to_bytes(C)])

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
        out[m] = sbox_map(flip_oe(C[m], r, conv) ^ shift_lr(C[m], r))
    return out

MAPS = {'3-3': internal_33, '3-2': internal_32}

def gen(puf, N, rng, scheme, conv):
    C = rng.integers(0, 2, size=(N, 64)).astype(np.uint8)
    Rs = labels(puf, C)
    Ci = MAPS[scheme](C, Rs, conv)
    return C, labels(puf, Ci)

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

t0 = time.time()
s0, s1 = int(sys.argv[1]), int(sys.argv[2])
newfile = not os.path.exists(CSV)
f = open(CSV, 'a', newline='')
w = csv.writer(f)
if newfile: w.writerow(['test', 'config', 'instance', 'seed', 'acc_gat', 'acc_ung'])
for inst in range(s0, s1):
    for s in [0, 1, 2]:
        for scheme in ['3-3', '3-2']:
            for conv in ['odd', 'even']:
                for nz in [0.0, 0.1]:
                    puf = ArbiterPUF(n=64, seed=400+inst, noisiness=nz)
                    Ctr, ytr = gen(puf, 20000, np.random.default_rng(2000+100*inst+10*s), scheme, conv)[:2]
                    Cte, yte = gen(puf, 10000, np.random.default_rng(3000+100*inst+10*s), scheme, conv)[:2]
                    u, g = t2_em(MAPS[scheme], conv, Ctr, ytr, Cte, yte)
                    w.writerow(('E3grid', f'{scheme}-{conv}-nz={nz}-EM-20k', inst, s, g, u)); f.flush()
                    print(f'E3grid {scheme} {conv} nz={nz} inst={inst} seed={s} gat={g} ung={u}', flush=True)
f.close()
print('elapsed', round(time.time()-t0, 1))
