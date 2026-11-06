"""R2: ROC-grid T0 arm (audit residual). Raw-bit LR per grid cell with the exact
seeds/N of pilot-e3grid.py (puf 400+inst; train rng 2000+100*inst+10*s 20k;
test rng 3000+100*inst+10*s 10k; nz per config). 8 configs x 10 inst x 3 seeds.
Writes experiments/encoding-induced-false-security-pufs-pilot-e3grid-t0.csv
cols: config,instance,seed,acc_t0
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
rows = []; t0 = time.time()
for inst in range(10):
    for s in [0, 1, 2]:
        for scheme in ['3-3', '3-2']:
            for conv in ['odd', 'even']:
                for nz in [0.0, 0.1]:
                    puf = ArbiterPUF(n=64, seed=400+inst, noisiness=nz)
                    Ctr, ytr = gen(puf, 20000, np.random.default_rng(2000+100*inst+10*s), scheme, conv)
                    Cte, yte = gen(puf, 10000, np.random.default_rng(3000+100*inst+10*s), scheme, conv)
                    c = LogisticRegression(max_iter=1000, solver='liblinear').fit(2*Ctr.astype(float)-1, ytr)
                    a = round(float(c.score(2*Cte.astype(float)-1, yte)), 4)
                    rows.append((f'{scheme}-{conv}-nz={nz}-EM-20k', inst, s, a))
    print('inst', inst, 'done', round(time.time()-t0, 1), flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-e3grid-t0.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['config', 'instance', 'seed', 'acc_t0']); w.writerows(rows)
from collections import defaultdict
d = defaultdict(list)
for c, i, s_, v in rows: d[c].append(v)
for k in sorted(d): print(k, 'mean', round(float(np.mean(d[k])), 4), 'range', (min(d[k]), max(d[k])))
