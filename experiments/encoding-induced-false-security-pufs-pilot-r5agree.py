"""R5 branch-agreement filing (V4-fix slice 7).
For ROC3 3-3 odd convention (roc3.py: 1-indexed odd bits = 0-indexed 0,2,4...;
flip odd/even by R*, AES S-box per byte, cyclic shift L/R by 1 by R*),
computes branch-agreement fraction P(f0(C)==f1(C)) where f_r(C)=APUF(C*_r) is
the branch-r internal response, inst 0-1 (roc3.py gen, N=10k test seeds
3000+inst, puf seeds 400+inst, noiseless), the random-selector ceiling
0.5+0.5*agree, and binomial SEs.
Writes experiments/encoding-induced-false-security-pufs-pilot-r5agree.csv
cols: instance,N,agree,ceiling,se_agree,se_ceiling_propagated,se_ceiling_binomial
SEs: se_agree=sqrt(a(1-a)/N); se_ceiling_propagated=0.5*se_agree;
se_ceiling_binomial=sqrt(c(1-c)/N) with c=ceiling.
"""
import csv, numpy as np
from pypuf.simulation import ArbiterPUF

def load_lib():
    src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
    ns = {}
    exec(src.split(chr(10) + "if __name__ == '__main__':")[0], ns)
    return ns
L = load_lib()
SBOX_ARR = L['SBOX_ARR']

def flip_oe(C, r):
    C = C.copy()
    idx = np.arange(0, 64, 2) if r == 0 else np.arange(1, 64, 2)
    C[:, idx] ^= 1
    return C

def shift_lr(C, r):
    return np.roll(C, -1 if r == 0 else 1, axis=1)

def roc3_internal(C, Rstar):
    N = len(C); out = np.zeros_like(C)
    for r in (0, 1):
        m = (Rstar == r)
        if m.sum() == 0: continue
        C1 = flip_oe(C[m], r)
        B = L['bits_to_bytes'](C1)
        C2 = L['bytes_to_bits'](SBOX_ARR[B])
        out[m] = shift_lr(C2, r)
    return out

def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)

rows = []
N = 10000
for inst in [0, 1]:
    puf = ArbiterPUF(n=64, seed=400+inst, noisiness=0.0)
    C = np.random.default_rng(3000+inst).integers(0, 2, size=(N, 64)).astype(np.uint8)
    C0 = roc3_internal(C, np.zeros(N, dtype=int))
    C1 = roc3_internal(C, np.ones(N, dtype=int))
    f0 = labels(puf, C0); f1 = labels(puf, C1)
    agree = float((f0 == f1).mean())
    ceiling = 0.5 + 0.5*agree
    se_agree = float(np.sqrt(agree*(1-agree)/N))
    se_prop = 0.5*se_agree
    se_bin = float(np.sqrt(ceiling*(1-ceiling)/N))
    rows.append((inst, N, round(agree, 4), round(ceiling, 4), round(se_agree, 5), round(se_prop, 5), round(se_bin, 5)))
    print(f'inst{inst} agree={agree:.4f} ceiling={ceiling:.4f} se_agree={se_agree:.5f}', flush=True)
with open('experiments/encoding-induced-false-security-pufs-pilot-r5agree.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['instance', 'N', 'agree', 'ceiling', 'se_agree', 'se_ceiling_propagated', 'se_ceiling_binomial'])
    w.writerows(rows)
