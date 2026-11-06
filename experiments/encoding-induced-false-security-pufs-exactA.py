"""Exact-spec reruns A (fast): OB-PUF Fig.3 vectors, LFSR-APUF Eq.7 exact, noisy spots.
OB exact: PV1 (1,2,3/010 + SR0101), PV2 (1,2,64/101 + SR1010),
          PV3 (1,63,64/010 + SR1001), PV4 (62,63,64/101 + SR0110), 1-indexed.
LFSR exact: 68-bit external C (n=64+4, block segmentation: C1=C[0:64], C2=C[64:68]),
  Galois LFSR taps [64,63,61,60] (stand-in documented: paper omits taps),
  Shift=C2*Base, Base=10 -> 0..150 clocks, state->APUF.
Noisy: noisiness=0.1 (~4.3% intra-HD, pinned calibration).
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01, nz=0.0):
    return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)

def lr_acc(Xtr, ytr, Xte, yte, seed=0):
    if len(np.unique(ytr)) < 2:
        return float((yte == np.bincount(yte).argmax()).mean())
    c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
    c.fit(Xtr.astype(float), ytr); return float(c.score(Xte.astype(float), yte))

def parity_feats(C):
    S = 1 - 2*C.astype(float); X = np.ones((len(C), 65))
    for i in range(64): X[:, i] = np.prod(S[:, i:], axis=1)
    return X

def load_gate1():
    src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
    ns = {}
    exec(src.split('INSTANCES =')[0], ns)
    return ns
G1 = load_gate1()

# ---- OB exact ----
OB_POS = [[0,1,2],[0,1,63],[0,62,63],[61,62,63]]
OB_VAL = np.array([[0,1,0],[1,0,1],[0,1,0],[1,0,1]], dtype=np.uint8)
OB_SR = np.array([[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,1,0]], dtype=np.uint8)
def ob_full(Cob, i):
    N = len(Cob); C = np.zeros((N, 64), dtype=np.uint8)
    for t in range(N):
        pos = sorted(OB_POS[i]); vals = OB_VAL[i]; src = 0; p = 0
        for dst in range(64):
            if p < 3 and dst == pos[p]: C[t, dst] = vals[p]; p += 1
            else: C[t, dst] = Cob[t, src]; src += 1
    return C

def ob_exact_trial(inst, Ntr, Nte, seed, nz=0.0):
    pufs = [ArbiterPUF(n=64, seed=300+inst*10+j, noisiness=nz) for j in range(4)]
    rng = np.random.default_rng(seed)
    def gen(N, r):
        Cob = r.integers(0, 2, size=(N, 61)).astype(np.uint8)
        ii = r.integers(0, 4, size=N)
        Rob = np.zeros((N, 4), dtype=int)
        for i in range(4):
            m = ii == i
            if m.sum() == 0: continue
            Cf = ob_full(Cob[m], i)
            Rob[m] = np.column_stack([labels(p, Cf) for p in pufs]) ^ OB_SR[i]
        return Cob, Rob
    Ctr, Ytr = gen(Ntr, np.random.default_rng(seed+1))
    Cte, Yte = gen(Nte, np.random.default_rng(seed+2))
    return [lr_acc(Ctr, Ytr[:, j], Cte, Yte[:, j], seed) for j in range(4)]

# ---- LFSR exact ----
def galois_clock(s, clocks, taps=(0, 1, 3, 4)):
    s = s.copy()
    for _ in range(clocks):
        fb = s[-1]
        for t in taps: s[t] ^= fb
        s = np.roll(s, 1); s[0] = fb
    return s

def lfsr_exact_internal(C68, base=10):
    N = len(C68); out = np.zeros((N, 64), dtype=np.uint8)
    for t in range(N):
        C1, C2 = C68[t, :64], C68[t, 64:]
        k = int(C2.dot([8, 4, 2, 1])) * base
        out[t] = galois_clock(C1, k)
    return out

def lfsr_exact_trial(inst, base, Ntr, Nte, seed, nz=0.0):
    puf = ArbiterPUF(n=64, seed=200+inst, noisiness=nz)
    rng = np.random.default_rng(seed)
    Ctr = rng.integers(0, 2, size=(Ntr, 68)).astype(np.uint8)
    Cte = rng.integers(0, 2, size=(Nte, 68)).astype(np.uint8)
    return lr_acc(Ctr, labels(puf, lfsr_exact_internal(Ctr, base)),
                  Cte, labels(puf, lfsr_exact_internal(Cte, base)), seed)

rows = []; t0 = time.time()
for inst in [0, 1]:
    a = ob_exact_trial(inst, 20000, 5000, 7000+inst)
    m = round(float(np.mean(a)), 4)
    rows.append(('OB-PUF-exact', 'Fig3-vectors,N=20k', inst, m))
    print('OB-exact', inst, [round(x, 4) for x in a], 'mean', m, flush=True)
for inst in [0, 1]:
    a = lfsr_exact_trial(inst, 10, 10000, 5000, 6000+inst)
    rows.append(('LFSR-exact', 'Eq7-Base10,N=10k', inst, round(a, 4)))
    print('LFSR-exact', inst, round(a, 4), flush=True)
# noisy spots: Gate1 plain/T0/T1 N=5k,20k nz=0.1, 2 inst x 1 seed
for inst in [0, 1]:
    puf = ArbiterPUF(n=64, seed=inst, noisiness=0.1)
    rt = np.random.default_rng(91000+inst)
    Cte = rt.integers(0, 2, size=(5000, 64)).astype(np.uint8)
    yp = labels(puf, Cte); ys = labels(puf, G1['pi_sbox'](Cte))
    r = np.random.default_rng(10000*inst)
    Ce = r.integers(0, 2, size=(20000, 64)).astype(np.uint8)
    ypp = labels(puf, Ce); yss = labels(puf, G1['pi_sbox'](Ce))
    for N in [5000, 20000]:
        ap = lr_acc(parity_feats(Ce[:N]), ypp[:N], parity_feats(Cte), yp, 0)
        at = lr_acc(Ce[:N].astype(float), yss[:N], Cte.astype(float), ys, 0)
        a1 = lr_acc(parity_feats(G1['pi_sbox'](Ce[:N])), yss[:N], parity_feats(G1['pi_sbox'](Cte)), ys, 0)
        rows.append(('noisy0.1-plain', f'N={N}', inst, round(ap, 4)))
        rows.append(('noisy0.1-T0', f'N={N}', inst, round(at, 4)))
        rows.append(('noisy0.1-T1', f'N={N}', inst, round(a1, 4)))
        print(f'noisy inst{inst} N={N} plain={ap:.3f} T0={at:.3f} T1={a1:.3f}', flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-exactA.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['design', 'config', 'instance', 'acc']); w.writerows(rows)
