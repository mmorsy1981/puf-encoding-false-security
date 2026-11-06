"""Gate2 T0 replication: RSO, LFSR-APUF, OB-PUF (approximate-but-documented).
Criterion: reproduce reported accuracy within 3pp.
Targets: RSO ~50% (m=32, 1M CRPs, 5 learners); LFSR-APUF 51.79% (Base=10);
OB-PUF Ppred ~52-56% vs Pmin 81.45% (1706.06232 Tables 3-4, CMA-ES; here LR proxy).
Simplifications (full specs need full text; abstract-level replication):
- RSO single-bit: C'=C XOR K[i], obs=APUF(C') XOR b[j], i,j per-query uniform.
- LFSR-APUF: 64-bit LFSR (taps 64,4,3,1,0), clocks=Base+popcount(C)%16, state->APUF.
- OB-PUF: k=64,m=3,p=4,n_ins=4; insertion positions spread; SR balanced; LR per output bit on raw partial.
Learner: sklearn LR liblinear (T0 generic learner on attacker-visible bits).
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

N64 = 64

def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def lr_acc(Xtr, ytr, Xte, yte, seed=0):
    if len(np.unique(ytr)) < 2:
        return float((yte == np.bincount(yte).argmax()).mean())
    c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
    c.fit(Xtr.astype(float), ytr); return float(c.score(Xte.astype(float), yte))

def parity_feats(C):
    S = 1 - 2*C.astype(float); X = np.ones((len(C), 65))
    for i in range(64): X[:, i] = np.prod(S[:, i:], axis=1)
    return X

# ---------- RSO ----------
def rso_trial(inst, m_keys, Ntr, Nte, seed):
    puf = ArbiterPUF(n=64, seed=100+inst, noisiness=0.0)
    rng = np.random.default_rng(seed)
    K = rng.integers(0, 2, size=(m_keys, 64)).astype(np.uint8)
    b = rng.integers(0, 2, size=(m_keys,))
    def gen(N, r):
        C = r.integers(0, 2, size=(N, 64)).astype(np.uint8)
        ii, jj = r.integers(0, m_keys, size=N), r.integers(0, m_keys, size=N)
        obs = np.zeros(N, dtype=int)
        for t in range(N):
            obs[t] = labels(puf, (C[t] ^ K[ii[t]]).reshape(1, -1))[0] ^ b[jj[t]]
        return C, obs
    Ctr, ytr = gen(Ntr, np.random.default_rng(seed+1))
    Cte, yte = gen(Nte, np.random.default_rng(seed+2))
    return lr_acc(Ctr, ytr, Cte, yte, seed)

# ---------- LFSR-APUF (approximate) ----------
TAPS = (64, 4, 3, 1, 0)
# 64-bit Fibonacci LFSR, primitive taps x^64+x^4+x^3+x+1 (approx replication;
# exact LFSR-APUF tap/clock schedule needs full text, documented in report).

def lfsr_internal(C, base):
    N = len(C); out = np.zeros_like(C)
    for t in range(N):
        s = C[t].copy(); k = base + int(C[t].sum()) % 16
        for _ in range(k):
            fb = s[0] ^ s[TAPS[0]-TAPS[1]] ^ s[TAPS[0]-TAPS[2]] ^ s[TAPS[0]-TAPS[3]]
            s = np.roll(s, -1); s[-1] = fb
        out[t] = s
    return out

def lfsr_trial(inst, base, Ntr, Nte, seed):
    puf = ArbiterPUF(n=64, seed=200+inst, noisiness=0.0)
    rng = np.random.default_rng(seed)
    Ctr = rng.integers(0, 2, size=(Ntr, 64)).astype(np.uint8)
    Cte = rng.integers(0, 2, size=(Nte, 64)).astype(np.uint8)
    return lr_acc(Ctr, labels(puf, lfsr_internal(Ctr, base)),
                  Cte, labels(puf, lfsr_internal(Cte, base)), seed)

# ---------- OB-PUF (approximate, (4,4,0)-like, m=3) ----------
# OB-PUF EXACT Fig.3 vectors (Gao et al. 1706.06232), 1-indexed figure -> 0-indexed here.
# PV1 pos 1,2,3 val 0,1,0 SR 0101; PV2 pos 1,2,64 val 1,0,1 SR 1010;
# PV3 pos 1,63,64 val 0,1,0 SR 1001; PV4 pos 62,63,64 val 1,0,1 SR 0110.
OB_POS = [[0, 1, 2], [0, 1, 63], [0, 62, 63], [61, 62, 63]]
OB_VAL = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
OB_SR = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)
def ob_full(Cob, i):
    N = len(Cob); C = np.zeros((N, 64), dtype=np.uint8)
    for t in range(N):
        pos = sorted(OB_POS[i]); vals = OB_VAL[i]; src = 0
        p = 0
        for dst in range(64):
            if p < 3 and dst == pos[p]: C[t, dst] = vals[p]; p += 1
            else: C[t, dst] = Cob[t, src]; src += 1
    return C

def ob_trial(inst, Ntr, Nte, seed):
    pufs = [ArbiterPUF(n=64, seed=300+inst*10+j, noisiness=0.0) for j in range(4)]
    rng = np.random.default_rng(seed)
    def gen(N, r):
        Cob = r.integers(0, 2, size=(N, 61)).astype(np.uint8)
        ii = r.integers(0, 4, size=N)
        Rob = np.zeros((N, 4), dtype=int)
        for i in range(4):
            m = ii == i
            if m.sum() == 0: continue
            Cf = ob_full(Cob[m], i)
            R = np.column_stack([labels(p, Cf) for p in pufs])
            Rob[m] = R ^ OB_SR[i]
        return Cob, Rob
    Ctr, Ytr = gen(Ntr, np.random.default_rng(seed+1))
    Cte, Yte = gen(Nte, np.random.default_rng(seed+2))
    return [lr_acc(Ctr, Ytr[:, j], Cte, Yte[:, j], seed) for j in range(4)]

rows = []; t0 = time.time()
for inst in [0, 1]:
    a = rso_trial(inst, 8, 100000, 10000, 5000+inst)
    rows.append(('RSO', 'm=8,N=100k', inst, round(a, 4), 0.50, round(abs(a-0.50), 4)))
    print('RSO', inst, round(a, 4), flush=True)
for inst in [0, 1]:
    a = lfsr_trial(inst, 10, 10000, 5000, 6000+inst)
    rows.append(('LFSR-APUF', 'Base=10,N=10k', inst, round(a, 4), 0.5179, round(abs(a-0.5179), 4)))
    print('LFSR', inst, round(a, 4), flush=True)
for inst in [0, 1]:
    accs = ob_trial(inst, 20000, 5000, 7000+inst)
    m = float(np.mean(accs))
    rows.append(('OB-PUF', 'N=20k,mean4bits', inst, round(m, 4), 0.54, round(abs(m-0.54), 4)))
    print('OBPUF', inst, [round(x, 4) for x in accs], flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-gate2.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['design', 'config', 'instance', 'acc', 'target', 'gap']); w.writerows(rows)
ok = sum(1 for r in rows if r[5] <= 0.03)
print(f'within-3pp: {ok}/{len(rows)}')
