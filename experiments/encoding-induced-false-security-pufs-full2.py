"""Full repeats 10x3, part 2 (self-contained): RSO-m32/1M, OB-Fig3, LFSR-Eq7, noisy spots."""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

t0 = time.time()
def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def pfeat(C):
    S = 1 - 2*C.astype(float); X = np.ones((len(C), 65))
    for i in range(64): X[:, i] = np.prod(S[:, i:], axis=1)
    return X
def lr_acc(Xtr, ytr, Xte, yte, seed=0):
    if len(np.unique(ytr)) < 2:
        return float((yte == np.bincount(yte).argmax()).mean())
    c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
    c.fit(Xtr.astype(float), ytr); return float(c.score(Xte.astype(float), yte))
def emit(row):
    with open('experiments/encoding-induced-false-security-pufs-fullB.csv', 'a', newline='') as f:
        csv.writer(f).writerow(row)

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
def ob_trial(inst, Ntr, Nte, seed, nz=0.0):
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
def galois_clock(s, clocks, taps=(0, 1, 3, 4)):
    s = s.copy()
    for _ in range(clocks):
        fb = s[-1]
        for t in taps: s[t] ^= fb
        s = np.roll(s, 1); s[0] = fb
    return s
def lfsr_internal(C68, base=10):
    N = len(C68); out = np.zeros((N, 64), dtype=np.uint8)
    for t in range(N):
        k = int(C68[t, 64:].dot([8, 4, 2, 1])) * base
        out[t] = galois_clock(C68[t, :64], k)
    return out
def lfsr_trial(inst, base, Ntr, Nte, seed, nz=0.0):
    puf = ArbiterPUF(n=64, seed=200+inst, noisiness=nz)
    rng = np.random.default_rng(seed)
    Ctr = rng.integers(0, 2, size=(Ntr, 68)).astype(np.uint8)
    Cte = rng.integers(0, 2, size=(Nte, 68)).astype(np.uint8)
    return lr_acc(Ctr, labels(puf, lfsr_internal(Ctr, base)), Cte, labels(puf, lfsr_internal(Cte, base)), seed)

with open('experiments/encoding-induced-false-security-pufs-fullB.csv', 'w', newline='') as f:
    csv.writer(f).writerow(['section', 'design', 'config', 'instance', 'seed', 'acc'])

for inst in range(10):
    puf = ArbiterPUF(n=64, seed=100+inst, noisiness=0.0)
    rng = np.random.default_rng(5000+inst)
    K = rng.integers(0, 2, size=(32, 64)).astype(np.uint8); b = rng.integers(0, 2, size=(32,))
    rtr = np.random.default_rng(5001+inst); rte = np.random.default_rng(5002+inst)
    Ctr = rtr.integers(0, 2, size=(1000000, 64)).astype(np.uint8)
    ii, jj = rtr.integers(0, 32, size=1000000), rtr.integers(0, 32, size=1000000)
    ytr = (labels(puf, Ctr ^ K[ii]) ^ b[jj]).astype(int)
    Cte = rte.integers(0, 2, size=(10000, 64)).astype(np.uint8)
    iii, jjj = rte.integers(0, 32, size=10000), rte.integers(0, 32, size=10000)
    yte = (labels(puf, Cte ^ K[iii]) ^ b[jjj]).astype(int)
    a = lr_acc(Ctr, ytr, Cte, yte, 0)
    emit(['S3', 'RSO-m32-1M', 'N=1M', inst, 0, round(a, 4)])
    print('RSO', inst, round(a, 4), round(time.time()-t0, 1), flush=True)
    del Ctr, ytr, ii, jj
print('RSO done', round(time.time()-t0, 1), flush=True)
for inst in range(10):
    for asd in [0, 1, 2]:
        accs = ob_trial(inst, 20000, 5000, 7000+inst*10+asd)
        m = round(float(np.mean(accs)), 4)
        emit(['S3', 'OB-Fig3', 'N=20k-mean4', inst, asd, m])
    print('OB', inst, round(time.time()-t0, 1), flush=True)
for inst in range(10):
    for asd in [0, 1, 2]:
        a = lfsr_trial(inst, 10, 10000, 5000, 6000+inst*10+asd)
        emit(['S3', 'LFSR-Eq7', 'Base10-N=10k', inst, asd, round(a, 4)])
    print('LFSR', inst, round(time.time()-t0, 1), flush=True)
for inst in [0, 1, 2]:
    accs = ob_trial(inst, 20000, 5000, 7100+inst, nz=0.1)
    m = round(float(np.mean(accs)), 4)
    emit(['S4', 'OB-noisy01', 'N=20k-mean4', inst, 0, m])
    a = lfsr_trial(inst, 10, 10000, 5000, 7200+inst, nz=0.1)
    emit(['S4', 'LFSR-noisy01', 'Base10-N=10k', inst, 0, round(a, 4)])
    print('noisy inst', inst, 'OB', m, 'LFSR', round(a, 4), flush=True)
print('elapsed', round(time.time()-t0, 1), flush=True)
