"""Fine grid N=100..1000 (Review-3 issue B): resolve the censored 1.0x cost ratio.
Tiers: plain-parity, sbox-piparity-T1 (+ sbox-raw-T0 to document T0 censoring).
5 inst x 2 seeds (10 configs), held-out 5000. CSV cols: instance,aseed,N,tier,acc.
Then print interpolated N-to-90/95% per config (linear interp on acc curve; None if never).
"""
import time, csv, types, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read().split(chr(10) + "if __name__ == '__main__':")[0]
ns = {}; exec(src, ns)
g1 = types.SimpleNamespace(**{k: ns[k] for k in ('pi_sbox', 'parity_feats', 'labels')})

def run_lr(Xtr, ytr, Xte, yte, seed):
    if len(np.unique(ytr)) < 2:
        maj = np.bincount(yte).argmax()
        return float((yte == maj).mean())
    c = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear', random_state=seed)
    c.fit(Xtr, ytr)
    return float(c.score(Xte, yte))

INSTANCES = [0, 1, 2, 3, 4]; ASEEDS = [0, 1]
NGRID = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]; MTEST = 5000
rows = []; t0 = time.time()
for inst in INSTANCES:
    puf = ArbiterPUF(n=64, seed=inst, noisiness=0.0)
    rng_test = np.random.default_rng(90000+inst)
    Ctest_ext = rng_test.integers(0, 2, size=(MTEST, 64)).astype(np.uint8)
    Ctest_int = g1.pi_sbox(Ctest_ext)
    ytest_plain = g1.labels(puf, Ctest_ext)
    ytest_sbox = g1.labels(puf, Ctest_int)
    for asd in ASEEDS:
        rng = np.random.default_rng(10000*inst+asd)
        Ctr_ext = rng.integers(0, 2, size=(1000, 64)).astype(np.uint8)
        Ctr_int = g1.pi_sbox(Ctr_ext)
        ytr_plain = g1.labels(puf, Ctr_ext)
        ytr_sbox = g1.labels(puf, Ctr_int)
        for N in NGRID:
            Ce, Ci = Ctr_ext[:N], Ctr_int[:N]
            a = run_lr(g1.parity_feats(Ce), ytr_plain[:N], g1.parity_feats(Ctest_ext), ytest_plain, asd)
            rows.append((inst, asd, N, 'plain-parity', round(a, 4)))
            b = run_lr(Ce.astype(float), ytr_sbox[:N], Ctest_ext.astype(float), ytest_sbox, asd)
            rows.append((inst, asd, N, 'sbox-raw-T0', round(b, 4)))
            c = run_lr(g1.parity_feats(Ci), ytr_sbox[:N], g1.parity_feats(Ctest_int), ytest_sbox, asd)
            rows.append((inst, asd, N, 'sbox-piparity-T1', round(c, 4)))
    print('inst done', inst, flush=True)
with open('experiments/encoding-induced-false-security-pufs-gate1c-finegrid.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['instance', 'aseed', 'N', 'tier', 'acc']); w.writerows(rows)
print('elapsed', round(time.time()-t0, 1))
from collections import defaultdict
d = defaultdict(dict)
for inst, asd, N, tier, a in rows: d[(tier, inst, asd)][N] = a
for tier in ('plain-parity', 'sbox-piparity-T1', 'sbox-raw-T0'):
    hits90, hits95 = [], []
    for inst in INSTANCES:
        for asd in ASEEDS:
            curve = d[(tier, inst, asd)]
            xs = sorted(curve)
            for thr, lst in ((0.90, hits90), (0.95, hits95)):
                n2 = None
                for i, x in enumerate(xs):
                    if curve[x] >= thr:
                        n2 = float(x) if i == 0 else xs[i-1] + (thr - curve[xs[i-1]]) / (curve[x] - curve[xs[i-1]]) * (x - xs[i-1])
                        break
                lst.append(n2)
    print(tier, 'N-to-90:', ['%.0f' % v if v else None for v in hits90], 'N-to-95:', ['%.0f' % v if v else None for v in hits95])
