"""Gate1 control rows (Review-3 issue A): the missing cells that hold the feature map fixed.
Tiers: plain-APUF raw-bit LR; S-box external-parity LR (parity feats on EXTERNAL bits, no inversion).
Same rig as gate1.py: 5 inst x 2 seeds, NGRID={1000,5000,20000} (table cols), held-out 5000.
Writes experiments/encoding-induced-false-security-pufs-gate1b-controls.csv
cols: instance,aseed,N,tier,acc
"""
import time, csv, numpy as np, types
src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read().split(chr(10) + "if __name__ == '__main__':")[0]
ns = {}; exec(src, ns)
g1 = types.SimpleNamespace(**{k: ns[k] for k in ('SBOX_ARR', 'pi_sbox', 'pi_inv', 'parity_feats', 'to_pm', 'labels')})
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

def run_lr(Xtr, ytr, Xte, yte, seed):
    if len(np.unique(ytr)) < 2:
        maj = np.bincount(yte).argmax()
        return float((yte == maj).mean())
    c = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear', random_state=seed)
    c.fit(Xtr, ytr)
    return float(c.score(Xte, yte))

INSTANCES = [0, 1, 2, 3, 4]; ASEEDS = [0, 1]; NGRID = [1000, 5000, 20000]; MTEST = 5000
rows = []; t0 = time.time()
for inst in INSTANCES:
    puf = ArbiterPUF(n=64, seed=inst, noisiness=0.0)
    rng_test = np.random.default_rng(90000+inst)
    Ctest_ext = rng_test.integers(0, 2, size=(MTEST, 64)).astype(np.uint8)
    ytest_plain = g1.labels(puf, Ctest_ext)
    ytest_sbox = g1.labels(puf, g1.pi_sbox(Ctest_ext))
    for asd in ASEEDS:
        rng = np.random.default_rng(10000*inst+asd)
        Ctr_ext = rng.integers(0, 2, size=(max(NGRID), 64)).astype(np.uint8)
        ytr_plain = g1.labels(puf, Ctr_ext)
        ytr_sbox = g1.labels(puf, g1.pi_sbox(Ctr_ext))
        for N in NGRID:
            Ce = Ctr_ext[:N]
            a = run_lr(Ce.astype(float), ytr_plain[:N], Ctest_ext.astype(float), ytest_plain, asd)
            rows.append((inst, asd, N, 'plain-raw-T0control', round(a, 4)))
            b = run_lr(g1.parity_feats(Ce), ytr_sbox[:N],
                       g1.parity_feats(Ctest_ext), ytest_sbox, asd)
            rows.append((inst, asd, N, 'sbox-external-parity-noinv', round(b, 4)))
    print('inst done', inst, flush=True)
with open('experiments/encoding-induced-false-security-pufs-gate1b-controls.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['instance', 'aseed', 'N', 'tier', 'acc']); w.writerows(rows)
print('elapsed', round(time.time()-t0, 1))
from collections import defaultdict
d = defaultdict(list)
for inst, asd, N, tier, a in rows: d[(tier, N)].append(a)
for k in sorted(d): print(k, 'mean', round(float(np.mean(d[k])), 4), 'range', (min(d[k]), max(d[k])))
