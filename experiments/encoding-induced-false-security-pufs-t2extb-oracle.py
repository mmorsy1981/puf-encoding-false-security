"""OB-PUF oracle ceiling + inner-APUF accuracy (Review-3 issues E/F).
(a) Oracle per-bit ceiling on t2ext.py's exact test sets (seeds 7100+inst): knows all 4 APUFs,
    not the per-query pattern. Reports ceiling, empirical argmax, tie fraction, EM/t2ext.csv ratio.
(b) Inner-APUF accuracy (Delvaux-analogous metric): parity-LR per APUF trained on TRUE-pattern
    de-obfuscated challenges (known pattern in sim), scored on held-out. Expect ~1.0 in clean sim.
Writes experiments/encoding-induced-false-security-pufs-t2extb-oracle.csv
cols: instance,metric,value
"""
import csv, importlib.util, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
src = open('experiments/encoding-induced-false-security-pufs-t2ext.py').read().split('rows = []')[0]
ns = {}; exec(src, ns)
class T2: pass
t2 = T2()
for k in ('ob_full', 'ob_gen', 'labels', 'pfeat', 'OB_SR', 'OB_POS', 'OB_VAL'):
    setattr(t2, k, ns[k])

rows = []
for inst in [0, 1, 2]:
    Cte, Yte = t2.ob_gen(inst, 5000, 7100+inst)[0:2]
    pufs = [ArbiterPUF(n=64, seed=300+inst*10+j, noisiness=0.0) for j in range(4)]
    r = np.random.default_rng(7100+inst)
    ii = r.integers(0, 4, size=5000)
    cand = np.stack([np.column_stack([t2.labels(p, t2.ob_full(Cte, i)) for p in pufs]) ^ t2.OB_SR[i] for i in range(4)])
    pr1 = cand.mean(0)
    bayes = float(np.maximum(pr1, 1 - pr1).mean())
    pred = (pr1 > 0.5).astype(int); ties = pr1 == 0.5
    emp = float(np.where(ties, 0.5, (pred == Yte)).mean())
    rows.append((inst, 'oracle-ceiling', round(bayes, 4)))
    rows.append((inst, 'oracle-empirical', round(emp, 4)))
    rows.append((inst, 'tie-fraction', round(float(ties.mean()), 4)))
    print('inst', inst, 'ceiling', round(bayes, 4), 'emp', round(emp, 4), 'ties', round(float(ties.mean()), 4), flush=True)
    # inner-APUF accuracy: own fresh self-consistent draw (true patterns known by construction)
    r3 = np.random.default_rng(9000+inst)
    Cob3 = r3.integers(0, 2, size=(20000, 61)).astype(np.uint8)
    ii3 = r3.integers(0, 4, size=20000)
    Cte3 = r3.integers(0, 2, size=(5000, 61)).astype(np.uint8)
    iite = r3.integers(0, 4, size=5000)
    Ytr3 = np.zeros((20000, 4), dtype=int); Yte3 = np.zeros((5000, 4), dtype=int)
    for i in range(4):
        m = ii3 == i; Ytr3[m] = np.column_stack([t2.labels(p, t2.ob_full(Cob3[m], i)) for p in pufs]) ^ t2.OB_SR[i]
        me = iite == i; Yte3[me] = np.column_stack([t2.labels(p, t2.ob_full(Cte3[me], i)) for p in pufs]) ^ t2.OB_SR[i]
    accs = []
    for j in range(4):
        Xtr = np.vstack([t2.pfeat(t2.ob_full(Cob3[ii3 == i], i)) for i in range(4)])
        ytr = np.concatenate([(Ytr3[ii3 == i, j] ^ t2.OB_SR[i, j]) for i in range(4)])
        Xte = np.vstack([t2.pfeat(t2.ob_full(Cte3[iite == i], i)) for i in range(4)])
        yte = np.concatenate([(Yte3[iite == i, j] ^ t2.OB_SR[i, j]) for i in range(4)])
        c = LogisticRegression(max_iter=1000, solver='liblinear').fit(Xtr, ytr)
        accs.append(round(float(c.score(Xte, yte)), 4))
    rows.append((inst, 'inner-APUF-mean', round(float(np.mean(accs)), 4)))
    print('inst', inst, 'inner-APUF per-PUF', accs, flush=True)
with open('experiments/encoding-induced-false-security-pufs-t2extb-oracle.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['instance', 'metric', 'value']); w.writerows(rows)
