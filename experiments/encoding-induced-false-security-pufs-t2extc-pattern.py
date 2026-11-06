"""OB-PUF pattern-recovery rate (Review v2 §6.5): break-relevant metric.
Reruns t2ext.py EM training (same seeds/instances), then scores (a) final expert
assignment vs TRUE pattern, (b) gating-LR prediction vs true pattern on test —
i.e., how often the attacker recovers the latent pattern (Delvaux-relevant quantity).
Writes experiments/encoding-induced-false-security-pufs-t2extc-pattern.csv
cols: instance,metric,value
"""
import csv, numpy as np
src = open('experiments/encoding-induced-false-security-pufs-t2ext.py').read().split('rows = []')[0]
ns = {}; exec(src, ns)
from sklearn.linear_model import LogisticRegression
rows = []
for inst in [0, 1, 2]:
    Ctr, Ytr = ns['ob_gen'](inst, 20000, 7000+inst)[0:2]
    Cte, Yte = ns['ob_gen'](inst, 5000, 7100+inst)[0:2]
    r = np.random.default_rng(7100+inst); iite = r.integers(0, 4, size=5000)
    # replicate EM training to recover assignments (same code path as ob_em, capture asg+gate)
    rng = np.random.default_rng(0); asg = rng.integers(0, 4, size=len(Ctr))
    for it in range(5):
        experts = []
        for i in range(4):
            m = (asg == i)
            if m.sum() < 200: experts.append(None); continue
            experts.append([ns['fit_lr'](ns['pfeat'](ns['ob_full'](Ctr[m], i)), Ytr[m, j], 0) for j in range(4)])
        agree = np.zeros((len(Ctr), 4))
        for i in range(4):
            if experts[i] is None: continue
            Fi = ns['ob_full'](Ctr, i)
            for j in range(4):
                agree[:, i] += (experts[i][j].predict(ns['pfeat'](Fi)) == Ytr[:, j])
        new = np.argmax(agree, axis=1)
        new[agree.max(axis=1) == 0] = asg[agree.max(axis=1) == 0]
        asg = new
        gate = ns['fit_lr'](ns['pfeat'](Ctr, 61), asg, 0)
    gr = gate.predict(ns['pfeat'](Cte, 61))
    rtr = np.random.default_rng(7000+inst); _ = rtr.integers(0, 2, size=(20000, 61)); iitrue = rtr.integers(0, 4, size=20000)
    rows.append((inst, 'pattern-recovery-train', round(float((asg == iitrue).mean()), 4)))
    import itertools
    best = max(float(((np.array(p)[gr]) == iite).mean()) for p in itertools.permutations(range(4)))
    rows.append((inst, 'gating-vs-true-pattern', round(float((gr == iite).mean()), 4)))
    rows.append((inst, 'gating-perm-matched', round(best, 4)))
    print('inst', inst, 'gate-vs-true', round(float((gr == iite).mean()), 4), flush=True)
with open('experiments/encoding-induced-false-security-pufs-t2extc-pattern.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['instance', 'metric', 'value']); w.writerows(rows)
