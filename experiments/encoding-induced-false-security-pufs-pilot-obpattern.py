"""OB pattern-recovery FIX (V4-fix slice 4).
Fixes t2extc-pattern.py line-16 bug: test true patterns `iite` were drawn from
fresh default_rng(7100+inst) WITHOUT the Cob (5000,61) dummy draw that ob_gen
performs, so labels misalign. Fix adds the dummy draw so labels align
(equivalently, reuse ob_gen's internal ii). Train true `iitrue` already had the
dummy draw and is unchanged.
Regenerates corrected metrics to NEW file
pilot-obpattern-fix.csv (original t2extc-pattern.csv untouched).
Adds permutation-matched scores via Hungarian (scipy if available, else brute
4! which is exact Hungarian for 4 clusters): best remap of EM cluster ids to
true patterns, then accuracy — train (asg vs iitrue) and test gating (gr vs
iite). Raw (unpermuted) scores also filed for reference.
Original EM replication kept byte-identical (5 iters, seed 0, fit_lr liblinear).
Writes experiments/encoding-induced-false-security-pufs-pilot-obpattern-fix.csv
cols: instance,metric,value
"""
import csv, itertools, numpy as np
src = open('experiments/encoding-induced-false-security-pufs-t2ext.py').read().split('rows = []')[0]
ns = {}; exec(src, ns)
from sklearn.linear_model import LogisticRegression
try:
    from scipy.optimize import linear_sum_assignment as _lsa
    _HAS_HUNG = True
except Exception:
    _HAS_HUNG = False

def best_perm_acc(pred, true, k=4):
    """Hungarian (max agreement) remap of pred ids to true ids, then accuracy."""
    C = np.zeros((k, k), dtype=int)
    for a in range(k):
        for b in range(k):
            C[a, b] = int((((pred == a) & (true == b)).sum()))
    if _HAS_HUNG:
        ri, ci = _lsa(-C)
        perm = [None]*k
        for r_, c_ in zip(ri, ci):
            perm[r_] = c_
        mapped = np.array(perm, dtype=int)[pred]
        return float((mapped == true).mean()), tuple(int(x) for x in perm)
    best = -1; bestp = None
    for p in itertools.permutations(range(k)):
        acc = float(((np.array(p)[pred]) == true).mean())
        if acc > best: best, bestp = acc, p
    return best, tuple(int(x) for x in bestp)

rows = []
for inst in [0, 1, 2]:
    Ctr, Ytr = ns['ob_gen'](inst, 20000, 7000+inst)[0:2]
    Cte, Yte = ns['ob_gen'](inst, 5000, 7100+inst)[0:2]
    # FIXED test truth: replicate ob_gen's draw order (Cob dummy then ii)
    r = np.random.default_rng(7100+inst)
    _ = r.integers(0, 2, size=(5000, 61))
    iite = r.integers(0, 4, size=5000)
    # train truth (already correct in original): Cob dummy then ii
    rtr = np.random.default_rng(7000+inst)
    _ = rtr.integers(0, 2, size=(20000, 61))
    iitrue = rtr.integers(0, 4, size=20000)
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
    raw_train = round(float((asg == iitrue).mean()), 4)
    perm_train, ptrain = best_perm_acc(asg, iitrue, 4)
    perm_train = round(float(perm_train), 4)
    raw_gate = round(float((gr == iite).mean()), 4)
    perm_gate, pgate = best_perm_acc(gr, iite, 4)
    perm_gate = round(float(perm_gate), 4)
    rows.append((inst, 'pattern-recovery-train-raw', raw_train))
    rows.append((inst, 'pattern-recovery-train-perm-matched', perm_train))
    rows.append((inst, 'gating-vs-true-pattern-raw', raw_gate))
    rows.append((inst, 'gating-perm-matched', perm_gate))
    print(f'inst {inst} train raw {raw_train} perm {perm_train} {ptrain} | gate raw {raw_gate} perm {perm_gate} {pgate} hungarian={_HAS_HUNG}', flush=True)
with open('experiments/encoding-induced-false-security-pufs-pilot-obpattern-fix.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['instance', 'metric', 'value']); w.writerows(rows)
