"""Learned inner-APUF accuracy (V4 deferred item): can the attacker learn the four
inner APUFs blind? Reuses t2ext.py EM (4 experts x 4 bits, 5 iters, seed 0) on
20k train / 5k test (same seeds as t2ext), then Hungarian-remaps expert ids to
true pattern ids (train agreement) and evaluates per-expert per-bit accuracy on
test samples carrying the TRUE remapped pattern. Reports mean over experts/bits
(= learned inner-APUF accuracy) per instance. Compares against the oracle
ceiling (true patterns known) and the chance level for a single APUF model.
Writes experiments/encoding-induced-false-security-pufs-pilot-obinner.csv
cols: instance, learned_inner_acc, oracle_inner_acc, n_train.
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
try:
    from scipy.optimize import linear_sum_assignment as _lsa
    _HUNG = True
except Exception:
    _HUNG = False

src = open('experiments/encoding-induced-false-security-pufs-t2ext.py').read().split("rows = []; t0 = time.time()")[0]
ns = {}
exec(src, ns)
ob_gen, pfeat, ob_full, fit_lr = ns['ob_gen'], ns['pfeat'], ns['ob_full'], ns['fit_lr']

def best_perm(pred, true, k=4):
    C = np.zeros((k, k), dtype=int)
    for a in range(k):
        for b in range(k):
            C[a, b] = int((((pred == a) & (true == b)).sum()))
    if _HUNG:
        ri, ci = _lsa(-C)
        perm = [None]*k
        for r_, c_ in zip(ri, ci): perm[r_] = c_
        return [int(x) for x in perm]
    import itertools
    best, bestp = -1, None
    for p in itertools.permutations(range(k)):
        acc = float(((np.array(p)[pred]) == true).mean())
        if acc > best: best, bestp = acc, p
    return [int(x) for x in bestp]

rows = []; t0 = time.time()
for inst in [0, 1, 2]:
    pufs = [ArbiterPUF(n=64, seed=300+inst*10+j, noisiness=0.0) for j in range(4)]
    Ctr, Ytr = ns['ob_gen'](inst, 20000, 7000+inst)[0:2]
    Cte, Yte = ns['ob_gen'](inst, 5000, 7100+inst)[0:2]
    r = np.random.default_rng(7100+inst); _ = r.integers(0, 2, size=(5000, 61)); iite = r.integers(0, 4, size=5000)
    rtr = np.random.default_rng(7000+inst); _ = rtr.integers(0, 2, size=(20000, 61)); iitrue = rtr.integers(0, 4, size=20000)
    rng = np.random.default_rng(0); asg = rng.integers(0, 4, size=len(Ctr))
    for it in range(5):
        experts = []
        for i in range(4):
            m = (asg == i)
            if m.sum() < 200: experts.append(None); continue
            experts.append([fit_lr(pfeat(ob_full(Ctr[m], i)), Ytr[m, j], 0) for j in range(4)])
        agree = np.zeros((len(Ctr), 4))
        for i in range(4):
            if experts[i] is None: continue
            Fi = ob_full(Ctr, i)
            for j in range(4):
                agree[:, i] += (experts[i][j].predict(pfeat(Fi)) == Ytr[:, j])
        asg = agree.argmax(1)
    perm = best_perm(asg, iitrue)
    accs = []
    for e in range(4):
        p = perm[e]
        if experts[e] is None: continue
        m = (iite == p)
        if m.sum() == 0: continue
        Fe = ob_full(Cte[m], p)
        hit = np.mean([experts[e][j].predict(pfeat(Fe)) == Yte[m, j] for j in range(4)])
        accs.append(float(hit))
    learned = round(float(np.mean(accs)), 4) if accs else float('nan')
    orow = [float(ns['fit_lr'](ns['pfeat'](ns['ob_full'](Ctr[iitrue == i], i)), Ytr[iitrue == i, j], 0).score(ns['pfeat'](ns['ob_full'](Cte[iite == i], i)), Yte[iite == i, j])) for i in range(4) for j in range(4)]
    oracle = round(float(np.mean(orow)), 4)
    rows.append(('OBinner', inst, learned, oracle, len(Ctr)))
    print(f'OBinner inst{inst} learned={learned} oracle={oracle}', flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-obinner.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'instance', 'learned_inner_acc', 'oracle_inner_acc', 'n_train']); w.writerows(rows)
