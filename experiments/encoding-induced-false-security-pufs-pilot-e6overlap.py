"""E6 cross-session transfer test (companion to pilot-e6repeat.py).
Same seeds/weights -> deterministic. Per C_OB response SETS under M1:
Jaccard(S1a,S1b) [same pattern set] vs Jaccard(S1a,S2) [fresh pattern set].
Writes one row per C_OB + summary to pilot-e6overlap.csv.
"""
import csv, time, numpy as np
from pypuf.simulation import ArbiterPUF

ns = {}
exec(open('experiments/encoding-induced-false-security-pufs-pilot-e6repeat.py').read().split("t0 = time.time()")[0], ns)
t0 = time.time()
pufs = [ArbiterPUF(n=64, seed=300+j, noisiness=0.0) for j in range(4)]
CobQ = np.random.default_rng(7).integers(0, 2, size=(20, 61)).astype(np.uint8)
S1 = (ns['OB_POS_FIG3'], ns['OB_VAL_FIG3'], ns['OB_SR_FIG3'])
S2 = ns['rand_session'](np.random.default_rng(99))
seeds = {'S1a': 11, 'S1b': 12, 'S2': 13}
sets = {}
for sess, S in [('S1a', S1), ('S1b', S1), ('S2', S2)]:
    POS, VAL, SR = S
    r = np.random.default_rng(seeds[sess])
    for q in range(20):
        s = set()
        for rep in range(4):
            i = r.integers(0, 4)
            s.add(ns['eval_resp'](pufs, CobQ[q], i, POS, VAL, SR))
        sets[(sess, q)] = s
rows = []
for q in range(20):
    a, b, c = sets[('S1a', q)], sets[('S1b', q)], sets[('S2', q)]
    jsame = round(len(a & b)/max(1, len(a | b)), 4)
    jcross = round(len(a & c)/max(1, len(a | c)), 4)
    rows.append(('E6', 'overlap', q, jsame, jcross))
print('mean same-set Jaccard:', round(float(np.mean([r[3] for r in rows])), 4), flush=True)
print('mean cross-set Jaccard:', round(float(np.mean([r[4] for r in rows])), 4), flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-e6overlap.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'challenge', 'jacc_same', 'jacc_cross']); w.writerows(rows)
