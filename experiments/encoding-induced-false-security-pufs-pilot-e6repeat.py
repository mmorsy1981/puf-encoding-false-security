"""E6 session-aware repeatability pilot — vetted NARROW (transfer UNVERIFIED).
Question: do same-C_OB repeats cluster into <=G stable groups (a) within a
session and (b) across session boundaries (fresh pattern sets)?
Setup: eta=4 APUFs (fixed weights, seeds 300..303), G=4 patterns/session.
Session model 1 (session-fixed set, per-query uniform draw — the most
attack-favorable reading of "reconfigured per session"): patterns fixed within
a session, device picks index uniformly per evaluation.
Session model 2 (strict one-time pad): fresh random pattern per evaluation.
Patterns: insertion of 3 bits into 61-bit C_OB (gate2 OB_POS convention);
session sets: S1 = gate2 Fig.3 vectors, S2 = fresh random draw (seed 99).
Protocol: Q=20 fixed C_OB x k=4 repeats, sessions {S1, S1-again, S2}.
Metrics per C_OB: distinct response-vectors within-S1, within-S1again,
pooled-S1 (=S1+S1again, same set), pooled-S1+S2 (across sets); same for model 2.
PASS bar (for E6 to proceed): <=G stable groups across session boundaries.
Expected per brief caveat: FAIL -> C3 stays "undecided".
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF

def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)

OB_POS_FIG3 = [[0, 1, 2], [0, 1, 63], [0, 62, 63], [61, 62, 63]]
OB_VAL_FIG3 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
OB_SR_FIG3 = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)

def ob_full(Cob, i, POS, VAL):
    N = len(Cob); C = np.zeros((N, 64), dtype=np.uint8)
    for t in range(N):
        pos = sorted(POS[i]); vals = VAL[i]; src = 0; p = 0
        for dst in range(64):
            if p < 3 and dst == pos[p]: C[t, dst] = vals[p]; p += 1
            else: C[t, dst] = Cob[t, src]; src += 1
    return C

def rand_session(rng):
    POS = [sorted(rng.choice(64, 3, replace=False).tolist()) for _ in range(4)]
    VAL = rng.integers(0, 2, size=(4, 3)).astype(np.uint8)
    SR = rng.integers(0, 2, size=(4, 4)).astype(np.uint8)
    return POS, VAL, SR

def eval_resp(pufs, Cob, i, POS, VAL, SR):
    Cf = ob_full(Cob.reshape(1, -1), i, POS, VAL)
    R = np.array([labels(p, Cf)[0] for p in pufs])
    return tuple((R ^ SR[i]).tolist())

t0 = time.time()
pufs = [ArbiterPUF(n=64, seed=300+j, noisiness=0.0) for j in range(4)]
CobQ = np.random.default_rng(7).integers(0, 2, size=(20, 61)).astype(np.uint8)
S1 = (OB_POS_FIG3, OB_VAL_FIG3, OB_SR_FIG3)
S2 = rand_session(np.random.default_rng(99))
rows = []
for model in ['M1-sess-fixed', 'M2-otp']:
    for sess, S in [('S1a', S1), ('S1b', S1), ('S2', S2)]:
        POS, VAL, SR = S
        r = np.random.default_rng({'M1-sess-fixed-S1a': 11, 'M1-sess-fixed-S1b': 12, 'M1-sess-fixed-S2': 13, 'M2-otp-S1a': 21, 'M2-otp-S1b': 22, 'M2-otp-S2': 23}[(model + '-' + sess)])
        for q in range(20):
            seen = set()
            for rep in range(4):
                if model == 'M1-sess-fixed':
                    i = r.integers(0, 4)
                else:
                    Pi, Vi, Si = rand_session(r)
                    i = 0; PSc, VSc, SSc = [Pi[0]], np.array([Vi[0]]), np.array([Si[0]])
                    seen.add(eval_resp(pufs, CobQ[q], 0, PSc, VSc, SSc)); continue
                seen.add(eval_resp(pufs, CobQ[q], i, POS, VAL, SR))
            rows.append(('E6', f'{model}-{sess}', q, len(seen)))
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-e6repeat.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'challenge', 'distinct4']); w.writerows(rows)
# summary
import collections
for model in ['M1-sess-fixed', 'M2-otp']:
    for sess in ['S1a', 'S1b', 'S2']:
        vals = [r[3] for r in rows if r[1] == f'{model}-{sess}']
        print(model, sess, 'mean-distinct-per-C:', round(float(np.mean(vals)), 2), 'max:', max(vals), flush=True)
