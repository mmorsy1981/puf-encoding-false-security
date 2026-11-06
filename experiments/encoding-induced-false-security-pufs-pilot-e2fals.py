"""E2 falsifier pair for ROC3-SPUF (scheme 3-3) — vetted pilot, run first.
E2-1 (gate learnability): supervised LR parity(C)->Rs (true R* from gen) at N=20k
  (same N as composite claim 0.9633/0.9517). PASS bar: clearly learnable (~0.9+).
  KILL: gate ~0.5 yet composite breaks -> claimed mechanism falsified.
E2-2 (random-bit control): generation-time swap Rs->u~Bernoulli(0.5); identical T2
  EM (8 iters, 20k train, gated+ungated decode). Expect collapse to mixture
  ceiling. KILL: no collapse -> mixture model wrong.
Reference floor: single-expert T0 (raw LR) on the random-bit data.
Harness: reuses roc3.py internals via exec (same as roc3.py does with gate1.py).
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

def load_roc3():
    src = open('experiments/encoding-induced-false-security-pufs-roc3.py').read()
    ns = {}
    exec(src.split("rows = []; t0 = time.time()")[0], ns)
    return ns
R = load_roc3()
roc3_internal, parity_feats, to_pm, labels = R['roc3_internal'], R['parity_feats'], R['to_pm'], R['labels']
t2_em = R['t2_em']

def gen_true(puf, N, rng):
    C = rng.integers(0, 2, size=(N, 64)).astype(np.uint8)
    Rs = labels(puf, C)
    Ci = roc3_internal(C, Rs)
    return C, labels(puf, Ci), Rs

def gen_randbit(puf, N, rng):
    C = rng.integers(0, 2, size=(N, 64)).astype(np.uint8)
    U = rng.integers(0, 2, size=N)
    Ci = roc3_internal(C, U)
    return C, labels(puf, Ci), U

def gate_acc(Ctr, Rstr, Cte, Rste, seed=0):
    g = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
    g.fit(parity_feats(Ctr), Rstr)
    return float(g.score(parity_feats(Cte), Rste))

def t0_raw(Ctr, ytr, Cte, yte, seed=0):
    c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
    c.fit(Ctr.astype(float), ytr)
    return float(c.score(Cte.astype(float), yte))

rows = []; t0 = time.time()
for inst in [0, 1]:
    puf = ArbiterPUF(n=64, seed=400+inst, noisiness=0.0)
    # E2-1: gate on true R*
    Ctr, ytr, Rstr = gen_true(puf, 20000, np.random.default_rng(2000+inst))
    Cte, yte, Rste = gen_true(puf, 10000, np.random.default_rng(3000+inst))
    ga = gate_acc(Ctr, Rstr, Cte, Rste, seed=inst)
    rows.append(('E2-1', 'gate-trueR*-20k', inst, round(ga, 4)))
    print('E2-1 inst', inst, 'gate acc', round(ga, 4), flush=True)
    # composite reference on same data (gated EM, identical to roc3.py)
    hist = t2_em(Ctr, ytr, Cte, yte)
    rows.append(('E2-ref', 'EM-20k-gated-trueR*', inst, hist[-1][1]))
    rows.append(('E2-ref', 'EM-20k-ungated-trueR*', inst, hist[-1][0]))
    print('E2-ref inst', inst, 'gated', hist[-1][1], 'ungated', hist[-1][0], flush=True)
    # E2-2: random-bit generation, identical attack pipeline
    Ctr2, ytr2, Utr = gen_randbit(puf, 20000, np.random.default_rng(2100+inst))
    Cte2, yte2, Ute = gen_randbit(puf, 10000, np.random.default_rng(3100+inst))
    hist2 = t2_em(Ctr2, ytr2, Cte2, yte2)
    rows.append(('E2-2', 'EM-20k-gated-randbit', inst, hist2[-1][1]))
    rows.append(('E2-2', 'EM-20k-ungated-randbit', inst, hist2[-1][0]))
    print('E2-2 inst', inst, 'gated', hist2[-1][1], 'ungated', hist2[-1][0], flush=True)
    fl = t0_raw(Ctr2, ytr2, Cte2, yte2, seed=inst)
    rows.append(('E2-2', 'T0-raw-floor-randbit', inst, round(fl, 4)))
    print('E2-2 inst', inst, 'T0 floor', round(fl, 4), flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-e2fals.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'instance', 'acc']); w.writerows(rows)
