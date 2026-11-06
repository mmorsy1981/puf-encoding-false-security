"""XGB on PLAIN raw bits (review M4 control) — is XGB's 0.59 S-box lift
S-box-specific or general tree advantage on raw APUF bits?
Same protocol: N=50k/10k, 5 inst x 2 seeds, XGBClassifier(200, depth 6).
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from xgboost import XGBClassifier

def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)

rows = []; t0 = time.time()
for inst in [0, 1, 2, 3, 4]:
    for asd in [0, 1]:
        puf = ArbiterPUF(n=64, seed=inst, noisiness=0.0)
        rng = np.random.default_rng(10000*inst+asd)
        Ctr = rng.integers(0, 2, size=(50000, 64)).astype(np.uint8)
        Cte = np.random.default_rng(90000+inst).integers(0, 2, size=(10000, 64)).astype(np.uint8)
        ytr = labels(puf, Ctr); yte = labels(puf, Cte)
        c = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          subsample=0.8, random_state=asd, n_jobs=8)
        c.fit(Ctr.astype(float), ytr)
        a = round(float(c.score(Cte.astype(float), yte)), 4)
        rows.append(('XGBT0', 'plain-raw-50k', inst*10+asd, a))
        print('plain inst', inst, 'asd', asd, a, flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-xgbt0-plain.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'run', 'acc']); w.writerows(rows)
