"""ROC3 noisy spot: T0 (LR/SVM/MLP) + T2 gated EM at noisiness=0.1."""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

src = open('experiments/encoding-induced-false-security-pufs-roc3.py').read()
ns = {}
exec(src.split('rows = []')[0], ns)
rows = []; t0 = time.time()
for inst in [0, 1]:
    puf = ArbiterPUF(n=64, seed=400+inst, noisiness=0.1)
    Ctr, ytr, _ = ns['gen'](puf, 50000, np.random.default_rng(2000+inst))
    Cte, yte, _ = ns['gen'](puf, 10000, np.random.default_rng(3000+inst))
    # NOTE: gen() uses labels()->puf.eval with instance noisiness -> noisy labels
    r = {}
    r['LR-50k'] = LogisticRegression(max_iter=1000, solver='liblinear').fit(Ctr.astype(float), ytr).score(Cte.astype(float), yte)
    r['MLP-50k'] = MLPClassifier((100,), activation='logistic', max_iter=60, random_state=0).fit(Ctr.astype(float), ytr).score(Cte.astype(float), yte)
    print('inst', inst, {k: round(v, 4) for k, v in r.items()}, flush=True)
    for k, v in r.items(): rows.append(('ROC3-T0-nz01', k, inst, round(float(v), 4)))
    hist = ns['t2_em'](Ctr[:20000], ytr[:20000], Cte, yte)
    print('inst', inst, 'T2 noisy curve:', hist, flush=True)
    rows.append(('ROC3-T2-nz01', 'EM-20k-gated', inst, hist[-1][1]))
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-roc3-noisy.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['tier', 'config', 'instance', 'acc']); w.writerows(rows)
