"""ROC3 expansion to n=5 instances (review C2) — headline-break instance count.
Same protocol as roc3.py (T0 LR-50k/SVM-10k/MLP-50k + T2 EM-20k gated/ungated),
instances 2,3,4 (puf seed 400+inst, train rng 2000+inst, test rng 3000+inst).
Writes pilot-roc3-expand.csv (roc3.csv untouched).
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def load_roc3():
    src = open('experiments/encoding-induced-false-security-pufs-roc3.py').read()
    ns = {}
    exec(src.split("rows = []; t0 = time.time()")[0], ns)
    return ns
R = load_roc3()
roc3_internal, parity_feats, to_pm, labels, t2_em = R['roc3_internal'], R['parity_feats'], R['to_pm'], R['labels'], R['t2_em']
gen = R['gen']

rows = []; t0 = time.time()
for inst in [2, 3, 4]:
    puf = ArbiterPUF(n=64, seed=400+inst, noisiness=0.0)
    Ctr, ytr, Rs = gen(puf, 50000, np.random.default_rng(2000+inst))
    Cte, yte, _ = gen(puf, 10000, np.random.default_rng(3000+inst))
    r = {}
    r['LR-50k'] = LogisticRegression(max_iter=1000, solver='liblinear').fit(Ctr.astype(float), ytr).score(Cte.astype(float), yte)
    sub, sut = Ctr[:10000], ytr[:10000]
    r['SVM-10k'] = SVC(kernel='rbf').fit(sub.astype(float), sut).score(Cte.astype(float), yte)
    r['MLP-50k'] = MLPClassifier((100,), activation='logistic', max_iter=60, random_state=0).fit(Ctr.astype(float), ytr).score(Cte.astype(float), yte)
    print('inst', inst, {k: round(v, 4) for k, v in r.items()}, flush=True)
    for k, v in r.items(): rows.append(('ROC3x', 'T0-'+k, inst, round(float(v), 4)))
    hist = t2_em(Ctr[:20000], ytr[:20000], Cte, yte)
    rows.append(('ROC3x', 'EM-20k-gated', inst, hist[-1][1]))
    rows.append(('ROC3x', 'EM-20k-ungated', inst, hist[-1][0]))
    print('inst', inst, 'T2-EM gated/ungated:', hist[-1], flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-roc3x.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'instance', 'acc']); w.writerows(rows)
