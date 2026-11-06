"""Full repeats 10x3 + XGB + noisy spots. Incremental CSV writes per section.
S1 Gate1: inst 0-9 x seeds 0-2, N=[500,1000,5000], tiers plain/T0/T1.
S2 ROC3: T0 (LR/SVM-10k/MLP-50k/XGB-50k) + T2-EM gated (3 EM seeds) at 10 inst x 1 att seed... (time-box: 5 inst x 2 seeds).
S3 RSO m=32/1M 5x2; OB-exact 10x3 N=20k; LFSR-exact 10x3 N=10k.
S4 Noisy nz=0.1 spots: RSO/OB/LFSR (3x1) + ROC3 T2 (3x1).
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

t0 = time.time()
def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def pfeat(C):
    S = 1 - 2*C.astype(float); X = np.ones((len(C), 65))
    for i in range(64): X[:, i] = np.prod(S[:, i:], axis=1)
    return X
def lr_acc(Xtr, ytr, Xte, yte, seed=0):
    if len(np.unique(ytr)) < 2:
        return float((yte == np.bincount(yte).argmax()).mean())
    c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
    c.fit(Xtr.astype(float), ytr); return float(c.score(Xte.astype(float), yte))
def emit(path, row):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(row)

G1 = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
NS = {}
exec(G1.split('INSTANCES =')[0], NS)
for p in ['experiments/encoding-induced-false-security-pufs-fullA.csv',
          'experiments/encoding-induced-false-security-pufs-fullB.csv']:
    with open(p, 'w', newline='') as f:
        csv.writer(f).writerow(['section', 'design', 'config', 'instance', 'seed', 'acc'])

# ---- S1 Gate1 10x3 ----
for inst in range(10):
    puf = ArbiterPUF(n=64, seed=inst, noisiness=0.0)
    rt = np.random.default_rng(91000+inst)
    Cte = rt.integers(0, 2, size=(5000, 64)).astype(np.uint8)
    yp = labels(puf, Cte); ys = labels(puf, NS['pi_sbox'](Cte))
    for asd in [0, 1, 2]:
        r = np.random.default_rng(10000*inst+asd)
        Ce = r.integers(0, 2, size=(5000, 64)).astype(np.uint8)
        ypp = labels(puf, Ce); yss = labels(puf, NS['pi_sbox'](Ce))
        for N in [500, 1000, 5000]:
            a = lr_acc(pfeat(Ce[:N]), ypp[:N], pfeat(Cte), yp, asd)
            b = lr_acc(Ce[:N].astype(float), yss[:N], Cte.astype(float), ys, asd)
            c = lr_acc(pfeat(NS['pi_sbox'](Ce[:N])), yss[:N], pfeat(NS['pi_sbox'](Cte)), ys, asd)
            for tier, v in (('plain', a), ('T0', b), ('T1', c)):
                emit('experiments/encoding-induced-false-security-pufs-fullA.csv', ['S1', 'gate1', f'N={N}', inst, asd, round(v, 4)])
    print('S1 inst', inst, round(time.time()-t0, 1), flush=True)

# ---- S2 ROC3 5x2 + XGB + EM x3 seeds ----
src = open('experiments/encoding-induced-false-security-pufs-roc3.py').read()
R3 = {}
exec(src.split('rows = []')[0], R3)
for inst in [0, 1, 2, 3, 4]:
    puf = ArbiterPUF(n=64, seed=400+inst, noisiness=0.0)
    Ctr, ytr, _ = R3['gen'](puf, 50000, np.random.default_rng(2000+inst))
    Cte, yte, _ = R3['gen'](puf, 10000, np.random.default_rng(3000+inst))
    for asd in [0, 1]:
        a = lr_acc(Ctr, ytr, Cte, yte, asd)
        x = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=asd, n_jobs=4, eval_metric='logloss').fit(Ctr.astype(np.float32), ytr).score(Cte.astype(np.float32), yte)
        emit('experiments/encoding-induced-false-security-pufs-fullA.csv', ['S2', 'ROC3-T0-LR', 'N=50k', inst, asd, round(float(a), 4)])
        emit('experiments/encoding-induced-false-security-pufs-fullA.csv', ['S2', 'ROC3-T0-XGB', 'N=50k', inst, asd, round(float(x), 4)])
        print('S2 T0 inst', inst, 'seed', asd, round(float(a), 4), round(float(x), 4), flush=True)
    for esd in [0, 1, 2]:
        h = R3['t2_em'](Ctr[:20000], ytr[:20000], Cte, yte, seed=esd)
        emit('experiments/encoding-induced-false-security-pufs-fullA.csv', ['S2', 'ROC3-T2-gated', 'N=20k', inst, esd, h[-1][1]])
        print('S2 T2 inst', inst, 'eseed', esd, h[-1], flush=True)
print('S2 done', round(time.time()-t0, 1), flush=True)
