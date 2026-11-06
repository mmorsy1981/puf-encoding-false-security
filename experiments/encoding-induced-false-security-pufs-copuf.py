"""CO-PUF custom sim (Li et al. 2024 Sec 3), config (n=2,l=8,d=4) -> 64-bit challenges.
Per DS: 4 challenge bits select parallel/crossed delays on 4 upper + 4 lower MUXes.
Selection O=[max(U0,D0),min(U1,D1),min(U2,D2),max(U3,D3)]; fixed permutation T routes
O to next stage inputs (upper<-O[T0],O[T1]; lower<-O[T2],O[T3]); documented simplification
of the paper's cross-module wiring. Final Psi=[z0-z3, z1-z2]; R=XOR(sgn>0).
Comparator: plain 64-bit APUF in the same harness/learners.
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def to_pm(C01): return (2*C01.astype(np.int8) - 1)

class COPUF:
    def __init__(self, n_mod=2, l_ds=8, seed=0, sigma=1.0):
        rng = np.random.default_rng(seed)
        self.n, self.l = n_mod, l_ds
        self.D = rng.normal(0, sigma, size=(n_mod, l_ds, 2, 4, 2))
        self.T = np.array([2, 0, 3, 1])
    def eval(self, C):
        N = len(C); R = np.zeros(N, dtype=int)
        for t in range(N):
            su = np.zeros(4); sl = np.zeros(4); p = 0
            for m in range(self.n):
                for j in range(self.l):
                    b = C[t, p:p+4]; p += 4
                    U = su + self.D[m, j, 0, np.arange(4), b]
                    Dl = sl + self.D[m, j, 1, np.arange(4), b]
                    o = np.array([max(U[0], Dl[0]), min(U[1], Dl[1]),
                                  min(U[2], Dl[2]), max(U[3], Dl[3])])
                    o = o - o.mean()  # translation-invariant: no effect on responses (common offsets cancel in psi differences); kept for numerical stability
                    su = o[self.T]; sl = o[::-1]  # symmetric fan-out: reversed lower copy. Paper achieves balance via manual symmetric layout/routing; this is its model analogue (disclosed). 'base' (sl=o) gave degenerate bias 0.05-0.41; 'swap' centers much better.
            psi = np.array([o[0] - o[3], o[1] - o[2]])
            R[t] = int((psi > 0).sum() % 2)
        return R

def labels_puf(puf, C01):
    return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)

def trial(make_puf, Ntr, Nte, seed, mlp=False):
    rng = np.random.default_rng(seed)
    Ctr = rng.integers(0, 2, size=(Ntr, 64)).astype(np.uint8)
    Cte = rng.integers(0, 2, size=(Nte, 64)).astype(np.uint8)
    puf = make_puf()
    ytr = puf(Ctr) if callable(getattr(puf, '__call__', None)) else None
    return Ctr, Cte

t0 = time.time(); rows = []
for inst in [0, 1]:
    co = COPUF(seed=100+inst); ap = ArbiterPUF(n=64, seed=200+inst, noisiness=0.0)
    rng = np.random.default_rng(3000+inst)
    Ctr = rng.integers(0, 2, size=(20000, 64)).astype(np.uint8)
    Cte = rng.integers(0, 2, size=(5000, 64)).astype(np.uint8)
    yco_tr, yco_te = co.eval(Ctr), co.eval(Cte)
    yap_tr, yap_te = labels_puf(ap, Ctr), labels_puf(ap, Cte)
    r = {}
    r['CO-LR'] = LogisticRegression(max_iter=1000, solver='liblinear').fit(Ctr.astype(float), yco_tr).score(Cte.astype(float), yco_te)
    r['APUF-rawLR'] = LogisticRegression(max_iter=1000, solver='liblinear').fit(Ctr.astype(float), yap_tr).score(Cte.astype(float), yap_te)
    Xtr = np.ones((len(Ctr), 65)); S = 1-2*Ctr.astype(float)
    for ii in range(64): Xtr[:, ii] = np.prod(S[:, ii:], axis=1)
    Xte = np.ones((len(Cte), 65)); St = 1-2*Cte.astype(float)
    for ii in range(64): Xte[:, ii] = np.prod(St[:, ii:], axis=1)
    r['APUF-phiLR'] = LogisticRegression(max_iter=1000, solver='liblinear').fit(Xtr, yap_tr).score(Xte, yap_te)
    r['CO-SVM10k'] = SVC(kernel='rbf').fit(Ctr[:10000].astype(float), yco_tr[:10000]).score(Cte.astype(float), yco_te)
    r['CO-MLP'] = MLPClassifier((100,), activation='logistic', max_iter=60, random_state=0).fit(Ctr.astype(float), yco_tr).score(Cte.astype(float), yco_te)
    print('inst', inst, {k: round(float(v), 4) for k, v in r.items()}, 'CO bal', round(float(yco_tr.mean()), 3), flush=True)
    for k, v in r.items(): rows.append(('CO-PUF-custom', k, inst, round(float(v), 4)))
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-copuf.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['design', 'config', 'instance', 'acc']); w.writerows(rows)
