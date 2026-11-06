"""ROC3-SPUF exact-spec (Ma et al. 2023, Sec 3.1 scheme 3-3) T0 + T2.
Mechanism (64-bit): R*=APUF(C); odd/even bit-flip by R* -> C1;
AES S-box per byte -> C2; cyclic shift L/R by 1 by R* -> C*; R=APUF(C*).
Bit convention: 1-indexed odd bits = 0-indexed positions 0,2,4... (documented choice).
T0: LR/SVM-RBF/MLP on external C (paper: ~53% at 500k train).
T2: hard-EM mixture of 2 parity-LR experts (mapping C->C*|r is public given r).
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def load_lib():
    src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
    ns = {}
    exec(src.split(chr(10) + "if __name__ == '__main__':")[0], ns)
    return ns
L = load_lib()
SBOX_ARR = L['SBOX_ARR']

def flip_oe(C, r):
    C = C.copy()
    idx = np.arange(0, 64, 2) if r == 0 else np.arange(1, 64, 2)
    C[:, idx] ^= 1
    return C

def shift_lr(C, r):
    return np.roll(C, -1 if r == 0 else 1, axis=1)

def roc3_internal(C, Rstar):
    N = len(C); out = np.zeros_like(C)
    for r in (0, 1):
        m = (Rstar == r)
        if m.sum() == 0: continue
        C1 = flip_oe(C[m], r)
        B = L['bits_to_bytes'](C1)
        C2 = L['bytes_to_bits'](SBOX_ARR[B])
        out[m] = shift_lr(C2, r)
    return out

def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def parity_feats(C):
    S = 1 - 2*C.astype(float); X = np.ones((len(C), 65))
    for i in range(64): X[:, i] = np.prod(S[:, i:], axis=1)
    return X

def gen(puf, N, rng):
    C = rng.integers(0, 2, size=(N, 64)).astype(np.uint8)
    Rs = labels(puf, C)
    Ci = roc3_internal(C, Rs)
    return C, labels(puf, Ci), Rs

def t2_em(Ctr, ytr, Cte, yte, iters=8, seed=0):
    rng = np.random.default_rng(seed)
    asg = rng.integers(0, 2, size=len(Ctr))
    acc_hist = []
    for it in range(iters):
        models = []
        for r in (0, 1):
            m = (asg == r)
            if m.sum() < 100:
                models.append(None); continue
            clf = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
            clf.fit(parity_feats(roc3_internal(Ctr[m], np.full(m.sum(), r))), ytr[m])
            models.append(clf)
        # E-step: assign by agreement (margin tie-break)
        margins = np.zeros((len(Ctr), 2))
        for r in (0, 1):
            if models[r] is None: margins[:, r] = -np.inf; continue
            margins[:, r] = models[r].decision_function(parity_feats(roc3_internal(Ctr, np.full(len(Ctr), r))))
        agree = np.sign(margins) == (2*ytr - 1)[:, None]
        better = np.argmax(np.where(agree, np.abs(margins), -1), axis=1)
        # keep previous where neither agrees
        neither = ~agree.any(axis=1)
        better[neither] = asg[neither]
        asg = better
        # test accuracy with oracle-free decode: pick expert by |margin|
        tm = np.zeros((len(Cte), 2))
        for r in (0, 1):
            if models[r] is None: tm[:, r] = -np.inf; continue
            tm[:, r] = models[r].decision_function(parity_feats(roc3_internal(Cte, np.full(len(Cte), r))))
        pred = (tm[np.arange(len(Cte)), np.argmax(np.abs(tm), axis=1)] > 0).astype(int)
        acc = float((pred == yte).mean())
        acc_hist.append(round(acc, 4))
        # gated decode: train gate C -> assignment, then route per sample
        gate = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
        try:
            gate.fit(parity_feats(Ctr), asg)
            gr = gate.predict(parity_feats(Cte))
            gp = np.zeros(len(Cte), dtype=int)
            for r in (0, 1):
                m = (gr == r)
                if m.sum() and models[r] is not None:
                    gp[m] = (models[r].decision_function(parity_feats(roc3_internal(Cte[m], np.full(m.sum(), r)))) > 0).astype(int)
            gacc = float((gp == yte).mean())
        except Exception:
            gacc = acc
        acc_hist[-1] = (acc_hist[-1], round(gacc, 4))
    return acc_hist

rows = []; t0 = time.time()
for inst in [0, 1]:
    puf = ArbiterPUF(n=64, seed=400+inst, noisiness=0.0)
    Ctr, ytr, Rs = gen(puf, 50000, np.random.default_rng(2000+inst))
    Cte, yte, _ = gen(puf, 10000, np.random.default_rng(3000+inst))
    r = {}
    r['LR-50k'] = LogisticRegression(max_iter=1000, solver='liblinear').fit(Ctr.astype(float), ytr).score(Cte.astype(float), yte)
    sub, sut = Ctr[:10000], ytr[:10000]
    r['SVM-10k'] = SVC(kernel='rbf').fit(sub.astype(float), sut).score(Cte.astype(float), yte)
    r['MLP-50k'] = MLPClassifier((100,), activation='logistic', max_iter=60, random_state=0).fit(Ctr.astype(float), ytr).score(Cte.astype(float), yte)
    print('inst', inst, {k: round(v, 4) for k, v in r.items()}, flush=True)
    for k, v in r.items(): rows.append(('ROC3-T0', k, inst, round(float(v), 4)))
    hist = t2_em(Ctr[:20000], ytr[:20000], Cte, yte)
    print('inst', inst, 'T2-EM curve:', hist, flush=True)
    rows.append(('ROC3-T2', 'EM-20k-gated', inst, hist[-1][1]))
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-roc3.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['tier', 'config', 'instance', 'acc']); w.writerows(rows)
