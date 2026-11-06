"""XGB generic-learner T0 checks for S-box + LFSR bands (review M4).
Mirrors the ROC3 T0-XGB protocol (N=50k, 5 inst x 2 model seeds): a competent
non-LR generic learner must also fail on the uninformed T0 for the band claim
to be learner-independent.
(a) S-box raw bits -> sbox labels, 64-bit, N=50k/10k, 5 inst x 2 seeds.
(b) LFSR Eq.7 raw C1 bits -> labels, Base=10, N=10k/5k, 3 inst x 2 seeds.
XGBClassifier(n_estimators=200, max_depth=6, lr=0.1, subsample=0.8).
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from xgboost import XGBClassifier

def load_gate1():
    src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
    ns = {}
    exec(src.split(chr(10) + "if __name__ == '__main__':")[0], ns)
    return ns
G = load_gate1()
SBOX_ARR, to_pm, labels = G['SBOX_ARR'], G['to_pm'], G['labels']
bits_to_bytes, bytes_to_bits = G['bits_to_bytes'], G['bytes_to_bits']
def sbox_map(C): return bytes_to_bits(SBOX_ARR[bits_to_bytes(C)])

def load_eq7():
    src = open('experiments/encoding-induced-false-security-pufs-lfsr-eq7-parity.py').read()
    ns = {}
    exec(src.split("rows = []; t0 = time.time()")[0], ns)
    return ns
Q = load_eq7()

def xgb_acc(Xtr, ytr, Xte, yte, seed):
    if len(np.unique(ytr)) < 2:
        return float((yte == np.bincount(yte).argmax()).mean())
    c = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                      subsample=0.8, random_state=seed, n_jobs=8)
    c.fit(Xtr.astype(float), ytr)
    return float(c.score(Xte.astype(float), yte))

rows = []; t0 = time.time()
for inst in [0, 1, 2, 3, 4]:
    for asd in [0, 1]:
        puf = ArbiterPUF(n=64, seed=inst, noisiness=0.0)
        rng = np.random.default_rng(10000*inst+asd)
        Ctr = rng.integers(0, 2, size=(50000, 64)).astype(np.uint8)
        Cte = np.random.default_rng(90000+inst).integers(0, 2, size=(10000, 64)).astype(np.uint8)
        ytr = labels(puf, sbox_map(Ctr)); yte = labels(puf, sbox_map(Cte))
        a = xgb_acc(Ctr, ytr, Cte, yte, asd)
        rows.append(('XGBT0', 'sbox-raw-50k', inst*10+asd, round(a, 4)))
        print('sbox inst', inst, 'asd', asd, round(a, 4), flush=True)
for inst in [0, 1, 2]:
    for asd in [0, 1]:
        puf = ArbiterPUF(n=64, seed=200+inst, noisiness=0.0)
        rng = np.random.default_rng(6000+inst+1000*asd)
        Ctr = rng.integers(0, 2, size=(10000, 68)).astype(np.uint8)
        Cte = rng.integers(0, 2, size=(5000, 68)).astype(np.uint8)
        ytr = labels(puf, Q['lfsr_int'](Ctr, 10)); yte = labels(puf, Q['lfsr_int'](Cte, 10))
        a = xgb_acc(Ctr[:, :64], ytr, Cte[:, :64], yte, asd)
        rows.append(('XGBT0', 'lfsr-eq7-raw-10k', inst*10+asd, round(a, 4)))
        print('lfsr inst', inst, 'asd', asd, round(a, 4), flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-xgbt0.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'run', 'acc']); w.writerows(rows)
