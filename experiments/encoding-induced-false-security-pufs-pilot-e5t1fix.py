"""E5 T1-control fix: T1 MUST apply the known S-box map before parity feats.
Reruns only the T1 parity-LR control at each N on byte-identical regenerated data
(same seeds as pilot-e5scale.py: puf seed 0, train rng 11, test rng 12).
Expected: ~0.97+ at all N (gate1: 0.966 @1k), vs bigMLP 0.51->0.60.
"""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

def load_gate1():
    src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
    ns = {}
    exec(src.split(chr(10) + "if __name__ == '__main__':")[0], ns)
    return ns
G = load_gate1()
SBOX_ARR, parity_feats, to_pm, labels = G['SBOX_ARR'], G['parity_feats'], G['to_pm'], G['labels']
bits_to_bytes, bytes_to_bits = G['bits_to_bytes'], G['bytes_to_bits']
def sbox_map(C): return bytes_to_bits(SBOX_ARR[bits_to_bytes(C)])

t0 = time.time()
puf = ArbiterPUF(n=64, seed=0, noisiness=0.0)
Ctr_full = np.random.default_rng(11).integers(0, 2, size=(1000000, 64)).astype(np.uint8)
Cte = np.random.default_rng(12).integers(0, 2, size=(10000, 64)).astype(np.uint8)
ytr_full = np.zeros(1000000, dtype=int)
for s in range(0, 1000000, 100000):
    ytr_full[s:s+100000] = labels(puf, sbox_map(Ctr_full[s:s+100000]))
yte = labels(puf, sbox_map(Cte))
rows = []
for N in [20000, 50000, 100000, 200000, 500000, 1000000]:
    c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=0)
    c.fit(parity_feats(sbox_map(Ctr_full[:N])), ytr_full[:N])
    a = round(float(c.score(parity_feats(sbox_map(Cte)), yte)), 4)
    rows.append(('E5fix', f'T1par-mapped-N={N}', 0, a))
    print(f'T1 mapped N={N}', a, flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-e5t1fix.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'instance', 'acc']); w.writerows(rows)
