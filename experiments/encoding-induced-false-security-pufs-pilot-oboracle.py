"""OB oracle filing (V4-fix slice 5).
Adapts outputs/encoding-false-security-pufs-review-probes/probe3.py R4
(lines ~98-112) into an experiments artifact. Oracle knows the 4 APUFs and
observes the 4-bit response R: per challenge, enumerates the 4 candidate
responses (one per pattern) and picks uniformly among patterns consistent
with observed R. Reports per-instance oracle pattern-ID rate
E[1/k if true pattern consistent] and unique-pattern fraction P(k==1),
N=5000, seeds 7100+inst, puf seeds 300+inst*10+j, noiseless.
Writes experiments/encoding-induced-false-security-pufs-pilot-oboracle.csv
cols: instance,metric,value
"""
import csv, numpy as np
from pypuf.simulation import ArbiterPUF

def load_gate1():
    src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
    ns = {}
    exec(src.split(chr(10) + "if __name__ == '__main__':")[0].split('INSTANCES =')[0], ns)
    return ns
G = load_gate1()
labels = G['labels']

OB_POS = [[0,1,2],[0,1,63],[0,62,63],[61,62,63]]
OB_VAL = np.array([[0,1,0],[1,0,1],[0,1,0],[1,0,1]], dtype=np.uint8)
OB_SR = np.array([[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,1,0]], dtype=np.uint8)

def ob_full(Cob, i):
    C = np.zeros((len(Cob), 64), dtype=np.uint8); pos = sorted(OB_POS[i])
    C[:, [d for d in range(64) if d not in pos]] = Cob
    for p_, v in zip(pos, OB_VAL[i]): C[:, p_] = v
    return C

rows = []
for inst in (0, 1, 2):
    pufs = [ArbiterPUF(n=64, seed=300+inst*10+j, noisiness=0.0) for j in range(4)]
    r = np.random.default_rng(7100+inst)
    Cob = r.integers(0, 2, size=(5000, 61)).astype(np.uint8); ii = r.integers(0, 4, size=5000)
    cand = np.stack([np.column_stack([labels(p, ob_full(Cob, i)) for p in pufs]) ^ OB_SR[i] for i in range(4)])
    Y = cand[ii, np.arange(5000)]
    match = (cand == Y[None]).all(-1)
    k = match.sum(0)
    p_id = float(np.mean(match[ii, np.arange(5000)] / k))
    uniq = float(np.mean(k == 1))
    rows.append((inst, 'oracle-pattern-id-rate', round(p_id, 4)))
    rows.append((inst, 'unique-pattern-fraction', round(uniq, 4)))
    print(f'OB inst{inst}: oracle pattern-ID rate={p_id:.4f} unique={uniq:.3f}', flush=True)
with open('experiments/encoding-induced-false-security-pufs-pilot-oboracle.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['instance', 'metric', 'value']); w.writerows(rows)
