"""E4 parity-input ANN (V4-fix slice 3).
Adapts pilot-e4date.py (torch Rprop, 300 full-batch epochs, BCE, Tanh hidden)
to PARITY-FEATURE inputs (33 dims = 32 parity + bias) of the external 32-bit
challenge. 32-bit APUF, 30k train / 10k disjoint test, H in {10,30},
5 instances (puf seeds 0-4) x 3 seeds (torch seeds 0,1,2; data rngs
train 11+100*inst+1000*s, test 12+100*inst+1000*s), plain-32 AND sbox-32
(sbox32 bytewise AES map from e4date.py). Plus per-(inst,seed) T1 analog:
parity-LR on sbox-mapped challenges (informed, expect ~1.0) and plain parity-LR
baseline.
Writes experiments/encoding-induced-false-security-pufs-pilot-e4parity.csv
cols: test,config,instance,seed,acc
"""
import time, csv, numpy as np, torch
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

CSV = 'experiments/encoding-induced-false-security-pufs-pilot-e4parity.csv'

def load_gate1():
    src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
    ns = {}
    exec(src.split(chr(10) + "if __name__ == '__main__':")[0], ns)
    return ns
G = load_gate1()
SBOX_ARR = G['SBOX_ARR']
to_pm, labels = G['to_pm'], G['labels']

def bits_to_bytes32(C):
    N = C.shape[0]; B = np.zeros((N, 4), dtype=np.uint8)
    for i in range(4):
        B[:, i] = np.packbits(C[:, 8*i:8*i+8], axis=1).ravel()
    return B

def bytes_to_bits32(B):
    N = B.shape[0]; C = np.zeros((N, 32), dtype=np.uint8)
    for i in range(4):
        C[:, 8*i:8*i+8] = np.unpackbits(B[:, i:i+1], axis=1)
    return C

def sbox32(C): return bytes_to_bits32(SBOX_ARR[bits_to_bytes32(C)])

def pfeat32(C):
    S = 1 - 2*C.astype(float); X = np.ones((len(C), 33))
    for i in range(32): X[:, i] = np.prod(S[:, i:], axis=1)
    return X

def mlp_acc(Xtr, ytr, Xte, yte, H, seed, epochs=300):
    torch.manual_seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev).view(-1, 1)
    Xe = torch.tensor(Xte, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], H), torch.nn.Tanh(), torch.nn.Linear(H, 1)).to(dev)
    opt = torch.optim.Rprop(net.parameters(), lr=0.01)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(net(Xt), yt); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (net(Xe) > 0).float().cpu().numpy().ravel()
    return float((pred == yte).mean())

rows = []; t0 = time.time(); NTR, NTE = 30000, 10000
for inst in [0, 1, 2, 3, 4]:
    puf = ArbiterPUF(n=32, seed=inst, noisiness=0.0)
    for s in [0, 1, 2]:
        rng_tr = np.random.default_rng(11 + 100*inst + 1000*s)
        rng_te = np.random.default_rng(12 + 100*inst + 1000*s)
        Ctr = rng_tr.integers(0, 2, size=(NTR, 32)).astype(np.uint8)
        Cte = rng_te.integers(0, 2, size=(NTE, 32)).astype(np.uint8)
        Ctr_s = sbox32(Ctr); Cte_s = sbox32(Cte)
        ytr_plain = labels(puf, Ctr); yte_plain = labels(puf, Cte)
        ytr_s = labels(puf, Ctr_s); yte_s = labels(puf, Cte_s)
        Xtr_par = pfeat32(Ctr); Xte_par = pfeat32(Cte)
        # T1 analog + plain baseline (parity LR)
        for name, Xtr_, ytr_, Xte_, yte_ in [
            ('plain-T1par', Xtr_par, ytr_plain, Xte_par, yte_plain),
            ('sbox-T1par-mapped', pfeat32(Ctr_s), ytr_s, pfeat32(Cte_s), yte_s),
        ]:
            c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=s)
            c.fit(Xtr_, ytr_)
            a = round(float(c.score(Xte_, yte_)), 4)
            rows.append(('E4par', f'{name}-30k', inst, s, a))
            print(f'E4par inst{inst} seed{s} {name} {a}', flush=True)
        # parity-input ANN, plain and sbox (external parity feats, uninformed for sbox)
        for name, (ytrv, ytev) in [('plain', (ytr_plain, yte_plain)), ('sbox', (ytr_s, yte_s))]:
            for H in [10, 30]:
                a = mlp_acc(Xtr_par.astype(np.float32), ytrv, Xte_par.astype(np.float32), ytev, H, s)
                a = round(a, 4)
                rows.append(('E4par', f'{name}-32-H={H}-30k-parIn', inst, s, a))
                print(f'E4par inst{inst} seed{s} {name} H={H} {a}', flush=True)
print('elapsed', round(time.time()-t0, 1))
with open(CSV, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'instance', 'seed', 'acc']); w.writerows(rows)
