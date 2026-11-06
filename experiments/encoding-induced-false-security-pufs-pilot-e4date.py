"""E4 scoped DATE-2018 pilot — vetted PROCEED (disclosure BLOCKED, see below).
Disclosure attempt (2026-10-18): full-text search of on-file papers/DATE2018-1034.pdf
for hidden/hidden-width/neuron/epoch/split/train-split -> only "multilayer"
(1 hit); visual inspection of all content pages (screenshots pp.2-6) -> §III.B
gives only "feed-forward network with multilayer perceptron ... resilient
back-propagation ... MATLAB built-in ... test data not part of training data".
Hidden width / epochs / train-test split are ABSENT from the paper.
STATED ASSUMPTION (not the paper's content): 1-hidden-layer MLP, H in {10,30}
(H=10 = MATLAB feedforwardnet/patternnet default width), trainrp-equivalent =
torch Rprop, 32-bit APUF, 30k train / 10k disjoint test, Rijndael S-box
substitution per byte (same table as gate1.py).
Targets: launchpad plain-32 >95% @30k (paper: >95%); sbox-32 ~66.4% (paper 66.4%).
TCO-PUF and permutation arms out of pilot scope (documented).
"""
import time, csv, numpy as np, torch
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression  # noqa (kept for parity-feat T0 floor)

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

def mlp_acc(Xtr, ytr, Xte, yte, H, seed, epochs=300):
    torch.manual_seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev).view(-1, 1)
    Xe = torch.tensor(Xte, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(32, H), torch.nn.Tanh(), torch.nn.Linear(H, 1)).to(dev)
    opt = torch.optim.Rprop(net.parameters(), lr=0.01)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(net(Xt), yt); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (net(Xe) > 0).float().cpu().numpy().ravel()
    return float((pred == yte).mean())

rows = []; t0 = time.time(); NTR, NTE = 30000, 10000
puf = ArbiterPUF(n=32, seed=0, noisiness=0.0)
rng = np.random.default_rng(11)
Ctr = rng.integers(0, 2, size=(NTR, 32)).astype(np.uint8)
Cte = np.random.default_rng(12).integers(0, 2, size=(NTE, 32)).astype(np.uint8)
ytr_plain = labels(puf, Ctr); yte_plain = labels(puf, Cte)
Ctr_s = sbox32(Ctr); Cte_s = sbox32(Cte)
ytr_s = labels(puf, Ctr_s); yte_s = labels(puf, Cte_s)
for H in [10, 30]:
    a = mlp_acc(Ctr.astype(float)*2-1, ytr_plain, Cte.astype(float)*2-1, yte_plain, H, 0)
    rows.append(('E4', f'plain-32-H={H}-30k', 0, round(a, 4)))
    print(f'E4 plain-32 H={H}', round(a, 4), flush=True)
    b = mlp_acc(Ctr.astype(float)*2-1, ytr_s, Cte.astype(float)*2-1, yte_s, H, 0)
    rows.append(('E4', f'sbox-32-H={H}-30k', 0, round(b, 4)))
    print(f'E4 sbox-32 H={H}', round(b, 4), flush=True)
# T0 raw-LR floors for reference
for name, Xtr, ytr_, Xte, yte_ in [('plain', Ctr, ytr_plain, Cte, yte_plain), ('sbox', Ctr, ytr_s, Cte, yte_s)]:
    c = LogisticRegression(max_iter=1000, solver='liblinear').fit(Xtr.astype(float), ytr_)
    fl = round(float(c.score(Xte.astype(float), yte_)), 4)
    rows.append(('E4', f'{name}-32-T0raw-30k', 0, fl))
    print(f'E4 {name}-32 T0raw', fl, flush=True)
print('elapsed', round(time.time()-t0, 1))
with open('experiments/encoding-induced-false-security-pufs-pilot-e4date.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'instance', 'acc']); w.writerows(rows)
