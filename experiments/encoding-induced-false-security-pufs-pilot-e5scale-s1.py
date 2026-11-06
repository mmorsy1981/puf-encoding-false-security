"""E5 second seed (review C2) — independent instance/data/init.
Same grid/arch as pilot-e5scale.py (64->512->256->128->1, ReLU, Adam 1e-3,
batch 4096, 40 epochs): puf seed 1, train rng 111, test rng 112, torch manual
seed 1. Writes pilot-e5scale-s1.csv (original untouched).
"""
import time, csv, numpy as np, torch
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

CSV = 'experiments/encoding-induced-false-security-pufs-pilot-e5scale-s1.csv'
NGRID = [20000, 50000, 100000, 200000, 500000, 1000000]
EPOCHS, BS, LR = 40, 4096, 1e-3

def load_gate1():
    src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
    ns = {}
    exec(src.split(chr(10) + "if __name__ == '__main__':")[0], ns)
    return ns
G = load_gate1()
SBOX_ARR, parity_feats, to_pm, labels = G['SBOX_ARR'], G['parity_feats'], G['to_pm'], G['labels']
bits_to_bytes, bytes_to_bits = G['bits_to_bytes'], G['bytes_to_bits']
def sbox_map(C): return bytes_to_bits(SBOX_ARR[bits_to_bytes(C)])

def gen_labels(puf, C):
    Ci = sbox_map(C); n = len(C); y = np.zeros(n, dtype=int)
    for s in range(0, n, 100000):
        y[s:s+100000] = labels(puf, Ci[s:s+100000])
    return y

def bigmlp(Xtr, ytr, Xte, yte, seed):
    torch.manual_seed(seed)
    dev = 'cuda'
    net = torch.nn.Sequential(
        torch.nn.Linear(64, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt),
        batch_size=BS, shuffle=True, generator=torch.Generator().manual_seed(seed))
    for _ in range(EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
    with torch.no_grad():
        tr = (net(torch.tensor(Xtr, dtype=torch.float32).to(dev)) > 0).float().cpu().numpy().ravel()
        pr = (net(torch.tensor(Xte, dtype=torch.float32).to(dev)) > 0).float().cpu().numpy().ravel()
    return float((tr == ytr).mean()), float((pr == yte).mean())

t0 = time.time()
puf = ArbiterPUF(n=64, seed=1, noisiness=0.0)
Ctr_full = np.random.default_rng(111).integers(0, 2, size=(1000000, 64)).astype(np.uint8)
Cte = np.random.default_rng(112).integers(0, 2, size=(10000, 64)).astype(np.uint8)
ytr_full = gen_labels(puf, Ctr_full)
yte = gen_labels(puf, Cte)
with open(CSV, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'instance', 'acc', 'train_acc'])
    for N in NGRID:
        Xtr = (Ctr_full[:N].astype(np.float32)*2-1); Xte = (Cte.astype(np.float32)*2-1)
        c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=1)
        c.fit(parity_feats(sbox_map(Ctr_full[:N])), ytr_full[:N])
        t1 = round(float(c.score(parity_feats(sbox_map(Cte)), yte)), 4)
        tr, te = bigmlp(Xtr, ytr_full[:N], Xte, yte, 1)
        te, tr = round(te, 4), round(tr, 4)
        w.writerow(['E5s1', f'T1par-mapped-N={N}', 1, t1, '']); f.flush()
        w.writerow(['E5s1', f'bigMLP-N={N}', 1, te, tr]); f.flush()
        print(f'E5s1 N={N} T1={t1} bigMLP test={te} train={tr}', flush=True)
print('elapsed', round(time.time()-t0, 1))
