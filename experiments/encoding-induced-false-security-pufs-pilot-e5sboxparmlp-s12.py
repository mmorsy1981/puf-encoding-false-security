"""E5 sbox parity-input big-MLP, paired seeds 1-2 (V6 item 5).
Same as pilot-e5parmlp.py (seed 0) but puf/data/torch seeds paired to
pilot-e5plainparmlp.py seeds 1,2: (puf 1, train rng 111, test rng 112),
(puf 2, train rng 211, test rng 212). N grid 20k/200k/1M (plain grid).
Writes experiments/encoding-induced-false-security-pufs-pilot-e5sboxparmlp-s12.csv
cols: test,config,instance,acc,train_acc.
"""
import time, csv, os, numpy as np, torch
from pypuf.simulation import ArbiterPUF

CSV = 'experiments/encoding-induced-false-security-pufs-pilot-e5sboxparmlp-s12.csv'
NGRID = [20000, 200000, 1000000]
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

def bigmlp_par(Xtr, ytr, Xte, yte, seed):
    torch.manual_seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = torch.nn.Sequential(
        torch.nn.Linear(65, 512), torch.nn.ReLU(),
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

if __name__ == '__main__':
    t0 = time.time()
    rows = []
    for s, (pseed, rtr, rte) in {1: (1, 111, 112), 2: (2, 211, 212)}.items():
        puf = ArbiterPUF(n=64, seed=pseed, noisiness=0.0)
        Ctr_full = np.random.default_rng(rtr).integers(0, 2, size=(1000000, 64)).astype(np.uint8)
        Cte = np.random.default_rng(rte).integers(0, 2, size=(10000, 64)).astype(np.uint8)
        ytr_full = gen_labels(puf, Ctr_full); yte = gen_labels(puf, Cte)
        Xte_par = parity_feats(Cte).astype(np.float32)
        for N in NGRID:
            tr, te = bigmlp_par(parity_feats(Ctr_full[:N]).astype(np.float32), ytr_full[:N], Xte_par, yte, s)
            rows.append(('E5sboxpar', 'parMLP-sbox-N=%d' % N, s, round(te, 4), round(tr, 4)))
            print('E5sboxpar seed=%d N=%d test=%.4f train=%.4f' % (s, N, te, tr), flush=True)
    print('elapsed', round(time.time() - t0, 1))
    with open(CSV, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['test', 'config', 'instance', 'acc', 'train_acc']); w.writerows(rows)
