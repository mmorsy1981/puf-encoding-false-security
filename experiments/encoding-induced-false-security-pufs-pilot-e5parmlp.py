"""E5 parity-input big-MLP (V4-fix slice 1).
Same big-MLP as pilot-e5scale.py (512->256->128->1, ReLU, Adam 1e-3, batch 4096,
40 epochs, BCEWithLogits) but input = 65-dim parity feats of the EXTERNAL 64-bit
challenge (uninformed: no S-box map applied to inputs), labels = sbox-APUF
f(pi(C_ext)) with gate1 AES S-box pi. First Linear adjusted 64->65.
Data byte-identical to pilot-e5scale.py seed-0: puf seed 0, train rng 11 (1M),
test rng 12 (10k disjoint), nested subsets, torch seed 0. N grid
20k/50k/100k/200k/500k/1M. Per N also files the informed T1 parity-LR control
(parity feats of pi(C), expect ~0.99+) for contrast.
Writes experiments/encoding-induced-false-security-pufs-pilot-e5parmlp.csv
cols: test,config,instance,acc,train_acc. Incremental append+flush per N.
"""
import time, csv, os, numpy as np, torch
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

CSV = 'experiments/encoding-induced-false-security-pufs-pilot-e5parmlp.csv'
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

def done_N():
    if not os.path.exists(CSV): return set()
    out = set()
    for r in csv.DictReader(open(CSV)):
        if r['test'] == 'E5par' and 'parMLP' in r['config']:
            try:
                out.add(int(r['config'].split('-N=')[1]))
            except Exception:
                pass
    return out

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

t0 = time.time()
have = done_N()
puf = ArbiterPUF(n=64, seed=0, noisiness=0.0)
Ctr_full = np.random.default_rng(11).integers(0, 2, size=(1000000, 64)).astype(np.uint8)
Cte = np.random.default_rng(12).integers(0, 2, size=(10000, 64)).astype(np.uint8)
ytr_full = gen_labels(puf, Ctr_full)
yte = gen_labels(puf, Cte)
Xte_par = parity_feats(Cte).astype(np.float32)
Xte_mapped = parity_feats(sbox_map(Cte))
newfile = not os.path.exists(CSV)
f = open(CSV, 'a', newline='')
w = csv.writer(f)
if newfile: w.writerow(['test', 'config', 'instance', 'acc', 'train_acc'])
for N in NGRID:
    if N in have:
        print('skip', N, flush=True); continue
    Xtr_par = parity_feats(Ctr_full[:N]).astype(np.float32)
    c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=0)
    c.fit(parity_feats(sbox_map(Ctr_full[:N])), ytr_full[:N])
    t1 = round(float(c.score(Xte_mapped, yte)), 4)
    tr, te = bigmlp_par(Xtr_par, ytr_full[:N], Xte_par, yte, 0)
    te, tr = round(te, 4), round(tr, 4)
    w.writerow(['E5par', f'T1par-mapped-N={N}', 0, t1, '']); f.flush()
    w.writerow(['E5par', f'parMLP-N={N}', 0, te, tr]); f.flush()
    print(f'E5par N={N} T1mapped={t1} parMLP test={te} train={tr}', flush=True)
f.close()
print('elapsed', round(time.time()-t0, 1))
