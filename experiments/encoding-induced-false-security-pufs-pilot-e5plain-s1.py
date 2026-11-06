"""E5 plain seed-1 control (V4-fix slice 2).
Same big-MLP as pilot-e5scale.py (64->512->256->128->1, ReLU, Adam 1e-3,
batch 4096, 40 epochs) on PLAIN-APUF raw bits (no S-box): labels f(C_ext).
Settings mirror pilot-e5scale-s1.py: puf seed 1, train rng 111 (1M), test rng
112 (10k disjoint), torch manual seed 1. Same N grid nested subsets.
Per N also files the plain parity-LR baseline (expect ~0.99+).
Writes experiments/encoding-induced-false-security-pufs-pilot-e5plain-s1.csv
cols: test,config,instance,acc,train_acc. Incremental append+flush per N.
"""
import time, csv, os, numpy as np, torch
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

CSV = 'experiments/encoding-induced-false-security-pufs-pilot-e5plain-s1.csv'
NGRID = [20000, 50000, 100000, 200000, 500000, 1000000]
EPOCHS, BS, LR = 40, 4096, 1e-3

def load_gate1():
    src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
    ns = {}
    exec(src.split(chr(10) + "if __name__ == '__main__':")[0], ns)
    return ns
G = load_gate1()
parity_feats, labels = G['parity_feats'], G['labels']

def done_N():
    if not os.path.exists(CSV): return set()
    out = set()
    for r in csv.DictReader(open(CSV)):
        if r['test'] == 'E5plains1' and 'bigMLP-plain' in r['config']:
            try:
                out.add(int(r['config'].split('-N=')[1]))
            except Exception:
                pass
    return out

def bigmlp(Xtr, ytr, Xte, yte, seed):
    torch.manual_seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
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

def gen_labels(puf, C):
    n = len(C); y = np.zeros(n, dtype=int)
    for s in range(0, n, 100000):
        y[s:s+100000] = labels(puf, C[s:s+100000])
    return y

t0 = time.time()
have = done_N()
puf = ArbiterPUF(n=64, seed=1, noisiness=0.0)
Ctr_full = np.random.default_rng(111).integers(0, 2, size=(1000000, 64)).astype(np.uint8)
Cte = np.random.default_rng(112).integers(0, 2, size=(10000, 64)).astype(np.uint8)
ytr_full = gen_labels(puf, Ctr_full)
yte = gen_labels(puf, Cte)
Xte = (Cte.astype(np.float32)*2-1)
Xte_par = parity_feats(Cte)
newfile = not os.path.exists(CSV)
f = open(CSV, 'a', newline='')
w = csv.writer(f)
if newfile: w.writerow(['test', 'config', 'instance', 'acc', 'train_acc'])
for N in NGRID:
    if N in have:
        print('skip', N, flush=True); continue
    Xtr = (Ctr_full[:N].astype(np.float32)*2-1)
    c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=1)
    c.fit(parity_feats(Ctr_full[:N]), ytr_full[:N])
    t1 = round(float(c.score(Xte_par, yte)), 4)
    tr, te = bigmlp(Xtr, ytr_full[:N], Xte, yte, 1)
    te, tr = round(te, 4), round(tr, 4)
    w.writerow(['E5plains1', f'T1par-plain-N={N}', 1, t1, '']); f.flush()
    w.writerow(['E5plains1', f'bigMLP-plain-N={N}', 1, te, tr]); f.flush()
    print(f'E5plains1 N={N} T1plain={t1} bigMLP test={te} train={tr}', flush=True)
f.close()
print('elapsed', round(time.time()-t0, 1))
