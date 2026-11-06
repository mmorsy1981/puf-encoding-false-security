"""R3: plain raw-MLP seed 0 (audit residual). Same arch/epochs/seeds as e5scale.py
but labels from the PLAIN APUF (no S-box). puf seed 0, Ctr rng 11, Cte rng 12,
NGRID nested subsets of 1M, test 10k. Only bigMLP-plain rows (T1-plain covered
by pilot-e5plainparmlp.csv). Append+resume per N.
Writes experiments/encoding-induced-false-security-pufs-pilot-e5plainraw-s0.csv
cols: test,config,instance,acc,train_acc
"""
import time, csv, os, numpy as np, torch
from pypuf.simulation import ArbiterPUF
CSV = 'experiments/encoding-induced-false-security-pufs-pilot-e5plainraw-s0.csv'
NGRID = [20000, 50000, 100000, 200000, 500000, 1000000]
EPOCHS, BS, LR = 40, 4096, 1e-3
def load_gate1():
    src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
    ns = {}
    exec(src.split(chr(10) + "if __name__ == '__main__':")[0], ns)
    return ns
G = load_gate1()
to_pm, labels = G['to_pm'], G['labels']
def done_N():
    if not os.path.exists(CSV): return set()
    out = set()
    for r in csv.DictReader(open(CSV)):
        if 'bigMLP-plain-N=' in r['config']:
            out.add(int(r['config'].split('-N=')[1]))
    return out
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
have = done_N()
puf = ArbiterPUF(n=64, seed=0, noisiness=0.0)
Ctr_full = np.random.default_rng(11).integers(0, 2, size=(1000000, 64)).astype(np.uint8)
Cte = np.random.default_rng(12).integers(0, 2, size=(10000, 64)).astype(np.uint8)
ytr_full = labels(puf, Ctr_full).reshape(-1)
yte = labels(puf, Cte).reshape(-1)
newfile = not os.path.exists(CSV)
f = open(CSV, 'a', newline='')
w = csv.writer(f)
if newfile: w.writerow(['test', 'config', 'instance', 'acc', 'train_acc'])
for N in NGRID:
    if N in have:
        print('skip', N, flush=True); continue
    Xtr = (Ctr_full[:N].astype(np.float32)*2-1); Xte = (Cte.astype(np.float32)*2-1)
    tr, te = bigmlp(Xtr, ytr_full[:N], Xte, yte, 0)
    te, tr = round(te, 4), round(tr, 4)
    w.writerow(['E5plainraw', f'bigMLP-plain-N={N}', 0, te, tr]); f.flush()
    print(f'E5plainraw N={N} bigMLP test={te} train={tr}', flush=True)
f.close()
print('elapsed', round(time.time()-t0, 1))
