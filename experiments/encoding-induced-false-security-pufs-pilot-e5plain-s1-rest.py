"""E5 plain seed-1 resume (500k/1M only; 20k/50k/100k already in pilot-e5plain-s1.csv).
Identical seeds (puf 1, rng 111/112, torch 1) and nesting (Ctr_full[:N]) — byte-identical
regen of the same challenges. Writes pilot-e5plain-s1-rest.csv (same cols).
"""
import time, csv, numpy as np, torch
from pypuf.simulation import ArbiterPUF

t0 = time.time()
puf = ArbiterPUF(n=64, seed=1, noisiness=0.0)
Ctr_full = np.random.default_rng(111).integers(0, 2, size=(1000000, 64)).astype(np.uint8)
Cte = np.random.default_rng(112).integers(0, 2, size=(10000, 64)).astype(np.uint8)
def lab(C):
    n = len(C); y = np.zeros(n, dtype=int)
    for s in range(0, n, 100000):
        y[s:s+100000] = ((puf.eval((2*C[s:s+100000].astype(np.int8)-1)).reshape(-1)+1)//2).astype(int)
    return y
ytr_full = lab(Ctr_full); yte = lab(Cte)
def bigmlp(Xtr, ytr, Xte, yte, seed):
    torch.manual_seed(seed); dev = 'cuda'
    net = torch.nn.Sequential(torch.nn.Linear(64, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt), batch_size=4096, shuffle=True, generator=torch.Generator().manual_seed(seed))
    for _ in range(40):
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
    with torch.no_grad():
        tr = (net(torch.tensor(Xtr, dtype=torch.float32).to(dev)) > 0).float().cpu().numpy().ravel()
        pr = (net(torch.tensor(Xte, dtype=torch.float32).to(dev)) > 0).float().cpu().numpy().ravel()
    return float((tr == ytr).mean()), float((pr == yte).mean())
with open('experiments/encoding-induced-false-security-pufs-pilot-e5plain-s1-rest.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test', 'config', 'instance', 'acc', 'train_acc'])
    for N in [500000, 1000000]:
        Xtr = (Ctr_full[:N].astype(np.float32)*2-1); Xte = (Cte.astype(np.float32)*2-1)
        tr, te = bigmlp(Xtr, ytr_full[:N], Xte, yte, 1)
        w.writerow(['E5plains1', f'bigMLP-plain-N={N}', 1, round(te, 4), round(tr, 4)]); f.flush()
        print(f'E5plains1-resume N={N} test={round(te,4)} train={round(tr,4)}', flush=True)
print('elapsed', round(time.time()-t0, 1))
