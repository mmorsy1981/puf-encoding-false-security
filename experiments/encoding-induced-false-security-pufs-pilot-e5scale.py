"""E5 big-MLP scaling — vetted PROCEED as scoped (20k-1M window unmapped).
Uninformed learner (raw external bits -> sbox-APUF labels) vs T1 parity-LR
control per N. Architecture (documented choice, Mursi-style deep MLP):
64 -> 512 -> 256 -> 128 -> 1, ReLU, Adam 1e-3, batch 4096, 40 fixed epochs,
BCEWithLogits, seeds fixed. 64-bit APUF, S-box = gate1 table, nested subsets of
1M train, 10k disjoint test. Writes incrementally (append+flush per N) so a
timeout never loses finished N. Censoring preregistered: if big-MLP stays ~0.5
through 1M, margin is ">1M" (lower bound, not security).
"""
import time, csv, os, numpy as np, torch
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression

CSV = 'experiments/encoding-induced-false-security-pufs-pilot-e5scale.csv'
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
        if r['test'] == 'E5' and 'bigMLP' in r['config']:
            out.add(int(r['config'].split('-N=')[1].replace('k', '000').replace('M', '000000')))
    return out

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
have = done_N()
puf = ArbiterPUF(n=64, seed=0, noisiness=0.0)
Ctr_full = np.random.default_rng(11).integers(0, 2, size=(1000000, 64)).astype(np.uint8)
Cte = np.random.default_rng(12).integers(0, 2, size=(10000, 64)).astype(np.uint8)
ytr_full = gen_labels(puf, Ctr_full)
yte = gen_labels(puf, Cte)
newfile = not os.path.exists(CSV)
f = open(CSV, 'a', newline='')
w = csv.writer(f)
if newfile: w.writerow(['test', 'config', 'instance', 'acc', 'train_acc'])
for N in NGRID:
    if N in have:
        print('skip', N, flush=True); continue
    Xtr = (Ctr_full[:N].astype(np.float32)*2-1); Xte = (Cte.astype(np.float32)*2-1)
    c = LogisticRegression(max_iter=1000, solver='liblinear', random_state=0)
    c.fit(parity_feats(Ctr_full[:N]), ytr_full[:N])
    t1 = round(float(c.score(parity_feats(Cte), yte)), 4)
    tr, te = bigmlp(Xtr, ytr_full[:N], Xte, yte, 0)
    te, tr = round(te, 4), round(tr, 4)
    w.writerow(['E5', f'T1par-N={N}', 0, t1, '']); f.flush()
    w.writerow(['E5', f'bigMLP-N={N}', 0, te, tr]); f.flush()
    print(f'E5 N={N} T1={t1} bigMLP test={te} train={tr}', flush=True)
f.close()
print('elapsed', round(time.time()-t0, 1))
