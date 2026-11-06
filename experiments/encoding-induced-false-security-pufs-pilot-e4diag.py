"""E4 diagnostic: is the launchpad failure data or training-config?
Same plain-32 30k data as the pilot. Checks: (1) parity-LR (data sanity — must be
~0.99 if sim+labels are right); (2) sklearn MLPClassifier/adam (independent
learner cross-check); (3) torch Rprop longer + minibatch (config fix attempt).
"""
import time, numpy as np, torch
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def parity_feats(C):
    S = 1 - 2*C.astype(float); X = np.ones((len(C), 33))
    for i in range(32): X[:, i] = np.prod(S[:, i:], axis=1)
    return X

t0 = time.time()
puf = ArbiterPUF(n=32, seed=0, noisiness=0.0)
Ctr = np.random.default_rng(11).integers(0, 2, size=(30000, 32)).astype(np.uint8)
Cte = np.random.default_rng(12).integers(0, 2, size=(10000, 32)).astype(np.uint8)
ytr = labels(puf, Ctr); yte = labels(puf, Cte)
Xtr = Ctr.astype(float)*2-1; Xte = Cte.astype(float)*2-1

c = LogisticRegression(max_iter=1000, solver='liblinear').fit(parity_feats(Ctr), ytr)
print('parity-LR:', round(float(c.score(parity_feats(Cte), yte)), 4), flush=True)

m = MLPClassifier((100,), activation='tanh', max_iter=200, random_state=0)
m.fit(Xtr, ytr)
print('sklearn-MLP100-adam:', round(float(m.score(Xte, yte)), 4), 'iters:', m.n_iter_, flush=True)

for epochs, bs, lr in [(300, 30000, 0.01), (100, 512, 0.001)]:
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(32, 10), torch.nn.Tanh(), torch.nn.Linear(10, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(0))
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
    with torch.no_grad():
        pred = (net(torch.tensor(Xte, dtype=torch.float32)) > 0).float().numpy().ravel()
    print(f'torch-Adam-H10-ep{epochs}-bs{bs}-lr{lr}:', round(float((pred == yte).mean()), 4), flush=True)
print('elapsed', round(time.time()-t0, 1))
