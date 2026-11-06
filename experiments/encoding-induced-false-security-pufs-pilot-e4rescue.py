"""E4 launchpad rescue: MATLAB-default-faithful configs (MSE loss, minibatch Adam).
If plain-32 crosses ~0.9, rerun substitution with the winning config; else E4
launchpad stays BLOCKED with tested configs (documented, not claimed).
"""
import time, numpy as np, torch
from pypuf.simulation import ArbiterPUF

def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)

t0 = time.time()
puf = ArbiterPUF(n=32, seed=0, noisiness=0.0)
Ctr = np.random.default_rng(11).integers(0, 2, size=(30000, 32)).astype(np.uint8)
Cte = np.random.default_rng(12).integers(0, 2, size=(10000, 32)).astype(np.uint8)
ytr = labels(puf, Ctr).astype(np.float32); yte = labels(puf, Cte)
Xtr = (Ctr.astype(np.float32)*2-1); Xte = (Cte.astype(np.float32)*2-1)

for H, epochs in [(64, 200), (256, 120)]:
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(32, H), torch.nn.Tanh(), torch.nn.Linear(H, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    lossf = torch.nn.MSELoss()
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr).view(-1, 1)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt),
        batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(0))
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
    with torch.no_grad():
        pred = (net(torch.tensor(Xte)) > 0.5).float().numpy().ravel()
    print(f'MSE-Adam-H{H}-ep{epochs}:', round(float((pred == yte).mean()), 4), flush=True)
print('elapsed', round(time.time()-t0, 1))
