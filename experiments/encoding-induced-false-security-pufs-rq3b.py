"""RQ3 part 2: Fei-style Adv(N) grouping estimator vs LR accuracy; plot both.
Adv(N): P instances, M known CRPs; group by response pattern (size>=2);
fresh challenges; Adv = mean over groups of mean|group-mean response|.
P=200k, M in {1,2,4,8,12,16}, Meval=20. GPU batched.
LR: sklearn liblinear parity-LR at small N {8,16,32,64,128,256} (2 inst) + Gate1 CSV curve.
Plot: Adv(N) and (acc-0.5) vs N (log x). PNG out.
"""
import csv, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
dev = 'cuda'
torch.manual_seed(0)

def parity_mat(C):
    X = torch.ones(65, C.shape[0], device=C.device)
    suf = torch.ones(C.shape[0], device=C.device)
    for i in range(63, -1, -1):
        suf = suf * C[:, i]
        X[i] = suf
    return X

def adv_curve(P=200000, Ms=(1, 2, 4, 8, 12, 16), Meval=20, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    W = (torch.randn(P, 65, generator=g, device=dev))
    out = {}
    for M in Ms:
        Ck = torch.randint(0, 2, (M, 64), generator=g, device=dev).float() * 2 - 1
        Ce = torch.randint(0, 2, (20, 64), generator=g, device=dev).float() * 2 - 1
        Rk = torch.sign(W @ parity_mat(Ck))  # (P,M) +-1
        Re = torch.sign(W @ parity_mat(Ce))  # (P,20)
        key = (Rk > 0).to(torch.uint8).cpu().numpy()
        buf = np.zeros((P, (M + 7) // 8), dtype=np.uint8)
        for i in range(M):
            buf[:, i // 8] |= (key[:, i] << (i % 8))
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(P):
            groups[buf[i].tobytes()].append(i)
        biases = []
        Re_np = Re.cpu().numpy()
        for idxs in groups.values():
            if len(idxs) >= 2:
                biases.append(np.abs(Re_np[idxs].mean(axis=0)).mean())
        out[M] = (round(float(np.mean(biases)), 4), len(biases))
        print('M', M, 'Adv', out[M][0], 'groups', out[M][1], flush=True)
    return out

adv = adv_curve()
with open('experiments/encoding-induced-false-security-pufs-rq3-adv.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['M', 'Adv', 'ngroups'])
    for M, (a, n) in adv.items(): w.writerow([M, a, n])

# learner small-N (sklearn, CPU, pypuf sims)
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def pfeat(C):
    S = 1 - 2*C.astype(float); X = np.ones((len(C), 65))
    for i in range(64): X[:, i] = np.prod(S[:, i:], axis=1)
    return X
lr_small = []
for inst in [0, 1]:
    puf = ArbiterPUF(n=64, seed=inst, noisiness=0.0)
    rt = np.random.default_rng(77000+inst)
    Cte = rt.integers(0, 2, size=(5000, 64)).astype(np.uint8); yte = labels(puf, Cte)
    r = np.random.default_rng(78000+inst)
    Ctr = r.integers(0, 2, size=(256, 64)).astype(np.uint8); ytr = labels(puf, Ctr)
    for N in [8, 16, 32, 64, 128, 256]:
        c = LogisticRegression(max_iter=1000, solver='liblinear').fit(pfeat(Ctr[:N]), ytr[:N])
        lr_small.append((N, round(float(c.score(pfeat(Cte), yte)), 4)))
        print('LR N', N, 'inst', inst, round(float(c.score(pfeat(Cte), yte)), 4), flush=True)
with open('experiments/encoding-induced-false-security-pufs-rq3-lrsmall.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['N', 'acc']); w.writerows(lr_small)

# plot: Adv(N) + (LRacc-0.5), log-x
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(list(adv.keys()), [v[0] for v in adv.values()], 'o-', label='Adv(N) grouping (P=200k)')
xs = sorted(set(n for n, _ in lr_small))
ys = {}
for n, a in lr_small: ys.setdefault(n, []).append(a)
ax.plot(xs, [np.mean(ys[n]) - 0.5 for n in xs], 's--', label='LR acc-0.5 small-N (2 inst)')
gate = {}
with open('experiments/encoding-induced-false-security-pufs-gate1.csv') as f:
    for r in __import__('csv').DictReader(f):
        if r['tier'] == 'plain-parity' and int(r['instance']) < 2:
            gate.setdefault(int(r['N']), []).append(float(r['acc']))
gx = sorted(gate)
ax.plot(gx, [np.mean(gate[n]) - 0.5 for n in gx], '^-.', label='LR acc-0.5 Gate1 (2 inst avg)')
ax.set_xscale('log'); ax.set_xlabel('N (observed CRPs, log scale)')
ax.set_ylabel('advantage / (accuracy - 0.5)')
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig('experiments/encoding-induced-false-security-pufs-rq3.png', dpi=120)
print('PLOT-OK')
