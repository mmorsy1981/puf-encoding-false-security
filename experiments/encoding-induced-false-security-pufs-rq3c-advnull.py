"""Adv null-corrected (Review-3 issue D): null-subtracted Adv over >=10 challenge draws, M<=8.
Same estimator as rq3b.py (group by response pattern, size>=2, mean|group-mean| over Meval=20 evals).
Null: same group structure, responses shuffled across instances (fresh torch randperm per M).
CSV cols: draw,M,Adv,null,excess,ngroups. Summary prints mean excess per M.
"""
import csv, numpy as np, torch
dev = 'cuda'

def parity_mat(C):
    X = torch.ones(65, C.shape[0], device=C.device)
    suf = torch.ones(C.shape[0], device=C.device)
    for i in range(63, -1, -1):
        suf = suf * C[:, i]
        X[i] = suf
    return X

def adv_null(P=200000, M=8, Meval=20, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    W = torch.randn(P, 65, generator=g, device=dev)
    Ck = torch.randint(0, 2, (M, 64), generator=g, device=dev).float() * 2 - 1
    Ce = torch.randint(0, 2, (Meval, 64), generator=g, device=dev).float() * 2 - 1
    Rk = torch.sign(W @ parity_mat(Ck))
    Re = torch.sign(W @ parity_mat(Ce))
    key = (Rk > 0).to(torch.uint8).cpu().numpy()
    buf = np.zeros((P, (M + 7) // 8), dtype=np.uint8)
    for i in range(M):
        buf[:, i // 8] |= (key[:, i] << (i % 8))
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(P):
        groups[buf[i].tobytes()].append(i)
    Re_np = Re.cpu().numpy()
    Re_sh = Re_np[np.random.default_rng(seed).permutation(P)]
    tot = tot0 = 0.0; ng = 0
    for idxs in groups.values():
        if len(idxs) >= 2:
            tot += np.abs(Re_np[idxs].mean(axis=0)).mean()
            tot0 += np.abs(Re_sh[idxs].mean(axis=0)).mean()
            ng += 1
    return round(float(tot / ng), 4), round(float(tot0 / ng), 4), ng

rows = []
for draw in range(12):
    for M in (1, 2, 4, 8):
        a, a0, ng = adv_null(M=M, seed=1000 + draw)
        rows.append((draw, M, a, a0, round(a - a0, 4), ng))
        print('draw', draw, 'M', M, 'Adv', a, 'null', a0, 'excess', round(a - a0, 4), 'groups', ng, flush=True)
with open('experiments/encoding-induced-false-security-pufs-rq3c-advnull.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['draw', 'M', 'Adv', 'null', 'excess', 'ngroups']); w.writerows(rows)
ex = {}
for draw, M, a, a0, e, ng in rows:
    ex.setdefault(M, []).append(e)
print({M: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4), len(v)) for M, v in sorted(ex.items())} if False else {M: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) for M, v in sorted(ex.items())})
