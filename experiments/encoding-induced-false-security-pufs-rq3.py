"""RQ3: Sj/I2O1 raw vs de-obfuscated + Adv(N) vs learner accuracy.
Sj(P) = Pr_c[ f(c^j) != f(c) ] over uniform c; I2O1 = mean_j|Sj-0.5| (Stefani et al. ePrint 2024/378).
Designs (64-bit APUF, torch GPU batched): plain; S-box-APUF measured on RAW external c;
S-box-APUF measured DE-OBFUSCATED (internal pi(c)).
Adv(N): Fei-style grouping estimator (N small: 1..16, P=200k instances, Meval fresh).
Learner curve: Gate1 CSV (plain-parity LR) + small-N LR extension computed here.
"""
import time, csv, numpy as np, torch
dev = 'cuda'
torch.manual_seed(0)

def parity_mat(C):
    # C: (M,64) +-1 -> X (65,M)
    S = C
    X = torch.ones(65, C.shape[0], device=C.device)
    suf = torch.ones(C.shape[0], device=C.device)
    for i in range(63, -1, -1):
        suf = suf * S[:, i]
        X[i] = suf
    return X

def sbox_bytes_torch(B, lut):
    return lut[B]

def make_pi():
    src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
    ns = {}
    exec(src.split('INSTANCES =')[0], ns)
    return torch.tensor(list(ns['SBOX_ARR']), dtype=torch.long, device=dev)

LUT = make_pi()
def pi_torch(C01):
    W8 = (2**torch.arange(7, -1, -1, device=dev, dtype=torch.long))
    B = (C01.view(-1, 8, 8).long() * W8).sum(-1)
    return ((LUT[B].unsqueeze(-1) >> torch.arange(7, -1, -1, device=dev).view(1, 1, 8)) & 1).reshape(-1, 64).float()

def apuf_resp(W, X):
    return torch.sign(W @ X)

# ---------- Sj/I2O1 ----------
def sens(P=10, K=5000, design='plain', seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    ball = torch.randint(0, 2, (P, 65), generator=g, device=dev).float() * 2 - 1
    W = ball
    C = (torch.randint(0, 2, (K, 64), generator=g, device=dev).float())
    Cpm = C * 2 - 1
    if design == 'plain':
        Ci = Cpm
    else:
        Ci = pi_torch(C) * 2 - 1
    X = parity_mat(Ci)  # (65,K)
    R = torch.sign(W @ X)  # (P,K)
    Sj = torch.zeros(P, 64, device=dev)
    for j in range(64):
        Cj = Cpm.clone(); Cj[:, j] *= -1
        if design != 'plain':
            # flip external bit j, re-apply pi: need bits
            Cb = ((Cj + 1) / 2)
            Cj = pi_torch(Cb) * 2 - 1
        Rj = torch.sign(W @ parity_mat(Cj))
        Sj[:, j] = (Rj != R).float().mean(1)
    I2 = (Sj - 0.5).abs().mean(1)
    return Sj.mean(0).cpu().numpy(), float(I2.mean()), float(I2.std())

rows = []
for design in ['plain', 'sbox-raw', 'sbox-deobf']:
    d = 'plain' if design == 'plain' else ('sbox-raw' if design == 'sbox-raw' else 'plain')
    # for deobf: measure sensitivity of f w.r.t. internal coords = same as plain profile; compute directly
    Sj, m, s = sens(design='plain' if design != 'sbox-raw' else 'sbox', seed=1)
    if design == 'sbox-deobf':
        note = 'internal-coords(=plain profile)'
    else:
        note = 'external-coords'
    rows.append((design, note, round(m, 4), round(s, 4), [round(float(x), 3) for x in Sj[::8]]))
    print(design, 'I2O1 mean', round(m, 4), 'std', round(s, 4), 'Sj every-8th:', [round(float(x), 3) for x in Sj], flush=True)
with open('experiments/encoding-induced-false-security-pufs-rq3-sens.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['design', 'coords', 'I2O1_mean', 'I2O1_std', 'Sj_sample'])
    w.writerows(rows)
