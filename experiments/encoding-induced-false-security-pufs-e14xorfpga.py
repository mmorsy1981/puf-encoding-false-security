"""E14: encoding layers in front of REAL XOR Arbiter PUFs (measured 4-XOR and 5-XOR data).
Written and preregistered 2026-10-24, before any E14 run (only data sanity checks, listed below, ran).

Why. E8 replayed single APUFs; E11/E12 studied XOR bases in simulation only. This replay puts the AES S-box in front of
measured 4-XOR and 5-XOR APUFs (Mursi et al. 2020, from the pypuf data collection, 1M uniform challenges each, one
measurement per challenge). Corollary 1 makes the replay exact: (S^-1(c), r) are CRPs of "secret/public S-box, then the
same physical XOR PUF". Consequently the known-table (informed, T1) attack IS the plain attack on the measured data.
Data checks run before writing this file: challenges are +-1 int8 (1M x 64), responses +-1, challenge bit means 0.500,
majority rates 0.517 (4-XOR) and 0.502 (5-XOR).
Split as E8: last 100k = test, training prefix N, last 10% of the prefix = validation. gate1 conventions: bits = (c+1)/2,
labels = (r+1)/2. Learner family = E11's: E7 table features -> k linear heads -> soft XOR (prod of tanh), Adam LR 0.02,
batch 4096, >= 6000 steps and >= 20 epochs, BCE. GPU, seed 1000*k + r (+ 7 for the secret table). Every restart logged.

Modes (python ...-e14xorfpga.py <mode>):
  known  : true table fixed (informed attack = plain attack). k in {4, 5}, N in {20k, 50k, 100k, 200k, 500k, 900k}, restarts 0-2.
  secret : learned shared 256x8 table (T2), k in {4, 5}, N in {200k, 500k, 900k}, restarts 0-4, validation-selected best-of-5.
  t0     : uninformed parity-input MLP (65-512-256-128-1, Adam 1e-3, batch 4096, 40 epochs) on the external challenge,
           k in {4, 5}, N = 900k, one seed.
Success for the secret-table attack = accuracy >= known-table accuracy at the same k and N minus 0.05 (as E8).

Predictions (before any E14 run; author's prior in brackets):
  Q1 Known-table accuracy rises monotonically with N for both k (no drop of more than 0.01 between consecutive N,
     mean over restarts) and the 4-XOR curve lies above the 5-XOR curve at every N.                      [0.80]
  Q2 t0 stays within 0.03 of the majority rate for both k.                                                 [0.90]
  Q3 secret: best-of-5 stays within 0.03 of the majority rate at every N for both k (the attack does not
     succeed on measured 4-/5-XOR with up to 900k CRPs, as E11 found for 3-XOR in simulation).           [0.85]
A failure of the secret-table attack is a learner limit and is reported as one; the known-table rows are the primary result.
Writes experiments/encoding-induced-false-security-pufs-e14xorfpga.csv (append + resume)
cols: test,mode,device,variant,N,restart,acc,val_acc,maj,elapsed_s,hw
"""
import csv, os, sys, time, numpy as np, torch

E11 = {'__name__': 'e14_import'}
exec(open('experiments/encoding-induced-false-security-pufs-e11gen.py').read(), E11)
E13 = {'__name__': 'e14_import13'}
exec(open('experiments/encoding-induced-false-security-pufs-e13learners.py').read(), E13)
DEV, fit, accm, build, suffix_mask = E11['DEV'], E11['fit'], E11['accm'], E11['build'], E11['suffix_mask']
SBOX, bits_to_bytes = E11['SBOX'], E11['E7']['bits_to_bytes']
CSV = 'experiments/encoding-induced-false-security-pufs-e14xorfpga.csv'
DATA = {'xor4': ('data/fpga/MTZAA20_4XOR_1M.npz', 4), 'xor5': ('data/fpga/MTZAA20_5XOR_1M.npz', 5)}
NTEST = 100000
COLS = ['test', 'mode', 'device', 'variant', 'N', 'restart', 'acc', 'val_acc', 'maj', 'elapsed_s', 'hw']
HW = torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'


def load(dev):
    z = np.load(DATA[dev][0], allow_pickle=False)
    c01 = ((z['challenges'].astype(np.int16) + 1) // 2).astype(np.uint8)
    y = ((z['responses'].reshape(-1).astype(np.int16) + 1) // 2).astype(np.float32)
    inv = np.argsort(SBOX).astype(np.uint8)
    xb = inv[bits_to_bytes(c01)]
    return xb, y


def split(xb, y, N):
    T = lambda a, dt: torch.tensor(a, dtype=dt, device=DEV)
    X, Y = T(xb.astype(np.int64), torch.int64), T(y, torch.float32)
    nva = N // 10
    return X[:N - nva], Y[:N - nva], X[N - nva:N], Y[N - nva:N], X[-NTEST:], Y[-NTEST:]


def done():
    if not os.path.exists(CSV):
        return set()
    return {(r['mode'], r['device'], r['variant'], int(r['N']), int(r['restart'])) for r in csv.DictReader(open(CSV))}


def main(mode):
    have = done()
    new = not os.path.exists(CSV)
    f = open(CSV, 'a', newline='')
    w = csv.writer(f)
    if new:
        w.writerow(COLS)
    for dev, (path, k) in DATA.items():
        xb, y = load(dev)
        maj = float(max(y[-NTEST:].mean(), 1 - y[-NTEST:].mean()))
        plan = {'known': ('known', (20000, 50000, 100000, 200000, 500000, 900000), 3),
                'secret': ('shared', (200000, 500000, 900000), 5),
                't0': ('parityMLP', (900000,), 1)}[mode]
        variant, Ns, R = plan
        for N in Ns:
            for r in range(R):
                if (mode, dev, variant, N, r) in have:
                    continue
                xtr, ytr, xva, yva, xte, yte = split(xb, y, N)
                t0 = time.time()
                if mode == 't0':
                    gen = torch.Generator().manual_seed(1000 * k)
                    m = E13['ParityMLP']((512, 256, 128), gen).to(DEV)
                    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
                    lossf = torch.nn.BCEWithLogitsLoss()
                    n = len(xtr)
                    for _ in range(40):
                        perm = torch.randperm(n, generator=gen).to(DEV)
                        for s0 in range(0, n, 4096):
                            idx = perm[s0:s0 + 4096]
                            opt.zero_grad()
                            lossf(m(xtr[idx]), ytr[idx]).backward()
                            opt.step()
                    va = accm(m, xva, yva)
                else:
                    seed = 1000 * k + r + (7 if mode == 'secret' else 0)
                    m, gen = build('mask', 'known' if mode == 'known' else 'shared', seed, A=suffix_mask(), heads=k)
                    m, va = fit(m, gen, xtr, ytr, xva, yva)
                acc = accm(m, xte, yte)
                row = ['E14', mode, dev, variant, N, r, round(acc, 4), round(va, 4), round(maj, 4), round(time.time() - t0, 1), HW]
                w.writerow(row)
                f.flush()
                print(*row, flush=True)
    f.close()


if __name__ == '__main__':
    for mode in sys.argv[1:] or ['known', 't0', 'secret']:
        main(mode)
