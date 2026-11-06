"""E7b: follow-ups to E7 (structure-aware learning of an unknown bytewise S-box).
Written and preregistered 2026-10-20 after E7 instance 0 showed a sharp, bimodal transition
(shared table 0.61 @20k -> 0.97 @30k; per-byte 0.65 @60k -> 0.98 @100k): runs either find the
solution basin (>=0.97) or stall near 0.6. Reuses E7's model, data seeds and hyperparameters
(LR 0.02, batch 4096, >=6000 steps / >=20 epochs, Adam, BCE, 10% validation hold-out).
CPU, torch threads pinned to 14 (as E7 ran on this 16-core host). NOT bitwise reproducible at 14
threads: a same-seed rerun drifts by <=0.001 acc (multithreaded float accumulation; checked
post hoc: 1 thread is deterministic, 14 threads is not). Reproduction check vs E7: 13/21 rows
exact, 8 within 0.001, success/failure class identical on 21/21.

Modes, run in this order (python ...-e7b.py corrupt|ext|random|noise):
  corrupt : init the table from the TRUE G (magnitude 1, tanh(1)=0.76) with a fraction p of
            entries sign-flipped, p in {0, 0.10, 0.25, 0.50}; shared @20k and perbyte @60k
            (the failing cells), instances 0-2, one run each. p=0.5 is uncorrelated with the
            truth (control equivalent to a random-sign init).
  ext     : 10 instances x 3 restarts, EVERY restart logged (no best-of selection).
            shared N in {20k,30k,40k,60k}; perbyte N in {60k,100k,150k,200k}.
            (Grid widened before any E7b run, once the full E7 fine grid showed per-instance
            shared thresholds of 20-30k / 40-60k / 30-40k and perbyte inst 2 failing at 100k,
            succeeding at 200k; the first draft grid {20,25,30,40k} / {60,80,100,150k} would have
            censored those instances. E7 transitions are mostly bimodal; one intermediate cell,
            perbyte inst 1 @60k = 0.72.)
            Restart r on instance s uses seed 1000*s + r, as E7, so E7's best-of-3 rows at
            (inst 0-2, 20k) must reappear among these restarts (reproduction check).
  random  : per-device random secret S-box (rng 5000+s permutation of 0..255) instead of AES;
            instances 0-4, N in {30k, 100k}; shared x 3 restarts (all logged) + known control.
  noise   : AES S-box, ArbiterPUF noisiness=0.1 (same weights as the noiseless twin; measured
            clean-vs-noisy agreement 0.974); train on noisy labels; report test accuracy vs
            noisy labels (acc) and vs the noiseless twin (acc_clean); instances 0-4,
            N in {30k, 100k}; shared x 3 restarts + known control.
Success = test accuracy >= 0.95 (acc_clean for noise mode).

Predictions (before any E7b run):
  corrupt : shared@20k and perbyte@60k repaired (>=0.95) from p=0.10 and p=0.25 on >=2/3
            instances, not from p=0.50 -> the failures are basin-finding (optimisation), and
            the information to learn the table is present below the observed threshold.
            Falsified if p=0.10 also fails (then the failing N is information-limited).
  ext     : per-restart success rate rises from <=0.2 to >=0.8 across the shared grid
            (20k -> 60k) and across the perbyte grid (60k -> 200k).
  random  : same behaviour as AES: shared best-of-3 succeeds at 100k on 5/5 and at 30k on >=3/5.
  noise   : shared best-of-3 at 100k reaches acc_clean >= 0.95 on >=4/5, with noisy-label acc
            within 1.5pp of the 0.974 agreement ceiling; 30k may fail (noise raises the threshold).
Writes experiments/encoding-induced-false-security-pufs-pilot-e7b.csv (append + resume)
cols: test,mode,variant,instance,N,restart,init,acc,acc_clean,val_acc,elapsed_s
"""
import csv, os, sys, time, numpy as np, torch
from pypuf.simulation import ArbiterPUF

E7 = {'__name__': 'e7b_import'}
exec(open('experiments/encoding-induced-false-security-pufs-pilot-e7sboxlearn.py').read(), E7)
torch.set_num_threads(14)
CSV = 'experiments/encoding-induced-false-security-pufs-pilot-e7b.csv'
Structured, acc = E7['Structured'], E7['acc']
labels, bits_to_bytes, bytes_to_bits = E7['labels'], E7['bits_to_bytes'], E7['G1']['bytes_to_bits']
SBOX_AES, G_AES = E7['SBOX'], E7['G_TRUE']
LR, BS, MIN_STEPS, MIN_EPOCHS = E7['LR'], E7['BS'], E7['MIN_STEPS'], E7['MIN_EPOCHS']
SUCCESS = 0.95


def g_table(sbox):
    s_out = 1 - 2 * np.unpackbits(sbox.reshape(-1, 1), axis=1).astype(np.float64)
    return np.stack([np.prod(s_out[:, k:], axis=1) for k in range(8)], axis=1)


assert np.array_equal(g_table(SBOX_AES), G_AES)


def fit(model, gen, xtr, ytr, xva, yva):
    """Identical loop to E7's train(), on a model built by the caller."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(xtr)
    epochs = max(MIN_EPOCHS, -(-MIN_STEPS // max(1, n // BS)))
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, BS):
            idx = perm[s:s + BS]
            opt.zero_grad()
            lossf(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
    return model, acc(model, xva, yva)


def build(variant, seed, G=None, corrupt=None):
    """variant shared|perbyte (random init, consumes gen exactly as E7) or known (fixed G);
    corrupt=p re-initialises the learnable table from G with a fraction p of signs flipped."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    m = Structured(variant, gen)
    if variant == 'known':
        m.fixed = torch.tensor(G, dtype=torch.float32)
    if corrupt is not None:
        rng = np.random.default_rng(seed + 7)
        shape = (256, 8) if variant == 'shared' else (8, 256, 8)
        T = np.broadcast_to(G, shape).copy()
        T[rng.random(shape) < corrupt] *= -1
        with torch.no_grad():
            m.E.copy_(torch.tensor(T, dtype=torch.float32))   # tanh(1) = 0.76
    return m, gen


def data(s, N, sbox, noisiness=0.0):
    puf = ArbiterPUF(n=64, seed=s, noisiness=noisiness)
    clean = ArbiterPUF(n=64, seed=s, noisiness=0.0)
    Ctr = np.random.default_rng(11 + 100 * s).integers(0, 2, (N, 64)).astype(np.uint8)
    Cte = np.random.default_rng(12 + 100 * s).integers(0, 2, (10000, 64)).astype(np.uint8)
    smap = lambda C: bytes_to_bits(sbox[bits_to_bytes(C)])
    ytr = np.concatenate([labels(puf, smap(Ctr[i:i + 100000])) for i in range(0, N, 100000)])
    yte, yte_clean = labels(puf, smap(Cte)), labels(clean, smap(Cte))
    T = lambda a, dt: torch.tensor(a.astype(dt))
    nva = N // 10
    X, Y = T(bits_to_bytes(Ctr), np.int64), T(ytr, np.float32)
    return (X[:N - nva], Y[:N - nva], X[N - nva:], Y[N - nva:],
            T(bits_to_bytes(Cte), np.int64), T(yte, np.float32), T(yte_clean, np.float32))


def done():
    if not os.path.exists(CSV):
        return set()
    return {(r['mode'], r['variant'], int(r['instance']), int(r['N']), int(r['restart']), r['init'])
            for r in csv.DictReader(open(CSV))}


def run(mode, jobs):
    """jobs: list of (variant, instance, N, restart, init, sbox, G, noisiness)."""
    have = done()
    new = not os.path.exists(CSV)
    with open(CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['test', 'mode', 'variant', 'instance', 'N', 'restart', 'init',
                        'acc', 'acc_clean', 'val_acc', 'elapsed_s'])
        cache = {}
        for variant, s, N, r, init, sbox, G, nz in jobs:
            if (mode, variant, s, N, r, init) in have:
                print('skip', mode, variant, s, N, r, init, flush=True)
                continue
            key = (s, N, sbox.tobytes(), nz)
            if key not in cache:
                cache.clear()
                cache[key] = data(s, N, sbox, nz)
            xtr, ytr, xva, yva, xte, yte, yte_c = cache[key]
            t0 = time.time()
            p = None if init in ('random', 'fixed') else float(init.split('=')[1])
            m, gen = build(variant, 1000 * s + r, G=G, corrupt=p)
            m, va = fit(m, gen, xtr, ytr, xva, yva)
            row = ['E7b', mode, variant, s, N, r, init, round(acc(m, xte, yte), 4),
                   round(acc(m, xte, yte_c), 4), round(va, 4), round(time.time() - t0, 1)]
            w.writerow(row)
            f.flush()
            print(*row, flush=True)


def jobs_for(mode):
    J = []
    if mode == 'corrupt':
        for variant, N in (('shared', 20000), ('perbyte', 60000)):
            for s in range(3):
                for p in (0.0, 0.10, 0.25, 0.50):
                    J.append((variant, s, N, 0, f'corrupt={p:.2f}', SBOX_AES, G_AES, 0.0))
    elif mode == 'ext':
        for s in range(10):
            for variant, grid in (('shared', (20000, 30000, 40000, 60000)),
                                  ('perbyte', (60000, 100000, 150000, 200000))):
                for N in grid:
                    for r in range(3):
                        J.append((variant, s, N, r, 'random', SBOX_AES, None, 0.0))
    elif mode in ('random', 'noise'):
        for s in range(5):
            if mode == 'random':
                sbox = np.random.default_rng(5000 + s).permutation(256).astype(np.uint8)
                nz = 0.0
            else:
                sbox, nz = SBOX_AES, 0.1
            G = g_table(sbox)
            for N in (30000, 100000):
                J.append(('known', s, N, 0, 'fixed', sbox, G, nz))
                for r in range(3):
                    J.append(('shared', s, N, r, 'random', sbox, None, nz))
    return J


if __name__ == '__main__':
    for mode in (sys.argv[1:] or ['corrupt', 'ext', 'random', 'noise']):
        run(mode, jobs_for(mode))
