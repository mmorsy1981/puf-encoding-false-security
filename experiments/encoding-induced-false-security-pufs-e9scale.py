"""E9: scaling and robustness of the secret-S-box attack (E7) in simulation.
Written and preregistered 2026-10-20, before any E9 run.

Learner: E7's structure-aware model generalised to nb = n/8 bytes (tanh tables, Adam LR 0.02,
batch 4096, >=6000 steps / >=20 epochs, BCE, 10% validation), hyperparameters unchanged.
Data: external challenge bytes drawn uniform (rng 11+100s train, 12+100s test, 10k test),
internal challenge = AES S-box applied bytewise; labels (r+1)/2 as gate1. Instances s = 0-9,
restarts r = 0-2 (seed 1000*s + r), every restart logged; known = true table fixed (1 run).
Success = acc >= known-table acc at the same instance/N minus 0.05 (noise mode: on clean labels).
GPU (CUDA, TF32 off; init and batch order drawn on the CPU from the seeds, as E7gpu), one process
per mode, one CSV per mode; not bitwise reproducible (see E7b/E7gpu).

Modes (python ...-e9scale.py nscale|noise|stress):
  nscale : pypuf ArbiterPUF, n in {32, 64, 128} (4/8/16 bytes), noiseless.
           known  N {100,250,500,1000,2000,4000,8000,16000}
           shared N n=32 {5k,10k,20k,30k,60k}; n=64 {10k,20k,30k,40k,60k}; n=128 {10k,20k,40k,60k,100k,200k}
           perbyte N n=32 {10k,20k,40k,60k,100k}; n=64 {40k,60k,100k,150k,200k};
                     n=128 {100k,200k,300k,400k,800k}
           Extension rule (fixed now): if per-restart success at a variant's largest N is < 0.8,
           append 2x that N and repeat, up to 3.2M.
  noise  : n=64 pypuf ArbiterPUF, noisiness in {0, 0.05, 0.1, 0.2, 0.3, 0.5}; flip rate vs the
           noiseless twin logged; known N {1k, 5k, 20k, 100k}; shared N {30k,60k,100k,200k,400k};
           test accuracy on noisy labels (acc) and on the noiseless twin (acc_clean).
  stress : n=64 linear-threshold devices on parity features (own simulator, noiseless):
             gauss  : w ~ N(0,1) (control, same law as pypuf)
             ar09   : AR(1)-correlated weights along the chain, rho = 0.9, unit variance
             t3     : heavy-tailed weights, Student-t df=3, scaled to unit variance
             bias70 : gauss weights plus a constant offset giving P(r=1) = 0.70 (cf. AM21 top 0.695)
           known N {500,1000,2000,5000}; shared N {20k,30k,40k,60k,100k}.

Predictions (before any E9 run):
  N1 known-table N-to-0.95 grows roughly linearly in n: ratio n=128 / n=32 in [2.5, 6].
  N2 shared N50 (N at 50% per-restart success, log-interpolated) at n=128 is within 0.5-2x of
     n=64: the secret shared table has an n-independent description length (log2 256! bits).
  N3 perbyte N50 grows with the byte count: N50(128)/N50(64) and N50(64)/N50(32) in [1.3, 3.5].
  N4 hence the shared cost ratio (N50 / known N-to-0.95) does not grow with n.
  Q1 noise: best-of-3 shared succeeds at 200k on >=8/10 for noisiness <= 0.2; N50 rises
     monotonically with noisiness; noisiness 0.3/0.5 exploratory (no prediction).
  S1 stress: known reaches >=0.95 by 2000 on >=9/10 for every device model.
  S2 stress: shared per-restart success at 60k is within +-0.25 of the gauss control for each
     model.
Falsified (claim "a secret S-box costs ~10^2, independent of delay model") if S2 fails for any
model, or N2 fails upward (shared N50 at n=128 > 2x n=64).
Post-run note (2026-10-22, disclosed): an interrupted chain whose processes were reported stopped
kept running, so each mode ran twice concurrently from the same resume point into the same CSV.
Each key (model,n,nz,variant,instance,N,restart) was computed with the same seed and data; all
2051 duplicate pairs agree exactly (max |d acc| 0.0000). The CSVs keep the first row per key; the
second copies are in ...-e9scale-<mode>-dualrun.csv. No result depends on which copy is kept.
Writes experiments/encoding-induced-false-security-pufs-e9scale-<mode>.csv (append + resume)
cols: test,mode,model,n,nz,variant,instance,N,restart,acc,acc_clean,val_acc,flip,elapsed_s
"""
import csv, os, sys, time, numpy as np, torch
from pypuf.simulation import ArbiterPUF

E7 = {'__name__': 'e9_import'}
exec(open('experiments/encoding-induced-false-security-pufs-pilot-e7sboxlearn.py').read(), E7)
torch.set_num_threads(4)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False
SBOX, G_AES, factor_feats = E7['SBOX'], E7['G_TRUE'], E7['factor_feats']
LR, BS, MIN_STEPS, MIN_EPOCHS = E7['LR'], E7['BS'], E7['MIN_STEPS'], E7['MIN_EPOCHS']
CSV = 'experiments/encoding-induced-false-security-pufs-e9scale-{}.csv'
NTEST, SUCC_MARGIN, NMAX = 10000, 0.05, 3200000


class Model(torch.nn.Module):
    """E7 Structured model for nb bytes."""

    def __init__(self, variant, gen, nb):
        super().__init__()
        self.variant, self.nb = variant, nb
        if variant == 'known':
            self.register_buffer('fixed', torch.tensor(G_AES, dtype=torch.float32))
        else:
            shape = (256, 8) if variant == 'shared' else (nb, 256, 8)
            self.E = torch.nn.Parameter(torch.randn(shape, generator=gen))
        self.lin = torch.nn.Linear(8 * nb, 1)

    def forward(self, xb):
        T = self.fixed if self.variant == 'known' else torch.tanh(self.E)
        g = T[torch.arange(self.nb, device=xb.device)[None, :], xb] if self.variant == 'perbyte' else T[xb]
        P = g[:, :, 0]
        suf = torch.flip(torch.cumprod(torch.flip(P, [1]), 1), [1])
        suf = torch.cat([suf[:, 1:], torch.ones_like(suf[:, :1])], 1)
        return self.lin((g * suf[:, :, None]).reshape(len(xb), -1)).squeeze(1)


def int_feats(xb):
    """Direct parity features (1-2*bit form) of the internal challenge S(x), any byte count."""
    bits = np.unpackbits(SBOX[xb], axis=1).astype(np.float64)
    return np.flip(np.cumprod(np.flip(1 - 2 * bits, 1), 1), 1)


_x = np.random.default_rng(999).integers(0, 256, (500, 8)).astype(np.uint8)
assert np.allclose(int_feats(_x), factor_feats(_x, G_AES))       # matches E7 at n=64
_x16 = np.random.default_rng(998).integers(0, 256, (500, 16)).astype(np.int64)
_m = Model('known', None, 16)
_cap = {}
_m.lin.register_forward_hook(lambda mod, i, o: _cap.setdefault('x', i[0]))
_m(torch.tensor(_x16))
assert np.allclose(_cap['x'].numpy(), int_feats(_x16.astype(np.uint8)))   # generalised model, n=128


def acc(m, x, y):
    with torch.no_grad():
        return float(((m(x) > 0).float() == y).float().mean())


def fit(variant, seed, nb, xtr, ytr, xva, yva):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    m = Model(variant, gen, nb).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(xtr)
    for _ in range(max(MIN_EPOCHS, -(-MIN_STEPS // max(1, n // BS)))):
        perm = torch.randperm(n, generator=gen).to(DEV)
        for s in range(0, n, BS):
            idx = perm[s:s + BS]
            opt.zero_grad()
            lossf(m(xtr[idx]), ytr[idx]).backward()
            opt.step()
    return m, acc(m, xva, yva)


class LTF:
    """Noiseless linear threshold over parity features (1-2*bit form) for the stress models."""

    def __init__(self, model, n, s):
        rng = np.random.default_rng(80000 + s)
        if model == 'gauss':
            w = rng.standard_normal(n)
        elif model == 'ar09':
            e, w = rng.standard_normal(n), np.empty(n)
            w[0] = e[0]
            for i in range(1, n):
                w[i] = 0.9 * w[i - 1] + np.sqrt(1 - 0.81) * e[i]
        elif model == 't3':
            w = rng.standard_t(3, n) / np.sqrt(3.0)
        elif model == 'bias70':
            w = rng.standard_normal(n)
        self.w, self.b = w, 0.0
        if model == 'bias70':
            z = int_feats(np.random.default_rng(90000 + s).integers(0, 256, (200000, n // 8)).astype(np.uint8)) @ w
            self.b = -float(np.quantile(z, 0.30))                    # P(z + b > 0) = 0.70

    def eval(self, xb):
        return (int_feats(xb) @ self.w + self.b > 0).astype(np.float32)


def labels_pypuf(puf, xb):
    bits = np.unpackbits(SBOX[xb], axis=1)
    return ((puf.eval(2 * bits.astype(np.int8) - 1).reshape(-1) + 1) // 2).astype(np.float32)


def make_data(mode, model, n, nz, s, N):
    nb = n // 8
    xtr = np.random.default_rng(11 + 100 * s).integers(0, 256, (N, nb)).astype(np.uint8)
    xte = np.random.default_rng(12 + 100 * s).integers(0, 256, (NTEST, nb)).astype(np.uint8)
    if mode == 'stress':
        d = LTF(model, n, s)
        ytr, yte = d.eval(xtr), d.eval(xte)
        yte_c, flip = yte, 0.0
    else:
        puf = ArbiterPUF(n=n, seed=s, noisiness=nz)
        clean = ArbiterPUF(n=n, seed=s, noisiness=0.0)
        ytr = np.concatenate([labels_pypuf(puf, xtr[i:i + 100000]) for i in range(0, N, 100000)])
        yte, yte_c = labels_pypuf(puf, xte), labels_pypuf(clean, xte)
        flip = float((yte != yte_c).mean())
    nva = N // 10
    T = lambda a: torch.tensor(a).to(DEV)
    X, Y = T(xtr.astype(np.int64)), T(ytr)
    return (X[:N - nva], Y[:N - nva], X[N - nva:], Y[N - nva:], T(xte.astype(np.int64)),
            T(yte), T(yte_c), flip)


def done(path):
    if not os.path.exists(path):
        return {}
    return {(r['mode'], r['model'], int(r['n']), float(r['nz']), r['variant'], int(r['instance']),
             int(r['N']), int(r['restart'])): r for r in csv.DictReader(open(path))}


class Runner:
    def __init__(self, path):
        self.have = done(path)
        new = not os.path.exists(path)
        self.f = open(path, 'a', newline='')
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(['test', 'mode', 'model', 'n', 'nz', 'variant', 'instance', 'N', 'restart',
                             'acc', 'acc_clean', 'val_acc', 'flip', 'elapsed_s'])
        self.cache = {}

    def cell(self, mode, model, n, nz, variant, N, restarts, instances=range(10)):
        """Run one (variant, N) cell over instances x restarts; return list of result dicts."""
        out = []
        for s in instances:
            for r in range(restarts):
                key = (mode, model, n, float(nz), variant, s, N, r)
                if key in self.have:
                    out.append(self.have[key])
                    continue
                dk = (mode, model, n, nz, s, N)
                if dk not in self.cache:
                    self.cache = {dk: make_data(mode, model, n, nz, s, N)}
                xtr, ytr, xva, yva, xte, yte, yte_c, flip = self.cache[dk]
                t0 = time.time()
                m, va = fit(variant, 1000 * s + r, n // 8, xtr, ytr, xva, yva)
                row = ['E9', mode, model, n, nz, variant, s, N, r, round(acc(m, xte, yte), 4),
                       round(acc(m, xte, yte_c), 4), round(va, 4), round(flip, 4), round(time.time() - t0, 1)]
                self.w.writerow(row)
                self.f.flush()
                print(*row, flush=True)
                d = dict(zip(['test', 'mode', 'model', 'n', 'nz', 'variant', 'instance', 'N', 'restart',
                              'acc', 'acc_clean', 'val_acc', 'flip', 'elapsed_s'], map(str, row)))
                self.have[key] = d
                out.append(d)
        return out

    def known_acc(self, mode, model, n, nz, s, N, clean):
        r = self.have.get((mode, model, n, float(nz), 'known', s, N, 0))
        if r is None:
            self.cell(mode, model, n, nz, 'known', N, 1, [s])
            r = self.have[(mode, model, n, float(nz), 'known', s, N, 0)]
        return float(r['acc_clean' if clean else 'acc'])

    def sweep(self, mode, model, n, nz, variant, grid, extend):
        """Run the grid; with extend, double N past the grid until per-restart success >= 0.8."""
        grid = list(grid)
        i = 0
        while i < len(grid):
            N = grid[i]
            rows = self.cell(mode, model, n, nz, variant, N, 3)
            if variant != 'known' and i == len(grid) - 1 and extend and N * 2 <= NMAX:
                clean = mode == 'noise'
                ok = [float(r['acc_clean' if clean else 'acc']) >=
                      self.known_acc(mode, model, n, nz, int(r['instance']), N, clean) - SUCC_MARGIN
                      for r in rows]
                if np.mean(ok) < 0.8:
                    grid.append(N * 2)
                    print(f'# extension: {mode} {model} n={n} nz={nz} {variant}: success {np.mean(ok):.2f} at {N} -> {N * 2}', flush=True)
            i += 1


def main(mode):
    R = Runner(CSV.format(mode))
    if mode == 'nscale':
        known = (100, 250, 500, 1000, 2000, 4000, 8000, 16000)
        grids = {32: ((5000, 10000, 20000, 30000, 60000), (10000, 20000, 40000, 60000, 100000)),
                 64: ((10000, 20000, 30000, 40000, 60000), (40000, 60000, 100000, 150000, 200000)),
                 128: ((10000, 20000, 40000, 60000, 100000, 200000), (100000, 200000, 300000, 400000, 800000))}
        for n in (32, 64, 128):
            for N in known:
                R.cell(mode, 'pypuf', n, 0.0, 'known', N, 1)
            R.sweep(mode, 'pypuf', n, 0.0, 'shared', grids[n][0], True)
        for n in (32, 64, 128):
            R.sweep(mode, 'pypuf', n, 0.0, 'perbyte', grids[n][1], True)
    elif mode == 'noise':
        for nz in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
            for N in (1000, 5000, 20000, 100000):
                R.cell(mode, 'pypuf', 64, nz, 'known', N, 1)
            R.sweep(mode, 'pypuf', 64, nz, 'shared', (30000, 60000, 100000, 200000, 400000), False)
    elif mode == 'stress':
        for model in ('gauss', 'ar09', 't3', 'bias70'):
            for N in (500, 1000, 2000, 5000):
                R.cell(mode, model, 64, 0.0, 'known', N, 1)
            R.sweep(mode, model, 64, 0.0, 'shared', (20000, 30000, 40000, 60000, 100000), False)
    R.f.close()


if __name__ == '__main__':
    for mode in sys.argv[1:] or ['stress', 'noise', 'nscale']:
        main(mode)
