"""E8: FPGA replay. Encoding layers in front of REAL Arbiter PUFs, using public measured CRPs.
Written and preregistered 2026-10-20, before any E8 run (only data sanity checks were run).

Why replay is exact. An encoding layer is deterministic logic; only the APUF is physical. Given
measured pairs (c, r) with c uniform and a bijective byte map S, the pairs (S^-1(c), r) are exact
CRPs of the device "S, then that physical APUF" (the argument of Proposition 1). Consequences:
  - public-layer T1 on replayed data is bit-identical to the plain attack on the measured data, so
    E8 reports the plain (= T1) baseline once and spends its budget on what is NOT an identity:
    T0 learners on the encoded device, and the secret-table attack (E7) on real delays/noise/bias.
  - selector designs (ROC-SPUF) cannot be replayed: their internal challenge depends on the PUF's
    own response to the external challenge, which a fixed random CRP set does not contain.

Data (Zenodo 5221305, AM21 = Aghaie & Moradi, as packaged by pypuf 3.x; fetched directly, pypuf
2.2.0 kept for reproducibility). Six FPGA APUFs measured on the same 1M uniform challenges, one
measurement per challenge:
  top   : 64-stage APUF.                         SHA-256 prefix 774675e7cbad3294
  b0-b4 : 65-stage APUFs; bit 32 (a byte boundary) is the top APUF's response, the other 64 bits
          are the top challenge. Replayed device: S on the 64 challenge bits, bit 32 passed through
          as a public bit (exact for "S on 64 bits + one public bit -> 65-stage APUF").
          SHA-256 prefixes 24b363c0804b3ba3 0f3db4692ca1ff8e e80951d05c835d17 8042aebc709950cf
          8c952e204a7267f9
Checked before writing: challenges uniform (bit means 0.499-0.501, max |corr| 0.008, no
duplicates); bottom bit 32 == top response on 100%; pypuf parity order fits (sklearn parity LR on
top: 0.967 @1k, 0.985 @10k; reversed order stays at majority). Responses are biased: majority
rate 0.695 (top) and 0.53-0.69 (bottom), so chance = majority rate, reported per row as `maj`.
Split: last 100k CRPs = test; training prefix of N, last 10% of it = validation (as E7).
Conventions as gate1: bits = (c+1)/2, labels = (r+1)/2, parity features over 1-2*bit.

Modes (python ...-e8fpga.py base|e7|rand):
  base : known-table control (E7 learner with the true G fixed), N in {250,500,750,1000,1500,
         2000,3000,5000,10000,20000,100000,900000}; sklearn liblinear parity LR on the measured
         challenges (= T1), N in {1000,5000,10000,100000}; T0-raw LR (raw +-1 bits) and
         T0-competent parity LR on the AES-S-box-encoded challenges (no inversion), N in
         {1000,10000,100000,900000}; E5 parity-input MLP on the encoded challenges (512-256-128,
         Adam 1e-3, batch 4096, 40 epochs, seed 0), N in {200000, 900000}.
  e7   : secret AES S-box, structure-aware T2 learner (E7 model, hyperparameters unchanged),
         shared N in {20k,30k,40k,60k,100k,200k}, perbyte N in {60k,100k,150k,200k,400k},
         5 restarts each (seed 1000*d + r), every restart logged.
  rand : per-device random secret S-box (rng 7000+d permutation), shared, N in {40k,100k},
         5 restarts.
Success (primary) = acc >= known-table acc at the same device and N minus 0.05 (the real
ceiling is below the simulated 0.997; in E7b simulation every success and no failure met this);
secondary = acc >= 0.95. Best-of-5 = validation-selected restart.

Predictions (before any E8 run):
  P1 known-table control reaches >=0.95 by N=2000 on 6/6 devices, plateau 0.97-0.99.
  P2 on the S-box-encoded device, T0-raw and T0-competent LR stay within 0.03 of the majority
     rate at 100k on 6/6; the E5 MLP at 900k stays >=0.20 below the known-table plateau on 6/6.
  P3 E7 shared: best-of-5 succeeds by 60k on >=5/6 devices; per-restart success at 200k >=0.8.
  P4 E7 perbyte: best-of-5 succeeds by 200k on >=5/6 devices.
  P5 matched ratio (first grid N with best-of-5 success / known-table N-to-0.95, per device):
     median 15-150x shared (sim 43x), 50-500x perbyte (sim 142x).
  P6 rand: shared best-of-5 succeeds at 100k on >=5/6.
  Falsified (for the paper's claim): shared best-of-5 fails at 200k on >=3/6 devices, i.e. real
  delays, noise or bias break the factorised attack.
Disclosed peek: a smoke test (not logged; predictions above were written before it) ran restart 0
on top/b0: known @1k/@10k 0.957/0.985 and 0.952/0.981; shared @40k 0.973 (top), 0.642 (b0).
Writes experiments/encoding-induced-false-security-pufs-e8fpga.csv (append + resume)
cols: test,mode,device,variant,sbox,N,restart,acc,val_acc,maj,elapsed_s,hw
"""
import csv, hashlib, os, sys, time, numpy as np, torch
from sklearn.linear_model import LogisticRegression

E7 = {'__name__': 'e8_import'}
exec(open('experiments/encoding-induced-false-security-pufs-pilot-e7sboxlearn.py').read(), E7)
bits_to_bytes, SBOX_AES, G_AES = E7['bits_to_bytes'], E7['SBOX'], E7['G_TRUE']
parity_feats = E7['parity_feats']
LR, BS, MIN_STEPS, MIN_EPOCHS = E7['LR'], E7['BS'], E7['MIN_STEPS'], E7['MIN_EPOCHS']
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False
CSV = 'experiments/encoding-induced-false-security-pufs-e8fpga.csv'
DATA = 'data/fpga'
DEVICES = {'top': '774675e7cbad3294', 'b0': '24b363c0804b3ba3', 'b1': '0f3db4692ca1ff8e',
           'b2': 'e80951d05c835d17', 'b3': '8042aebc709950cf', 'b4': '8c952e204a7267f9'}
NTEST, IPOS = 100000, 32


def g_table(sbox):
    s_out = 1 - 2 * np.unpackbits(sbox.reshape(-1, 1), axis=1).astype(np.float64)
    return np.stack([np.prod(s_out[:, k:], axis=1) for k in range(8)], axis=1)


assert np.array_equal(g_table(SBOX_AES), G_AES)


def load(dev):
    p = f'{DATA}/AM21_{dev}.npz'
    assert hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16] == DEVICES[dev], p
    z = np.load(p, allow_pickle=False)
    C, R = z['challenges'], z['responses'].reshape(-1)
    ib = None
    if C.shape[1] == 65:
        ib, C = C[:, IPOS].astype(np.float32), np.delete(C, IPOS, axis=1)
    c01 = ((C.astype(np.int16) + 1) // 2).astype(np.uint8)       # gate1: pm = 2*bit - 1
    y = ((R + 1) // 2).astype(np.float32)
    return c01, ib, y


def internal_feats(c01, ib):
    """Direct parity features of the measured (internal) challenge, incl. the public bit."""
    X = parity_feats(c01)[:, :64]
    if ib is None:
        return X
    s = np.concatenate([1 - 2 * c01[:, :IPOS].astype(np.float64), -ib[:, None].astype(np.float64),
                        1 - 2 * c01[:, IPOS:].astype(np.float64)], 1)
    return np.flip(np.cumprod(np.flip(s, 1), 1), 1)


class Model(torch.nn.Module):
    """E7 Structured model, extended by one public bit at a byte boundary (65-stage chains)."""

    def __init__(self, variant, gen, G, pub):
        super().__init__()
        self.variant, self.pub = variant, pub
        if variant == 'known':
            self.register_buffer('fixed', torch.tensor(G, dtype=torch.float32))
        else:
            shape = (256, 8) if variant == 'shared' else (8, 256, 8)
            self.E = torch.nn.Parameter(torch.randn(shape, generator=gen))
        self.lin = torch.nn.Linear(65 if pub else 64, 1)

    def forward(self, xb, ib):
        T = self.fixed if self.variant == 'known' else torch.tanh(self.E)
        g = T[torch.arange(8, device=xb.device)[None, :], xb] if self.variant == 'perbyte' else T[xb]
        P = g[:, :, 0]
        suf = torch.flip(torch.cumprod(torch.flip(P, [1]), 1), [1])
        suf = torch.cat([suf[:, 1:], torch.ones_like(suf[:, :1])], 1)
        f = g * suf[:, :, None]                                      # (N, 8, 8)
        if not self.pub:
            return self.lin(f.reshape(len(xb), 64)).squeeze(1)
        hi = IPOS // 8
        s = -ib[:, None]                                             # public bit, 1-2*bit form
        x_pub = s[:, 0] * suf[:, hi - 1]                             # prod over bytes >= hi
        feats = torch.cat([(f[:, :hi] * s[:, :, None]).reshape(len(xb), -1), x_pub[:, None],
                           f[:, hi:].reshape(len(xb), -1)], 1)
        return self.lin(feats).squeeze(1)


def accm(m, xb, ib, y):
    with torch.no_grad():
        return float(((m(xb, ib) > 0).float() == y).float().mean())


def fit(variant, seed, G, pub, xtr, itr, ytr, xva, iva, yva):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    m = Model(variant, gen, G, pub).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(xtr)
    for _ in range(max(MIN_EPOCHS, -(-MIN_STEPS // max(1, n // BS)))):
        perm = torch.randperm(n, generator=gen).to(DEV)
        for s in range(0, n, BS):
            idx = perm[s:s + BS]
            opt.zero_grad()
            lossf(m(xtr[idx], None if itr is None else itr[idx]), ytr[idx]).backward()
            opt.step()
    return m, accm(m, xva, iva, yva)


def mlp(Xtr, ytr, Xte, yte, seed=0):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256),
                              torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    X, Y = torch.tensor(Xtr, dtype=torch.float32, device=DEV), torch.tensor(ytr, device=DEV)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(40):
        perm = torch.randperm(len(X), generator=gen).to(DEV)
        for s in range(0, len(X), 4096):
            idx = perm[s:s + 4096]
            opt.zero_grad()
            lossf(net(X[idx]).squeeze(1), Y[idx]).backward()
            opt.step()
    with torch.no_grad():
        p = net(torch.tensor(Xte, dtype=torch.float32, device=DEV)).squeeze(1) > 0
    return float((p.float().cpu().numpy() == yte).mean())


def sk_lr(Xtr, ytr, Xte, yte):
    return float(LogisticRegression(solver='liblinear', C=1.0, max_iter=1000).fit(Xtr, ytr).score(Xte, yte))


def encode(c01, sbox):
    """External challenge bytes: S^-1 applied bytewise, so that S(external) = measured."""
    inv = np.argsort(sbox).astype(np.uint8)
    return inv[bits_to_bytes(c01)]


def ext_feats(xb, ib, raw):
    """Features of the EXTERNAL (encoded) challenge, without inversion (T0)."""
    bits = np.unpackbits(xb, axis=1)
    if raw:
        X = 1 - 2 * bits.astype(np.float32)
        return X if ib is None else np.concatenate([X, ib[:, None]], 1)
    return internal_feats(bits, ib).astype(np.float32)


def done():
    if not os.path.exists(CSV):
        return set()
    return {(r['mode'], r['device'], r['variant'], r['sbox'], int(r['N']), int(r['restart']))
            for r in csv.DictReader(open(CSV))}


class Writer:
    def __init__(self):
        self.have = done()
        new = not os.path.exists(CSV)
        self.f = open(CSV, 'a', newline='')
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(['test', 'mode', 'device', 'variant', 'sbox', 'N', 'restart', 'acc',
                             'val_acc', 'maj', 'elapsed_s', 'hw'])

    def skip(self, *k):
        return k in self.have

    def row(self, mode, dev, variant, sbox, N, r, acc, va, maj, t0):
        hw = torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'
        row = ['E8', mode, dev, variant, sbox, N, r, round(acc, 4),
               '' if va is None else round(va, 4), round(maj, 4), round(time.time() - t0, 1), hw]
        self.w.writerow(row)
        self.f.flush()
        print(*row, flush=True)


def check_factor(c01, ib):
    """Startup assert: the known-table model's features equal the direct internal features."""
    n = 2000
    xb = torch.tensor(encode(c01[:n], SBOX_AES).astype(np.int64))
    m = Model('known', torch.Generator().manual_seed(0), G_AES, ib is not None)
    captured = {}
    m.lin.register_forward_hook(lambda mod, inp, out: captured.setdefault('x', inp[0]))
    m(xb, None if ib is None else torch.tensor(ib[:n]))
    assert np.allclose(captured['x'].numpy(), internal_feats(c01[:n], None if ib is None else ib[:n])), \
        'factorised features do not match direct parity features'


def run(mode):
    W = Writer()
    for dev in DEVICES:
        c01, ib, y = load(dev)
        check_factor(c01, ib)
        maj = float(max(y[-NTEST:].mean(), 1 - y[-NTEST:].mean()))
        pub = ib is not None
        T = lambda a, dt: torch.tensor(a, dtype=dt, device=DEV)

        def split(xb, N):
            nva = N // 10
            I = None if ib is None else T(ib, torch.float32)
            sl = lambda a, s: None if a is None else a[s]
            X, Y = T(xb.astype(np.int64), torch.int64), T(y, torch.float32)
            return (X[:N - nva], sl(I, slice(0, N - nva)), Y[:N - nva],
                    X[N - nva:N], sl(I, slice(N - nva, N)), Y[N - nva:N],
                    X[-NTEST:], sl(I, slice(len(y) - NTEST, None)), Y[-NTEST:])

        def t2(variant, sbox_name, sbox, Ns, restarts):
            xb = encode(c01, sbox)
            for N in Ns:
                for r in range(restarts):
                    if W.skip(mode, dev, variant, sbox_name, N, r):
                        continue
                    xtr, itr, ytr, xva, iva, yva, xte, ite, yte = split(xb, N)
                    t0 = time.time()
                    m, va = fit(variant, 1000 * list(DEVICES).index(dev) + r, g_table(sbox), pub,
                                xtr, itr, ytr, xva, iva, yva)
                    W.row(mode, dev, variant, sbox_name, N, r, accm(m, xte, ite, yte), va, maj, t0)

        if mode == 'base':
            t2('known', 'aes', SBOX_AES, (250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000,
                                          20000, 100000, 900000), 1)
            Xin = internal_feats(c01, ib).astype(np.float32)
            xb = encode(c01, SBOX_AES)
            Xraw, Xpar = ext_feats(xb, ib, True), ext_feats(xb, ib, False)
            yte = y[-NTEST:]
            for variant, X, Ns in (('T1-parityLR', Xin, (1000, 5000, 10000, 100000)),
                                   ('T0-rawLR', Xraw, (1000, 10000, 100000, 900000)),
                                   ('T0-parityLR', Xpar, (1000, 10000, 100000, 900000))):
                for N in Ns:
                    if W.skip(mode, dev, variant, 'aes', N, 0):
                        continue
                    t0 = time.time()
                    W.row(mode, dev, variant, 'aes', N, 0, sk_lr(X[:N], y[:N], X[-NTEST:], yte), None, maj, t0)
            for N in (200000, 900000):
                if W.skip(mode, dev, 'T0-parityMLP', 'aes', N, 0):
                    continue
                t0 = time.time()
                Xp = np.concatenate([Xpar, np.ones((len(Xpar), 1), np.float32)], 1)
                W.row(mode, dev, 'T0-parityMLP', 'aes', N, 0, mlp(Xp[:N], y[:N], Xp[-NTEST:], yte), None, maj, t0)
        elif mode == 'e7':
            t2('shared', 'aes', SBOX_AES, (20000, 30000, 40000, 60000, 100000, 200000), 5)
            t2('perbyte', 'aes', SBOX_AES, (60000, 100000, 150000, 200000, 400000), 5)
        elif mode == 'rand':
            sb = np.random.default_rng(7000 + list(DEVICES).index(dev)).permutation(256).astype(np.uint8)
            t2('known', 'rand', sb, (40000, 100000), 1)
            t2('shared', 'rand', sb, (40000, 100000), 5)
        torch.cuda.empty_cache() if DEV.type == 'cuda' else None
    W.f.close()


if __name__ == '__main__':
    for mode in sys.argv[1:] or ['base', 'e7', 'rand']:
        run(mode)
