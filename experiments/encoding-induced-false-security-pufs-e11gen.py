"""E11: generalisation of the secret-S-box result (E7) beyond one APUF, one layer, one learner.
Written and preregistered 2026-10-20, before any E11 run.

Common rig: 64-bit challenges as 8 uniform external bytes x (rng 11+100s train, 12+100s test,
10k test), AES S-box (gate1), labels (r+1)/2 via gate1 labels(); learner = E7's tanh-table model
and loop (Adam LR 0.02, batch 4096, >=6000 steps / >=20 epochs, BCE, 10% validation) unless the
mode varies it; seed 1000*s + r; GPU with CPU-side init/batch order (as E7gpu); every restart logged.

Modes (python ...-e11gen.py <mode>; one CSV per mode):
  mix   : secret S-box, then a PUBLIC GF(2)-linear layer L on the 64 bits, then the APUF.
          Parity feature k of the internal challenge is prod_i s_i^{A_ki}, A_k = XOR_{j>=k} L_j, with
          s_i = 1-2*(bit i of S(x)): still a product of per-byte S-box output signs, so the E7
          table model applies with mask A instead of the suffix mask. Layers: 'id' (L = I, the E7
          control) and 'lin' (random invertible L, rng 6000, public). Variants known/shared;
          N known {500,1k,2k,5k}; shared {20k,30k,40k,60k,100k,200k}; instances 0-9, 3 restarts.
  spn2  : two rounds, x -> S -> public bit permutation P (rng 6001) -> S -> APUF (non-linear
          mixing across bytes; no byte factorisation). Variants: known (true S), 'mlp' (E5 parity
          MLP on external bits), 'shared1' (E7 one-layer model, misspecified), 'shared2' (two-layer
          structured model: learned table, soft second lookup by multilinear extension).
          N {100k, 400k, 1M}; instances 0-4; 3 restarts (mlp 1).
  xor   : base PUF = pypuf XORArbiterPUF(n=64, k, seed=s), k in {2, 3}; secret AES S-box.
          Model: E7 table features -> k linear heads -> prod tanh -> BCE on (1+prod)/2.
          k=2: known N {2k,5k,10k,20k,50k}, shared {50k,100k,200k,400k}
          k=3: known N {10k,20k,50k,100k}, shared {100k,200k,400k,800k}; instances 0-4, 5 restarts.
  hparam: n=64 shared, instances 0-9 x 3 restarts at N=30k: lr in {0.005, 0.01, 0.05};
          lr 0.02 with 24000 steps. Plus N=20k, lr 0.02, restarts 0-9 (best-of-k check).
  impl  : informed T1 on the S-box device, N {500,1000,5000}, instances 0-9: sklearn liblinear
          (the paper's), sklearn newton-cg (C=1; sklearn's lbfgs and newton-cholesky crash natively in
          this env with scipy 1.18.0, Windows 0xc06d007f), torch full-batch L-BFGS logistic regression, and the
          E7 known-table model (Adam). CPU except the last.
  n30   : E7gpu rig (same data and seeds as E7b), shared @30k and perbyte @100k, instances 10-29,
          3 restarts; rows go to the E7gpu CSV with mode 'n30'.
Success = acc >= 0.95 (xor: best-of-5 >= 0.95).

Predictions (before any E11 run):
  M1 mix/lin: known >= 0.95 by 2k on >=9/10; shared best-of-3 succeeds by 200k on >=8/10.
     (A public linear layer after a byte-local secret S-box does not stop the attack.)
  M2 mix/id reproduces the E7b shared success pattern (per-restart rate at 30k/60k within 0.2).
  S0 spn2: exploratory. Expected: mlp and shared1 stay < 0.70 at 1M; shared2 unknown.
  X1 xor known: best-of-5 >= 0.95 by 20k (k=2) and by 100k (k=3) on >=4/5.
  X2 xor shared, k=2: best-of-5 >= 0.95 by 400k on >=3/5; k=3 exploratory.
  H1 hparam: per-restart success at 30k for lr 0.01 and 0.05 within +-0.25 of lr 0.02 (E7b: 0.43).
  H2 best-of-10 at 20k succeeds on >=6/10 instances (independent restarts at p=0.2 would give
     0.89; instance heterogeneity lowers it).
  I1 impl: every implementation's mean T1 accuracy within 0.01 of liblinear at N >= 1000.
  N1 n30: per-restart success over the 30 instances stays inside the 10-instance 95% CIs
     (shared 30k: [0.25, 0.63]; perbyte 100k: [0.65, 0.94]).

Post-run correction (2026-10-20, disclosed): the first mix run used a MaskModel that learned the
8 output-bit signs per byte value and multiplied up to 64 tanh terms per feature. That is not E7's
learner (E7 learns the within-byte parity table G directly, <= 8 tanh terms per feature), and it
stalled: mix/id/shared 0/62 runs >= 0.95, best 0.762, at 20k-200k (M2 control failed, which
exposed the bug; known-table rows unaffected and kept). Those 62 rows are archived in
...-e11gen-mix-bitsign.csv. MaskModel now learns one character table over the distinct byte masks
of A; for the suffix mask it equals E7's Structured('shared') seed for seed (asserted in _checks).
Smoke check after the fix (id, instance 0, 60k, restarts 0-2): 0.985/0.986/0.985. For mix/lin
the table has 219 mask columns (id: 8), a weaker prior than E7's; M1 is kept as written.
hparam, xor and spn2/shared1 had not run with the old model; impl's known-adam rows are unaffected.
Dual run (2026-10-22): spn2 ran twice concurrently (a chain reported stopped kept running); all 109
duplicate pairs agree exactly; the CSV keeps the first row per key, second copies in
...-e11gen-spn2-dualrun.csv. No other E11 CSV has duplicate keys.
cols: test,mode,cfg,variant,instance,N,restart,acc,val_acc,elapsed_s
"""
import csv, os, sys, time, numpy as np, torch
from pypuf.simulation import ArbiterPUF, XORArbiterPUF
from sklearn.linear_model import LogisticRegression

E7 = {'__name__': 'e11_import'}
exec(open('experiments/encoding-induced-false-security-pufs-pilot-e7sboxlearn.py').read(), E7)
SBOX, G_AES, labels, parity_feats = E7['SBOX'], E7['G_TRUE'], E7['labels'], E7['parity_feats']
LR, BS, MIN_STEPS, MIN_EPOCHS = E7['LR'], E7['BS'], E7['MIN_STEPS'], E7['MIN_EPOCHS']
torch.set_num_threads(4)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False
CSV = 'experiments/encoding-induced-false-security-pufs-e11gen-{}.csv'
NTEST = 10000
SIGN = torch.tensor(1 - 2 * np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.float32))
S_SIGN = torch.tensor(1 - 2 * np.unpackbits(SBOX[:, None], axis=1).astype(np.float32))  # (256, 8), +-1


def sbits(xb):
    return np.unpackbits(SBOX[xb], axis=1)


def suffix_mask():
    return np.triu(np.ones((64, 64), dtype=bool))          # A[k, i] = (i >= k)


def gf2_invertible(rng):
    while True:
        L = rng.integers(0, 2, (64, 64)).astype(np.uint8)
        M, r = L.copy(), 0
        for c in range(64):
            piv = next((i for i in range(r, 64) if M[i, c]), None)
            if piv is None:
                continue
            M[[r, piv]] = M[[piv, r]]
            for i in range(64):
                if i != r and M[i, c]:
                    M[i] ^= M[r]
            r += 1
        if r == 64:
            return L


L_LIN = gf2_invertible(np.random.default_rng(6000))
PERM = np.random.default_rng(6001).permutation(64)


def mask_for(L):
    return (np.cumsum(L[::-1], axis=0)[::-1] % 2).astype(bool)   # A_k = XOR_{j>=k} L_j


# ---------------------------------------------------------------- models
class TableFeats(torch.nn.Module):
    """Soft S-box output signs per byte: known (true) or learned tanh table, shared over bytes."""

    def __init__(self, variant, gen):
        super().__init__()
        self.variant = variant
        if variant.startswith('known'):
            self.register_buffer('fixed', S_SIGN.clone())
        else:
            self.E = torch.nn.Parameter(torch.randn((256, 8), generator=gen))

    def table(self):
        return self.fixed if self.variant.startswith('known') else torch.tanh(self.E)


class MaskModel(torch.nn.Module):
    """feature_k = prod_i s_i^{A_ki} = prod_b chi_{m_kb}(S(x_b)), m_kb = row k of A on byte b.

    E7's parameterisation: one learned character table H[x, m] = tanh(E[x, m]) over the distinct
    non-zero byte masks m (descending, so for the suffix mask H is exactly E7's G table and the
    model equals E7's Structured('shared') seed for seed), so a feature is a product of <= 8 tanh
    terms, one per byte. heads: 1 (APUF) or k (XOR)."""

    def __init__(self, variant, gen, A, heads=1):
        super().__init__()
        m = np.packbits(np.asarray(A, dtype=np.uint8).reshape(64, 8, 8), axis=2)[:, :, 0].astype(int)
        M = sorted({int(v) for v in m.ravel() if v}, reverse=True)
        pos = {v: j for j, v in enumerate(M)}
        self.register_buffer('I', torch.tensor([[pos.get(int(v), len(M)) for v in row] for row in m]))
        self.variant = variant
        if variant.startswith('known'):
            bits = torch.tensor(np.unpackbits(np.array(M, dtype=np.uint8)[:, None], axis=1).astype(bool))
            self.register_buffer('fixed', torch.where(bits[None], S_SIGN[:, None, :], torch.ones(())).prod(2))
        else:
            self.E = torch.nn.Parameter(torch.randn((256, len(M)), generator=gen))
        self.heads = heads
        self.lin = torch.nn.Linear(64, heads)

    def table(self):
        return self.fixed if self.variant.startswith('known') else torch.tanh(self.E)

    def feats(self, xb):
        H = self.table()
        H = torch.cat([H, torch.ones_like(H[:, :1])], 1)            # column len(M): mask 0 -> 1
        return H[xb[:, None, :], self.I[None]].prod(2)

    def forward(self, xb):
        z = self.lin(self.feats(xb))
        if self.heads == 1:
            return z.squeeze(1)
        p = torch.tanh(z).prod(1)                            # XOR of k sign outputs, soft
        return torch.logit(((1 + p) / 2).clamp(1e-6, 1 - 1e-6))


class SPN2Model(torch.nn.Module):
    """Two rounds: soft lookup -> public bit permutation -> soft second lookup (multilinear)."""

    def __init__(self, variant, gen):
        super().__init__()
        self.tf = TableFeats(variant, gen)
        self.register_buffer('perm', torch.tensor(PERM))
        self.register_buffer('sign', SIGN.clone())
        self.lin = torch.nn.Linear(64, 1)

    def forward(self, xb):
        T = self.tf.table()
        v = T[xb].reshape(len(xb), 64)[:, self.perm].reshape(len(xb), 8, 8)
        w = torch.ones((len(xb), 8, 256), device=v.device)
        for j in range(8):
            w = w * (1 + v[:, :, j:j + 1] * self.sign[None, None, :, j]) / 2
        out = (w @ T).reshape(len(xb), 64)                    # soft output signs of round 2
        suf = torch.flip(torch.cumprod(torch.flip(out, [1]), 1), [1])
        return self.lin(suf).squeeze(1)


class MLP(torch.nn.Module):
    def __init__(self, gen):
        super().__init__()
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
        self.net = torch.nn.Sequential(torch.nn.Linear(65, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256),
                                       torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(),
                                       torch.nn.Linear(128, 1))

    def forward(self, xb):
        bits = ((xb[:, :, None] >> torch.arange(7, -1, -1, device=xb.device)) & 1).reshape(len(xb), 64)
        s = 1 - 2 * bits.float()
        f = torch.flip(torch.cumprod(torch.flip(s, [1]), 1), [1])
        return self.net(torch.cat([f, torch.ones_like(f[:, :1])], 1)).squeeze(1)


# ---------------------------------------------------------------- training
def accm(m, x, y):
    with torch.no_grad():
        out = []
        for i in range(0, len(x), 20000):
            out.append((m(x[i:i + 20000]) > 0).float())
        return float((torch.cat(out) == y).float().mean())


def fit(m, gen, xtr, ytr, xva, yva, lr=LR, min_steps=MIN_STEPS, epochs=None, bs=BS):
    m = m.to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(xtr)
    E = epochs or max(MIN_EPOCHS, -(-min_steps // max(1, n // bs)))
    for _ in range(E):
        perm = torch.randperm(n, generator=gen).to(DEV)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            lossf(m(xtr[idx]), ytr[idx]).backward()
            opt.step()
    return m, accm(m, xva, yva)


def build(kind, variant, seed, **kw):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    if kind == 'mask':
        return MaskModel(variant, gen, kw['A'], kw.get('heads', 1)), gen
    if kind == 'spn2':
        return SPN2Model(variant, gen), gen
    if kind == 'mlp':
        return MLP(gen), gen


# ---------------------------------------------------------------- data
def internal_bits(cfg, xb):
    if cfg in ('id', 'xor2', 'xor3', 'hp', 'impl'):
        return sbits(xb)
    if cfg == 'lin':
        return (sbits(xb).astype(np.int64) @ L_LIN.T.astype(np.int64) % 2).astype(np.uint8)
    if cfg == 'spn2':
        u = sbits(xb)[:, PERM]
        return sbits(np.packbits(u, axis=1))
    raise ValueError(cfg)


def device_for(cfg, s):
    if cfg in ('xor2', 'xor3'):
        return XORArbiterPUF(n=64, k=int(cfg[-1]), seed=s)
    return ArbiterPUF(n=64, seed=s, noisiness=0.0)


def make_data(cfg, s, N):
    puf = device_for(cfg, s)
    xtr = np.random.default_rng(11 + 100 * s).integers(0, 256, (N, 8)).astype(np.uint8)
    xte = np.random.default_rng(12 + 100 * s).integers(0, 256, (NTEST, 8)).astype(np.uint8)
    lab = lambda x: np.concatenate([labels(puf, internal_bits(cfg, x[i:i + 100000]))
                                    for i in range(0, len(x), 100000)]).astype(np.float32)
    ytr, yte = lab(xtr), lab(xte)
    nva = N // 10
    T = lambda a: torch.tensor(a).to(DEV)
    X, Y = T(xtr.astype(np.int64)), T(ytr)
    return X[:N - nva], Y[:N - nva], X[N - nva:], Y[N - nva:], T(xte.astype(np.int64)), T(yte), (xtr, ytr, xte, yte)


# ---------------------------------------------------------------- self-checks (run at import)
def _checks():
    xb = np.random.default_rng(999).integers(0, 256, (300, 8)).astype(np.uint8)
    direct = lambda bits: parity_feats(bits)[:, :64]
    for cfg, A in (('id', suffix_mask()), ('lin', mask_for(L_LIN))):
        m = MaskModel('known', None, A)
        assert np.allclose(m.feats(torch.tensor(xb.astype(np.int64))).numpy(), direct(internal_bits(cfg, xb))), cfg
    xt = torch.tensor(xb.astype(np.int64))                    # shared/id must be E7's learner, seed for seed
    a, _ = build('mask', 'shared', 4242, A=suffix_mask())
    torch.manual_seed(4242)
    b = E7['Structured']('shared', torch.Generator().manual_seed(4242))
    with torch.no_grad():
        assert torch.allclose(a(xt), b(xt), atol=1e-6), 'mask/shared != E7 Structured'
    m = SPN2Model('known', None)
    cap = {}
    m.lin.register_forward_hook(lambda mod, i, o: cap.setdefault('x', i[0]))
    m(torch.tensor(xb.astype(np.int64)))
    assert np.allclose(cap['x'].numpy(), direct(internal_bits('spn2', xb)), atol=1e-5), 'spn2'


_checks()


# ---------------------------------------------------------------- runner
class Runner:
    COLS = ['test', 'mode', 'cfg', 'variant', 'instance', 'N', 'restart', 'acc', 'val_acc', 'elapsed_s']

    def __init__(self, mode):
        self.mode, self.path = mode, CSV.format(mode)
        self.have = set()
        if os.path.exists(self.path):
            self.have = {(r['cfg'], r['variant'], int(r['instance']), int(r['N']), int(r['restart']))
                         for r in csv.DictReader(open(self.path))}
        new = not os.path.exists(self.path)
        self.f = open(self.path, 'a', newline='')
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(self.COLS)
        self.cache = {}

    def data(self, cfg, s, N):
        k = (cfg, s, N)
        if k not in self.cache:
            self.cache = {k: make_data(cfg, s, N)}
        return self.cache[k]

    def log(self, cfg, variant, s, N, r, acc, va, t0):
        row = ['E11', self.mode, cfg, variant, s, N, r, round(acc, 4), '' if va is None else round(va, 4),
               round(time.time() - t0, 1)]
        self.w.writerow(row)
        self.f.flush()
        print(*row, flush=True)

    def run(self, cfg, variant, kind, Ns, insts, restarts, **kw):
        for s in insts:
            for N in Ns:
                for r in range(restarts):
                    if (cfg, variant, s, N, r) in self.have:
                        continue
                    xtr, ytr, xva, yva, xte, yte, _ = self.data(cfg, s, N)
                    t0 = time.time()
                    mvar = 'shared' if variant.startswith('shared') else ('known' if variant.startswith('known') else variant)
                    m, gen = build(kind, mvar, 1000 * s + r, **kw)
                    fk = {k: kw[k] for k in ('lr', 'min_steps', 'epochs') if k in kw}
                    m, va = fit(m, gen, xtr, ytr, xva, yva, **fk)
                    self.log(cfg, variant, s, N, r, accm(m, xte, yte), va, t0)


def impl_mode(R):
    for s in range(10):
        for N in (500, 1000, 5000):
            xtr, ytr, xva, yva, xte, yte, (xtr_np, ytr_np, xte_np, yte_np) = R.data('impl', s, N)
            Ftr, Fte = parity_feats(sbits(xtr_np)), parity_feats(sbits(xte_np))
            for name in ('sk-liblinear', 'sk-newtoncg', 'torch-lbfgs'):
                if ('impl', name, s, N, 0) in R.have:
                    continue
                t0 = time.time()
                if name == 'sk-liblinear':
                    a = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000).fit(Ftr, ytr_np).score(Fte, yte_np)
                elif name == 'sk-newtoncg':
                    a = LogisticRegression(solver='newton-cg', C=1.0, max_iter=1000).fit(Ftr, ytr_np).score(Fte, yte_np)
                else:
                    X, Y = torch.tensor(Ftr, dtype=torch.float64), torch.tensor(ytr_np, dtype=torch.float64)
                    wv = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
                    opt = torch.optim.LBFGS([wv], max_iter=500, line_search_fn='strong_wolfe')

                    def closure():
                        opt.zero_grad()
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ wv, Y) + 0.5 * (wv ** 2).sum() / len(X)
                        loss.backward()
                        return loss
                    opt.step(closure)
                    a = float((((torch.tensor(Fte) @ wv.detach()) > 0).double() == torch.tensor(yte_np, dtype=torch.float64)).double().mean())
                R.log('impl', name, s, N, 0, float(a), None, t0)
    R.run('impl', 'known-adam', 'mask', (500, 1000, 5000), range(10), 1, A=suffix_mask())


def main(mode):
    if mode == 'n30':
        G = {'__name__': 'e11_n30'}
        exec(open('experiments/encoding-induced-false-security-pufs-pilot-e7gpu.py').read(), G)
        B = G['B']
        jobs = [(v, s, N, r, 'random', B['SBOX_AES'], None, 0.0)
                for s in range(10, 30) for v, N in (('shared', 30000), ('perbyte', 100000)) for r in range(3)]
        G['run']('n30', jobs)
        return
    R = Runner(mode)
    if mode == 'mix':
        for cfg, A in (('id', suffix_mask()), ('lin', mask_for(L_LIN))):
            R.run(cfg, 'known', 'mask', (500, 1000, 2000, 5000), range(10), 1, A=A)
            R.run(cfg, 'shared', 'mask', (20000, 30000, 40000, 60000, 100000, 200000), range(10), 3, A=A)
    elif mode == 'spn2':
        R.run('spn2', 'known', 'spn2', (100000, 400000, 1000000), range(5), 1)
        R.run('spn2', 'mlp', 'mlp', (100000, 400000, 1000000), range(5), 1, epochs=40)
        R.run('spn2', 'shared1', 'mask', (100000, 400000, 1000000), range(5), 3, A=suffix_mask())
        R.run('spn2', 'shared2', 'spn2', (100000, 400000, 1000000), range(5), 3)
    elif mode == 'xor':
        for k, kn, sh in ((2, (2000, 5000, 10000, 20000, 50000), (50000, 100000, 200000, 400000)),
                          (3, (10000, 20000, 50000, 100000), (100000, 200000, 400000, 800000))):
            R.run(f'xor{k}', 'known', 'mask', kn, range(5), 5, A=suffix_mask(), heads=k)
            R.run(f'xor{k}', 'shared', 'mask', sh, range(5), 5, A=suffix_mask(), heads=k)
    elif mode == 'hparam':
        for lr in (0.005, 0.01, 0.05):
            R.run('hp', f'shared-lr{lr}', 'mask', (30000,), range(10), 3, A=suffix_mask(), lr=lr)
        R.run('hp', 'shared-steps24k', 'mask', (30000,), range(10), 3, A=suffix_mask(), min_steps=24000)
        R.run('hp', 'shared-lr0.02', 'mask', (20000, 30000), range(10), 10, A=suffix_mask())
    elif mode == 'impl':
        impl_mode(R)
    R.f.close()


if __name__ == '__main__':
    for mode in sys.argv[1:]:
        main(mode)
