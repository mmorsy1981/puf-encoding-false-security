"""E12: the boundary of the secret-S-box attack. Is it the layer, or our learner?
Written and preregistered 2026-10-22, before any E12 run (only the import-time self-checks below,
which train nothing, were executed).

Why. The paper's method claim is that a learner's failure measures the learner, not the layer.
E11 produced three failures of our own structure-aware learner (T2) with the information provably
present (known table succeeds): a public GF(2)-linear layer after the secret S-box (mix/lin: 0/180
runs, max 0.510 at 200k; known 10/10 at 1k), 2-/3-XOR bases (xor2 best-of-5 2/5 at 400k, xor3 0/5
at 800k; known 5/5 by 2k/10k), and E9's cost growth with n (shared N50 12.5k/33.9k/106.5k at
n = 32/64/128, N2 falsified) under a FIXED step budget. E12 asks, for each, whether a better-matched
learner or budget removes the failure. Each outcome maps to a fixed wording (table at the end).

Common rig: E11 data path (AES S-box, labels via gate1, rng 11+100s train / 12+100s test, 10k test,
10% validation) and E11 training loop (Adam LR 0.02, batch 4096, >=6000 steps / >=20 epochs, BCE)
unless a part varies it; E12b uses E9's rig unchanged except the step budget. GPU (CUDA, TF32 off),
init and batch order drawn on the CPU from the seeds (as E7gpu); one process; every restart logged.
Best-of-k = validation-selected restart. CIs: Clopper-Pearson 95%. N50 = log-interpolated N at 50%
per-restart success (as E9). All parts run sequentially in one process (memory: one job at a time).

E12a  mix   learner matched to linear mixing (cfg 'id' = suffix mask, the E7 control; 'lin' =
            E11's public invertible L, rng 6000).
  Diagnosis it targets (disclosed peek, one unlogged diagnostic run, 2026-10-22, lin, instance 0,
  60k, E11 219-column character model): random init gives mean |feature| 0.0094 at initialisation
  and test acc 0.494; a warm start with 100% / 90.1% correct signs reaches 0.861 / 0.824. Reading:
  vanishing products from a random start, plus an over-parameterised table (219 free columns,
  56,064 parameters) that does not tie the columns to one S-box. E11's archived bit-sign model
  (tanh, N(0,1) init, products of up to 64 terms) stalled on 'id' too (0/62, best 0.762), so
  saturation, not the bit parameterisation alone, is the hypothesised fix.
  Model BitModel: one shared 256x8 table of S-box OUTPUT-BIT signs sigma (2,048 parameters, the true
  dimensionality), feature_k = prod_i sigma_i^{A_ki} over the 64 pre-L bits, then a linear layer.
    bit-ste (PRIMARY): forward sigma = sign(E) exactly +-1 (no vanishing), backward straight-through
             with d tanh(E)/dE; E ~ N(0,1).
    bit-sat (secondary): sigma = tanh(E), saturated init E = atanh(0.95) * random sign + N(0,0.1^2),
             so |sigma| ~ 0.95 and a 32-term product starts near 0.95^32 ~ 0.19, not ~1e-7.
    known  : sigma fixed to the true signs (control).
  Grid: known N {1k, 2k} x 1; id: bit-ste/bit-sat N {60k, 200k}; lin: bit-ste/bit-sat N {60k, 100k,
  200k, 400k, 1M}; instances 0-9, restarts 0-4 (seed 1000*s + r).
E12a  warm  basin of attraction (exploratory): lin, 60k, BitModel bit-ste initialised at the TRUE
            signs with a fraction p in {0.05, 0.15, 0.30, 0.45} of the 2,048 signs flipped,
            |E| = atanh(0.95); instances 0-4, one run each (seed 1000*s + 500 + 100p).
E12b  budget  E9 nscale rig and learner (E7 Structured, shared table), identical data and seeds
            (paired with the E9 1x rows), only MIN_STEPS changed: n=128, N {40k, 60k, 100k, 200k},
            steps {12000 (2x), 24000 (4x)}; n=64, N {20k, 30k, 40k, 60k}, steps 24000 (4x);
            instances 0-9, restarts 0-2. Success = acc >= E9 known-table acc (same instance, largest
            known N <= N) - 0.05, as E9.
E12c  xor   E11 xor learner unchanged (shared E7 table -> k linear heads -> soft XOR), more restarts
            and data, fresh seeds 100000 + 1000*s + r: k=2 N {400k, 800k, 1.6M}, restarts 0-19;
            k=3 N {800k, 1.6M, 3.2M}, restarts 0-9; instances 0-4. Success = acc >= 0.95 (as E11).

Predictions (before any E12 run; author's prior probability in brackets):
  K   known BitModel reaches >= 0.95 by 2k on >= 9/10 instances for id and for lin.           [0.95]
  G   GATE: bit-ste on id, best-of-5 >= 0.95 by 200k on >= 8/10.                             [0.6]
      If G fails, E12a says nothing about the lin layer (the learner fails where the layer is
      known to be learnable) and is reported as "inconclusive: learner inadequate".
  A1  PRIMARY: bit-ste on lin, best-of-5 >= 0.95 by 1M on >= 5/10.                           [0.4]
  A2  bit-sat on lin: exploratory, no directional prediction.
  W   warm: success (>= 0.95) for p <= 0.15 on >= 4/5, failure for p >= 0.30 on >= 4/5.      [0.5]
  B1  n=64: per-restart success at 4x steps within +-0.15 of the paired 1x rate at every N
      (E11 hparam steps24k at 30k: 0.47 vs 0.45).                                           [0.7]
  B2  n=128: N50 at 4x steps in [0.6, 1.0] x the 1x N50 (106.5k): budget matters, modestly.  [0.5]
  B3  r4 = N50(n=128, 4x) / N50(n=64, 4x) > 2.0, i.e. the growth with n is mostly real for this
      learner, not a budget artefact (1x: r = 3.14).                                        [0.65]
  X1  k=2: best-of-20 >= 0.95 by 1.6M on >= 4/5 (E11: per-restart 0.12 / 0.08 at 200k / 400k). [0.5]
  X2  k=3: exploratory, no prediction.

Decision table (fixed now; the paper uses the matching wording):
  E12a  G pass & A1 pass   -> "a public GF(2)-linear layer after a secret byte S-box does not stop
                              a structure-aware attacker" (cost reported vs the known table).
        G pass & A1 fails  -> "resisted two structure-aware learners up to 1M CRPs while the known
                              table succeeds at 1k: a learner bound, not evidence of security".
        G fails            -> "inconclusive (learner inadequate)"; the paper keeps E11's "resisted
                              our learner" wording and adds no claim.
  E12b  r4 <= 2.0          -> "the growth in E9 was largely an optimisation-budget effect";
        r4  > 2.0          -> "the cost of a secret table grows with n for this learner"
                              (reported with both budgets).
  E12c  X1 pass            -> "XOR bases raise the restart cost but do not stop the attack (k=2)";
        X1 fails           -> "k=2 XOR resisted this learner on >= 2/5 instances to 1.6M (learner
                              bound)"; k=3 reported descriptively.
Falsified (for the paper's "learned, not hidden" claim beyond byte-local layers): G pass & A1 fail.
Writes experiments/encoding-induced-false-security-pufs-e12-<part>.csv (append + resume)
cols: test,part,cfg,variant,n,instance,N,restart,steps,acc,val_acc,elapsed_s,hw
"""
import csv, os, sys, time, numpy as np, torch

E11 = {'__name__': 'e12_import'}
exec(open('experiments/encoding-induced-false-security-pufs-e11gen.py').read(), E11)
E9 = {'__name__': 'e12_import'}
exec(open('experiments/encoding-induced-false-security-pufs-e9scale.py').read(), E9)
DEV, S_SIGN, L_LIN = E11['DEV'], E11['S_SIGN'], E11['L_LIN']
suffix_mask, mask_for, internal_bits, parity_feats = E11['suffix_mask'], E11['mask_for'], E11['internal_bits'], E11['parity_feats']
CSV = 'experiments/encoding-induced-false-security-pufs-e12-{}.csv'
COLS = ['test', 'part', 'cfg', 'variant', 'n', 'instance', 'N', 'restart', 'steps', 'acc', 'val_acc', 'elapsed_s', 'hw']
KEY = ('cfg', 'variant', 'n', 'instance', 'N', 'restart', 'steps')
A_SAT = float(np.arctanh(0.95))
E9_MIN_STEPS = E9['MIN_STEPS']
assert E9['fit'].__globals__ is E9, 'E9.fit must read MIN_STEPS from the patched namespace'


# ---------------------------------------------------------------- E12a model
class BitModel(torch.nn.Module):
    """feature_k = prod_i sigma_i^{A_ki}; sigma = shared 256x8 table of S-box output-bit signs."""

    def __init__(self, variant, gen, A, p_flip=None):
        super().__init__()
        self.variant = variant
        self.register_buffer('A', torch.tensor(np.asarray(A, dtype=bool)))
        if variant == 'known':
            self.register_buffer('fixed', S_SIGN.clone())
        else:
            if variant == 'bit-ste':
                E = torch.randn((256, 8), generator=gen)
            elif variant == 'bit-sat':
                r = torch.where(torch.rand((256, 8), generator=gen) < 0.5, -1.0, 1.0)
                E = A_SAT * r + 0.1 * torch.randn((256, 8), generator=gen)
            elif variant == 'warm':
                flip = torch.zeros(256 * 8, dtype=torch.bool)
                flip[torch.randperm(256 * 8, generator=gen)[:round(p_flip * 2048)]] = True
                E = A_SAT * S_SIGN * torch.where(flip.reshape(256, 8), -1.0, 1.0)
            else:
                raise ValueError(variant)
            self.E = torch.nn.Parameter(E)
        self.lin = torch.nn.Linear(64, 1)

    def signs(self):
        if self.variant == 'known':
            return self.fixed
        t = torch.tanh(self.E)
        if self.variant == 'bit-sat':
            return t
        hard = torch.where(self.E >= 0, torch.ones_like(t), -torch.ones_like(t))
        return hard + (t - t.detach())                       # forward +-1, backward d tanh

    def feats(self, xb):
        s = self.signs()[xb].reshape(len(xb), 64)
        return torch.where(self.A[None], s[:, None, :], torch.ones((), device=s.device)).prod(2)

    def forward(self, xb):
        return self.lin(self.feats(xb)).squeeze(1)


def build_bit(variant, seed, A, p_flip=None):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    return BitModel(variant, gen, A, p_flip), gen


# ---------------------------------------------------------------- self-checks (run at import; no training)
def _checks():
    xb = np.random.default_rng(997).integers(0, 256, (400, 8)).astype(np.uint8)
    xt = torch.tensor(xb.astype(np.int64))
    for cfg, A in (('id', suffix_mask()), ('lin', mask_for(L_LIN))):
        ref = parity_feats(internal_bits(cfg, xb))[:, :64]
        m, _ = build_bit('known', 0, A)
        assert np.allclose(m.feats(xt).numpy(), ref), f'known BitModel != direct features ({cfg})'
        m, _ = build_bit('warm', 0, A, p_flip=0.0)
        f = m.feats(xt)
        assert np.allclose(f.detach().numpy(), ref), f'bit-ste at truth != direct features ({cfg})'
        m, _ = build_bit('bit-ste', 0, A)
        assert set(np.unique(m.signs().detach().numpy())) <= {-1.0, 1.0}, 'STE forward not +-1'
        torch.nn.functional.binary_cross_entropy_with_logits(m(xt), torch.rand(len(xt)).round()).backward()
        g = m.E.grad
        assert torch.isfinite(g).all() and (g != 0).any(), 'STE gives no gradient'
        m, _ = build_bit('bit-sat', 0, A)
        mag = m.signs().abs().mean().item()
        assert 0.93 < mag < 0.96, f'bit-sat init magnitude {mag}'
    for p in (0.05, 0.15, 0.30, 0.45):
        m, _ = build_bit('warm', 1, suffix_mask(), p_flip=p)
        wrong = int((torch.sign(m.E) != S_SIGN).sum())
        assert wrong == round(p * 2048), f'warm flips {wrong} != {round(p * 2048)}'


_checks()


# ---------------------------------------------------------------- runner
class Writer:
    def __init__(self, part):
        self.part, self.path = part, CSV.format(part)
        self.have = set()
        if os.path.exists(self.path):
            self.have = {tuple(r[k] for k in KEY) for r in csv.DictReader(open(self.path))}
        new = not os.path.exists(self.path)
        self.f = open(self.path, 'a', newline='')
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(COLS)
        self.hw = torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'

    def done(self, *key):
        return tuple(map(str, key)) in self.have

    def row(self, cfg, variant, n, s, N, r, steps, acc, va, t0):
        row = ['E12', self.part, cfg, variant, n, s, N, r, steps, round(acc, 4),
               '' if va is None else round(va, 4), round(time.time() - t0, 1), self.hw]
        self.w.writerow(row)
        self.f.flush()
        print(*row, flush=True)


def part_mix(W, warm=False):
    jobs = []
    if not warm:
        for cfg in ('id', 'lin'):
            jobs += [(cfg, 'known', None, s, N, 0, 1000 * s) for s in range(10) for N in (1000, 2000)]
        jobs += [('id', v, None, s, N, r, 1000 * s + r) for s in range(10) for N in (60000, 200000)
                 for v in ('bit-ste', 'bit-sat') for r in range(5)]
        jobs += [('lin', v, None, s, N, r, 1000 * s + r) for s in range(10)
                 for N in (60000, 100000, 200000, 400000, 1000000) for v in ('bit-ste', 'bit-sat') for r in range(5)]
    else:
        jobs += [('lin', f'warm-p{p:.2f}', p, s, 60000, 0, 1000 * s + 500 + round(100 * p))
                 for s in range(5) for p in (0.05, 0.15, 0.30, 0.45)]
    A = {'id': suffix_mask(), 'lin': mask_for(L_LIN)}
    cache = {}
    for cfg, variant, p, s, N, r, seed in jobs:
        if W.done(cfg, variant, 64, s, N, r, E11['MIN_STEPS']):
            continue
        if (cfg, s, N) not in cache:
            cache = {(cfg, s, N): E11['make_data'](cfg, s, N)}
        xtr, ytr, xva, yva, xte, yte, _ = cache[(cfg, s, N)]
        t0 = time.time()
        m, gen = build_bit('warm' if variant.startswith('warm') else variant, seed, A[cfg], p)
        m, va = E11['fit'](m, gen, xtr, ytr, xva, yva)
        W.row(cfg, variant, 64, s, N, r, E11['MIN_STEPS'], E11['accm'](m, xte, yte), va, t0)


def part_budget(W):
    plan = [(128, (40000, 60000, 100000, 200000), 12000), (128, (40000, 60000, 100000, 200000), 24000),
            (64, (20000, 30000, 40000, 60000), 24000)]
    try:
        for n, Ns, steps in plan:
            E9['MIN_STEPS'] = steps
            for s in range(10):
                for N in Ns:
                    data = None
                    for r in range(3):
                        if W.done('nscale', 'shared', n, s, N, r, steps):
                            continue
                        data = data or E9['make_data']('nscale', 'pypuf', n, 0.0, s, N)
                        xtr, ytr, xva, yva, xte, yte, _, _ = data
                        t0 = time.time()
                        m, va = E9['fit']('shared', 1000 * s + r, n // 8, xtr, ytr, xva, yva)
                        W.row('nscale', 'shared', n, s, N, r, steps, E9['acc'](m, xte, yte), va, t0)
    finally:
        E9['MIN_STEPS'] = E9_MIN_STEPS


def part_xor(W):
    for k, Ns, R in ((2, (400000, 800000, 1600000), 20), (3, (800000, 1600000, 3200000), 10)):
        for s in range(5):
            for N in Ns:
                data = None
                for r in range(R):
                    if W.done(f'xor{k}', 'shared', 64, s, N, r, E11['MIN_STEPS']):
                        continue
                    data = data or E11['make_data'](f'xor{k}', s, N)
                    xtr, ytr, xva, yva, xte, yte, _ = data
                    t0 = time.time()
                    m, gen = E11['build']('mask', 'shared', 100000 + 1000 * s + r, A=suffix_mask(), heads=k)
                    m, va = E11['fit'](m, gen, xtr, ytr, xva, yva)
                    W.row(f'xor{k}', 'shared', 64, s, N, r, E11['MIN_STEPS'], E11['accm'](m, xte, yte), va, t0)
                del data
                torch.cuda.empty_cache() if DEV.type == 'cuda' else None


def main(part):
    W = Writer(part)
    {'mix': lambda: part_mix(W), 'warm': lambda: part_mix(W, warm=True),
     'budget': lambda: part_budget(W), 'xor': lambda: part_xor(W)}[part]()
    W.f.close()


if __name__ == '__main__':
    for part in sys.argv[1:] or ['mix', 'warm', 'budget', 'xor']:
        main(part)
