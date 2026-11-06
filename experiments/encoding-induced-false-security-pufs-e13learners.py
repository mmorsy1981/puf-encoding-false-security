"""E13: stronger learners, for two open questions. (a) Is the generic uninformed baseline of the paper too weak?
(b) Can a different learner family cross the mixing boundary on which E11 and E12 failed?
Written and preregistered 2026-10-24, before any E13 run (only the import-time self-checks, which train nothing, ran).

Common rig: E11's (AES S-box, labels by gate1, rng 11+100s train / 12+100s test, 10k test, 10% validation, seed 1000*s + r),
GPU (CUDA, TF32 off, init and batch order drawn on the CPU from the seeds), one process, every restart logged.
Layers: cfg 'id' (secret S-box, then APUF) and 'lin' (secret S-box, public invertible GF(2) matrix L, rng 6000, then APUF).

Parts (python ...-e13learners.py <part>):
  t0plus : stronger UNINFORMED learners on the S-box variant ('id'), N = 1M, instances 0-2, seed 0.
           Input = parity features of the external challenge (65-dim), as in the paper's MLP.
           A: 1024-512-256-128, 40 epochs, Adam 1e-3 | B: same net, 200 epochs, 1e-3 |
           C: 2048-1024-512-256-128, 200 epochs, 3e-4 | D: 512-256-128, 200 epochs, 1e-3, weight decay 1e-4.
  be-id  : BYTE-EMBEDDING MLP (a generic network that knows the challenge is 8 bytes and nothing about the APUF):
           per-position embedding of the byte (8 x 256 x 64), concatenated (512) -> 1024-512-256-1, Adam 1e-3,
           >= 30000 steps. Identity layer, N = 200k, instances 0-9, restarts 0-2. This is the gate for be-lin.
  cal-id : COORDINATE-ASCENT learner (discrete, structure-aware): the shared table T: byte -> byte is a hard table,
           features X_k = prod_b chi_{m_kb}(T[x_b]) with +-1 characters; alternate (i) a logistic-regression refit of the 65
           weights (Newton, ridge 1e-2) and (ii) a block update of 16 random table entries, each set to the value that
           maximises the log-likelihood of the samples containing it (all other entries fixed; exact under the
           factorisation, scored for all 256 candidate values at once by a matrix product over the character columns).
           1200 iterations, random initial table, validation-selected best iterate. Identity layer, N in {60k, 200k},
           instances 0-9, restarts 0-2. This is the gate for cal-lin.
  be-lin, cal-lin : the same learners on the linear layer, N in {200k, 1M}, instances 0-9, restarts 0-2. Each runs ONLY IF its
           own identity gate passed (best-of-3 validation-selected >= 0.95 on >= 8/10 instances at 200k); otherwise the part
           prints the gate result and exits without training (decision rule fixed now).

Predictions (before any E13 run; author's prior in brackets):
  U1 t0plus: every configuration stays below 0.80 test accuracy at 1M on all three instances.             [0.85]
     If any reaches 0.90 the paper's statement that generic learners stall on the S-box variant is retracted.
  B1 be-id fails the gate (fewer than 8/10 instances solved by 200k).                                      [0.90]
  K1 cal-id passes the gate.                                                                               [0.35]
  K2 given K1, cal-lin solves >= 5/10 instances by 1M.                                                     [0.30]
Decision table: gate failed -> the mixing result stays "resisted our learners" and the learner is reported as inadequate;
gate passed and lin solved -> the secret-S-box result extends to public linear mixing and the boundary moves;
gate passed and lin unsolved -> a learner bound with a stronger learner behind it.
Disclosed peek: one timing probe (20 iterations of cal on instance 0, N = 200k, layers id and lin, seed 0; accuracy not
inspected) measured 0.39 s and 0.43 s per iteration, i.e. about 8 minutes per 1200-iteration run.
Writes experiments/encoding-induced-false-security-pufs-e13learners.csv (append + resume)
cols: test,part,cfg,learner,instance,N,restart,acc,val_acc,elapsed_s,hw
"""
import csv, os, sys, time, numpy as np, torch

E11 = {'__name__': 'e13_import'}
exec(open('experiments/encoding-induced-false-security-pufs-e11gen.py').read(), E11)
DEV, make_data, fit, accm = E11['DEV'], E11['make_data'], E11['fit'], E11['accm']
suffix_mask, mask_for, L_LIN = E11['suffix_mask'], E11['mask_for'], E11['L_LIN']
CSV = 'experiments/encoding-induced-false-security-pufs-e13learners.csv'
COLS = ['test', 'part', 'cfg', 'learner', 'instance', 'N', 'restart', 'acc', 'val_acc', 'elapsed_s', 'hw']
KEY = ('cfg', 'learner', 'instance', 'N', 'restart')
HW = torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'


# ---------------------------------------------------------------- t0plus and be: neural learners
def parity_from_bytes(xb):
    bits = ((xb[:, :, None] >> torch.arange(7, -1, -1, device=xb.device)) & 1).reshape(len(xb), 64)
    s = 1 - 2 * bits.float()
    f = torch.flip(torch.cumprod(torch.flip(s, [1]), 1), [1])
    return torch.cat([f, torch.ones_like(f[:, :1])], 1)


class ParityMLP(torch.nn.Module):
    def __init__(self, widths, gen):
        super().__init__()
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
        dims, layers = [65] + list(widths), []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(dims[-1], 1))

    def forward(self, xb):
        return self.net(parity_from_bytes(xb)).squeeze(1)


class ByteEmbedMLP(torch.nn.Module):
    def __init__(self, gen, d=64):
        super().__init__()
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
        self.emb = torch.nn.Embedding(8 * 256, d)
        self.register_buffer('off', torch.arange(8) * 256)
        self.net = torch.nn.Sequential(torch.nn.Linear(8 * d, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 512),
                                       torch.nn.ReLU(), torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))

    def forward(self, xb):
        return self.net(self.emb(xb + self.off).reshape(len(xb), -1)).squeeze(1)


# ---------------------------------------------------------------- cal: coordinate ascent over a hard shared table
class CAL:
    """Hard shared table T[x] in 0..255. feature_k = prod_b chi_{m_kb}(T[x_b]), chi_m(v) = (-1)^{popcount(v & m)}."""

    def __init__(self, A):
        m = np.packbits(np.asarray(A, dtype=np.uint8).reshape(64, 8, 8), axis=2)[:, :, 0].astype(int)     # (k, b) masks
        self.M = [0] + sorted({int(v) for v in m.ravel() if v}, reverse=True)
        pos = {v: j for j, v in enumerate(self.M)}
        self.I = torch.tensor([[pos[int(v)] for v in row] for row in m], device=DEV)                       # (64, 8)
        v = np.arange(256)[:, None]
        mm = np.array(self.M)[None, :]
        pc = np.array([[bin(a & b).count('1') & 1 for b in mm[0]] for a in v[:, 0]])
        self.C = torch.tensor(1.0 - 2.0 * pc, dtype=torch.float32, device=DEV)                             # (256, |M|)
        self.O = []
        for b in range(8):
            o = torch.zeros(64, len(self.M), device=DEV)
            o[torch.arange(64), self.I[:, b]] = 1.0
            self.O.append(o)

    def feats(self, T, xb):
        H = self.C[T]
        return H[xb[:, None, :], self.I[None]].prod(2)

    def fit_w(self, F, y, w, iters=2, ridge=1e-2):
        X = torch.cat([F, torch.ones_like(F[:, :1])], 1)
        for _ in range(iters):
            p = torch.sigmoid(X @ w)
            g = X.T @ (p - y) + ridge * w
            Hm = (X * (p * (1 - p))[:, None]).T @ X + ridge * torch.eye(65, device=DEV)
            w = w - torch.linalg.solve(Hm, g)
        return w

    def scores(self, T, xb, y, w, chunk=40000):
        H = self.C[T]
        score = torch.zeros(256, 256, device=DEV)
        for s0 in range(0, len(xb), chunk):
            x, yy = xb[s0:s0 + chunk], y[s0:s0 + chunk]
            F = H[x[:, None, :], self.I[None]].prod(2)
            for b in range(8):
                P = F * H[x[:, b][:, None], self.I[:, b][None]]
                U = (P * w[:64]) @ self.O[b]
                logit = U @ self.C.T + w[64]
                ll = yy[:, None] * torch.nn.functional.logsigmoid(logit) + (1 - yy[:, None]) * torch.nn.functional.logsigmoid(-logit)
                score.index_add_(0, x[:, b], ll)
        return score

    def acc(self, T, w, xb, y):
        F = self.feats(T, xb)
        return float((((torch.cat([F, torch.ones_like(F[:, :1])], 1) @ w) > 0).float() == y).float().mean())

    def run(self, xtr, ytr, xva, yva, seed, iters=1200, block=16):
        gen = torch.Generator().manual_seed(seed)
        T = torch.randint(0, 256, (256,), generator=gen).to(DEV)
        w = torch.zeros(65, device=DEV)
        best = (-1.0, T.clone(), w.clone())
        for it in range(iters):
            F = self.feats(T, xtr)
            w = self.fit_w(F, ytr, w)
            sc = self.scores(T, xtr, ytr, w)
            sel = torch.randperm(256, generator=gen)[:block].to(DEV)
            T[sel] = sc[sel].argmax(1)
            if it % 50 == 49 or it == iters - 1:
                w = self.fit_w(self.feats(T, xtr), ytr, w)
                va = self.acc(T, w, xva, yva)
                if va > best[0]:
                    best = (va, T.clone(), w.clone())
        return best


# ---------------------------------------------------------------- self-checks (run at import; no training)
def _checks():
    xb = np.random.default_rng(991).integers(0, 256, (300, 8)).astype(np.int64)
    xt = torch.tensor(xb, device=DEV)
    for cfg, A in (('id', suffix_mask()), ('lin', mask_for(L_LIN))):
        cal = CAL(A)
        ref = E11['parity_feats'](E11['internal_bits'](cfg, xb.astype(np.uint8)))[:, :64]
        got = cal.feats(torch.tensor(E11['SBOX'].astype(np.int64), device=DEV), xt).cpu().numpy()
        assert np.allclose(got, ref), f'CAL features with the true table != direct features ({cfg})'
    # the block update must be exact: scoring the true table's entry against all 256 candidates ranks the truth first
    # on noise-free data when the weights are the true ones
    cal = CAL(suffix_mask())
    puf = E11['ArbiterPUF'](n=64, seed=5)
    xs = np.random.default_rng(992).integers(0, 256, (20000, 8)).astype(np.uint8)
    y = torch.tensor(E11['labels'](puf, E11['internal_bits']('id', xs)).astype(np.float32), device=DEV)
    xs = torch.tensor(xs.astype(np.int64), device=DEV)
    Ttrue = torch.tensor(E11['SBOX'].astype(np.int64), device=DEV)
    w = cal.fit_w(cal.feats(Ttrue, xs), y, torch.zeros(65, device=DEV), iters=8)
    sc = cal.scores(Ttrue, xs, y, w)
    hit = float((sc.argmax(1) == Ttrue).float().mean())
    assert hit > 0.9, f'block update does not recover the true table from the true weights (hit rate {hit:.2f})'


_checks()


# ---------------------------------------------------------------- runner
class Writer:
    def __init__(self):
        self.have = set()
        if os.path.exists(CSV):
            self.have = {tuple(r[k] for k in KEY) for r in csv.DictReader(open(CSV))}
        new = not os.path.exists(CSV)
        self.f = open(CSV, 'a', newline='')
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(COLS)

    def done(self, *key):
        return tuple(map(str, key)) in self.have

    def row(self, part, cfg, learner, s, N, r, acc, va, t0):
        row = ['E13', part, cfg, learner, s, N, r, round(acc, 4), '' if va is None else round(va, 4), round(time.time() - t0, 1), HW]
        self.w.writerow(row)
        self.f.flush()
        print(*row, flush=True)


def gate_passed(learner):
    """best-of-3 validation-selected >= 0.95 on >= 8/10 identity-layer instances at N = 200k."""
    if not os.path.exists(CSV):
        return False, 0
    g = {}
    for r in csv.DictReader(open(CSV)):
        if r['cfg'] == 'id' and r['learner'] == learner and r['N'] == '200000':
            g.setdefault(int(r['instance']), []).append(r)
    k = sum(float(max(v, key=lambda r: float(r['val_acc']))['acc']) >= 0.95 for v in g.values())
    return (len(g) == 10 and k >= 8), k


def part_t0plus(W):
    cfgs = {'A': ((1024, 512, 256, 128), 40, 1e-3, 0.0), 'B': ((1024, 512, 256, 128), 200, 1e-3, 0.0),
            'C': ((2048, 1024, 512, 256, 128), 200, 3e-4, 0.0), 'D': ((512, 256, 128), 200, 1e-3, 1e-4)}
    for s in range(3):
        data = None
        for name, (widths, epochs, lr, wd) in cfgs.items():
            if W.done('id', f'parityMLP-{name}', s, 1000000, 0):
                continue
            data = data or make_data('id', s, 1000000)
            xtr, ytr, xva, yva, xte, yte, _ = data
            t0 = time.time()
            gen = torch.Generator().manual_seed(1000 * s)
            m = ParityMLP(widths, gen).to(DEV)
            opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
            lossf = torch.nn.BCEWithLogitsLoss()
            n = len(xtr)
            for _ in range(epochs):
                perm = torch.randperm(n, generator=gen).to(DEV)
                for s0 in range(0, n, 4096):
                    idx = perm[s0:s0 + 4096]
                    opt.zero_grad()
                    lossf(m(xtr[idx]), ytr[idx]).backward()
                    opt.step()
            W.row('t0plus', 'id', f'parityMLP-{name}', s, 1000000, 0, accm(m, xte, yte), accm(m, xva, yva), t0)


def part_be(W, cfg, Ns):
    for s in range(10):
        for N in Ns:
            for r in range(3):
                if W.done(cfg, 'byte-embed-MLP', s, N, r):
                    continue
                xtr, ytr, xva, yva, xte, yte, _ = make_data(cfg, s, N)
                t0 = time.time()
                gen = torch.Generator().manual_seed(1000 * s + r)
                m = ByteEmbedMLP(gen)
                m, va = fit(m, gen, xtr, ytr, xva, yva, lr=1e-3, min_steps=30000)
                W.row('be-' + cfg, cfg, 'byte-embed-MLP', s, N, r, accm(m, xte, yte), va, t0)


def part_cal(W, cfg, Ns):
    cal = CAL(suffix_mask() if cfg == 'id' else mask_for(L_LIN))
    for s in range(10):
        for N in Ns:
            data = None
            for r in range(3):
                if W.done(cfg, 'coord-ascent', s, N, r):
                    continue
                data = data or make_data(cfg, s, N)
                xtr, ytr, xva, yva, xte, yte, _ = data
                t0 = time.time()
                va, T, w = cal.run(xtr, ytr, xva, yva, 1000 * s + r)
                W.row('cal-' + cfg, cfg, 'coord-ascent', s, N, r, cal.acc(T, w, xte, yte), va, t0)


def main(part):
    W = Writer()
    if part == 't0plus':
        part_t0plus(W)
    elif part == 'be-id':
        part_be(W, 'id', (200000,))
    elif part == 'cal-id':
        part_cal(W, 'id', (60000, 200000))
    elif part in ('be-lin', 'cal-lin'):
        learner = 'byte-embed-MLP' if part == 'be-lin' else 'coord-ascent'
        ok, k = gate_passed(learner)
        print(f'gate for {learner}: {k}/10 identity instances solved at 200k -> {"PASS" if ok else "FAIL (part skipped)"}')
        if ok:
            (part_be if part == 'be-lin' else part_cal)(W, 'lin', (200000, 1000000))
    W.f.close()


if __name__ == '__main__':
    for part in sys.argv[1:] or ['t0plus', 'be-id', 'cal-id', 'be-lin', 'cal-lin']:
        main(part)
