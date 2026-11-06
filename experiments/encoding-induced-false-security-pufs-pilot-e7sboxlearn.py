"""E7: can a structure-aware attacker LEARN an unknown bytewise S-box jointly with the APUF?
Attacker knows the architecture (bytewise substitution, then a 64-bit APUF) but not the table.

Model. APUF parity feature i = prod_{j>=i} s_j factorises by byte: for i = 8b+k,
  X_i = G_b[x_b, k] * prod_{b'>b} G_{b'}[x_{b'}, 0],   G[x, k] = prod_{t=k..7} s'(S[x])_t,
so the composite is a linear threshold over per-byte lookups of the (unknown) byte map.
The learner fits tables E (tanh-squashed, init N(0,1)) plus 64 weights + bias with Adam/BCE:
  shared  : one 256x8 table for all bytes (attacker assumes the same S-box per byte)
  perbyte : eight 256x8 tables (no sharing assumption)
  known   : table fixed to the true G (rig control; must reach ~T1)
Selection by 10% held-out validation over R random restarts; test on 10k fresh CRPs.
Recovery = per table column, sign agreement with the true G column (max(a, 1-a): column sign
flips are a symmetry absorbed by the weights), averaged over columns.

Data paired with pilot-e5plainparmlp.py / e5sboxparmlp-s12.py: inst s -> puf seed s,
train rng 11+100s, test rng 12+100s (s=0: rng 11/12 as e5parmlp). Noiseless 64-bit APUF,
gate1 AES S-box. CPU (GPU busy with another job). N grid 20k/100k/200k/1M (nested).

Preregistered (written before any run):
  prediction: shared reaches >=0.95 test by N<=200k on >=2/3 instances; perbyte is slower
              (more parameters) but improves monotonically with N.
  rig check : 'known' must reach >=0.97 at 20k, else the factorisation or labels are wrong
              (a startup assert also checks the factorised features against gate1 exactly).
  failure   : if shared stays <0.70 through 1M with the known control passing, report it as
              evidence of hardness from an unknown static layer against this attacker
              (optimisation limit, not an impossibility result).
Post hoc (added after the preregistered grid ran, disclosed): N in {30k, 40k, 60k} via
`python ...e7sboxlearn.py fine`, to bound the 20k->100k transition. Same CSV; rows are
distinguishable by N. The 'recovery' column is column-wise and NOT identifiable (only the
per-byte weighted sum h_b and the byte parity are); see outputs/...-review-probes/probe_e7_ident.py
(P-recovery 1.000, h_b |corr| >= 0.998 at 100k on all 3 instances).
Writes experiments/encoding-induced-false-security-pufs-pilot-e7sboxlearn.csv
cols: test,variant,instance,N,acc,val_acc,recovery,restart_best,elapsed_s   (append + resume)
"""
import csv, os, sys, time, numpy as np, torch
from pypuf.simulation import ArbiterPUF

CSV = 'experiments/encoding-induced-false-security-pufs-pilot-e7sboxlearn.csv'
NGRID = [20000, 100000, 200000, 1000000]
INSTANCES = [0, 1, 2]
VARIANTS = ['known', 'shared', 'perbyte']
RESTARTS, BS, LR, MIN_STEPS, MIN_EPOCHS = 3, 4096, 0.02, 6000, 20
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

src = open('experiments/encoding-induced-false-security-pufs-gate1.py').read()
G1 = {}
exec(src.split(chr(10) + "if __name__ == '__main__':")[0].split('INSTANCES =')[0], G1)
SBOX, labels, parity_feats = G1['SBOX_ARR'], G1['labels'], G1['parity_feats']
bits_to_bytes, pi_sbox = G1['bits_to_bytes'], G1['pi_sbox']

# true per-byte lookup G[x, k] = prod_{t=k..7} (1 - 2*bit_t(S[x])), bits MSB-first (unpackbits)
out_bits = np.unpackbits(SBOX.reshape(-1, 1), axis=1).astype(np.float64)
s_out = 1 - 2 * out_bits
G_TRUE = np.stack([np.prod(s_out[:, k:], axis=1) for k in range(8)], axis=1)  # (256, 8)


def factor_feats(Xb, G):
    """Xb: (N,8) byte indices; G: (256,8) table -> (N,64) parity features."""
    g = G[Xb]                                   # (N, 8 bytes, 8 k)
    P = g[:, :, 0]                              # byte parities
    suf = np.ones_like(P)
    for b in range(6, -1, -1):
        suf[:, b] = suf[:, b + 1] * P[:, b + 1]
    return (g * suf[:, :, None]).reshape(len(Xb), 64)


_C = np.random.default_rng(999).integers(0, 2, (2000, 64)).astype(np.uint8)
assert np.array_equal(factor_feats(bits_to_bytes(_C), G_TRUE), parity_feats(pi_sbox(_C))[:, :64]), \
    'factorised features do not match gate1 parity_feats(pi_sbox(C))'


class Structured(torch.nn.Module):
    def __init__(self, variant, gen):
        super().__init__()
        self.variant = variant
        if variant == 'known':
            self.register_buffer('fixed', torch.tensor(G_TRUE, dtype=torch.float32))
        else:
            shape = (256, 8) if variant == 'shared' else (8, 256, 8)
            self.E = torch.nn.Parameter(torch.randn(shape, generator=gen))
        self.lin = torch.nn.Linear(64, 1)

    def table(self):
        if self.variant == 'known':
            return self.fixed
        return torch.tanh(self.E)

    def forward(self, xb):
        T = self.table()
        if self.variant == 'perbyte':
            g = T[torch.arange(8)[None, :], xb]          # (N, 8, 8)
        else:
            g = T[xb]                                    # (N, 8, 8)
        P = g[:, :, 0]
        suf = torch.flip(torch.cumprod(torch.flip(P, [1]), 1), [1])   # prod_{b'>=b}
        suf = torch.cat([suf[:, 1:], torch.ones_like(suf[:, :1])], 1)  # prod_{b'>b}
        return self.lin((g * suf[:, :, None]).reshape(len(xb), 64)).squeeze(1)


def recovery(model):
    if model.variant == 'known':
        return 1.0
    T = np.sign(model.table().detach().numpy())
    tabs = [T] if model.variant == 'shared' else list(T)
    agr = [max(a, 1 - a) for t in tabs for a in (t == np.sign(G_TRUE)).mean(0)]
    return float(np.mean(agr))


def acc(model, xb, y):
    with torch.no_grad():
        return float(((model(xb) > 0).float() == y).float().mean())


def train(variant, xtr, ytr, xva, yva, seed):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = Structured(variant, gen)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(xtr)
    steps_per_epoch = max(1, n // BS)
    epochs = max(MIN_EPOCHS, -(-MIN_STEPS // steps_per_epoch))
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, BS):
            idx = perm[s:s + BS]
            opt.zero_grad()
            lossf(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
    return model, acc(model, xva, yva)


def done():
    if not os.path.exists(CSV):
        return set()
    return {(r['variant'], int(r['instance']), int(r['N'])) for r in csv.DictReader(open(CSV))}


def main(ngrid, instances, variants):
    have = done()
    new = not os.path.exists(CSV)
    f = open(CSV, 'a', newline='')
    w = csv.writer(f)
    if new:
        w.writerow(['test', 'variant', 'instance', 'N', 'acc', 'val_acc', 'recovery', 'restart_best', 'elapsed_s'])
    for s in instances:
        puf = ArbiterPUF(n=64, seed=s, noisiness=0.0)
        Ctr = np.random.default_rng(11 + 100 * s).integers(0, 2, (max(ngrid), 64)).astype(np.uint8)
        Cte = np.random.default_rng(12 + 100 * s).integers(0, 2, (10000, 64)).astype(np.uint8)
        ytr_all = np.concatenate([labels(puf, pi_sbox(Ctr[i:i + 100000])) for i in range(0, len(Ctr), 100000)])
        yte = labels(puf, pi_sbox(Cte))
        Xtr_all = torch.tensor(bits_to_bytes(Ctr).astype(np.int64))
        Xte = torch.tensor(bits_to_bytes(Cte).astype(np.int64))
        Ytr_all = torch.tensor(ytr_all, dtype=torch.float32)
        Yte = torch.tensor(yte, dtype=torch.float32)
        for N in ngrid:
            nva = N // 10
            xtr, ytr = Xtr_all[:N - nva], Ytr_all[:N - nva]
            xva, yva = Xtr_all[N - nva:N], Ytr_all[N - nva:N]
            for v in variants:
                if (v, s, N) in have:
                    print('skip', v, s, N, flush=True)
                    continue
                t0 = time.time()
                best = None
                for r in range(1 if v == 'known' else RESTARTS):
                    m, va = train(v, xtr, ytr, xva, yva, 1000 * s + r)
                    if best is None or va > best[1]:
                        best = (m, va, r)
                m, va, r = best
                row = ['E7', v, s, N, round(acc(m, Xte, Yte), 4), round(va, 4), round(recovery(m), 4), r,
                       round(time.time() - t0, 1)]
                w.writerow(row)
                f.flush()
                print(*row, flush=True)
    f.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        CSV = CSV.replace('.csv', '-smoke.csv')
        main([20000], [0], VARIANTS)
    elif len(sys.argv) > 1 and sys.argv[1] == 'fine':
        main([30000, 40000, 60000], INSTANCES, ['shared', 'perbyte'])
    else:
        main(NGRID, INSTANCES, VARIANTS)
