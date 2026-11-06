"""E7gpu: GPU cross-check of E7/E7b, plus the matched-ratio grid (known-table control at small N).
Written 2026-10-20 after E7b finished on CPU. Reuses E7b's data(), build(), jobs_for() and E7's model
unchanged; only the arithmetic device differs. Initialisation and batch order are drawn on the
CPU from the same seeds as the CPU runs, then the model and data move to CUDA, so every GPU row
pairs one-to-one with a CPU row (same mode/variant/instance/N/restart/init). TF32 disabled.
GPU scatter-add in the table gradient is nondeterministic, so pairs are compared by outcome class
(success = acc >= 0.95; acc_clean for noise) and by |acc difference|, not bitwise.

Modes (python ...-e7gpu.py ratio|xcheck):
  ratio  : 'known' (true table fixed; same learner, same loop) on instances 0-9,
           N in {250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000}, restart 0. Gives the
           informed-attacker N-to-0.90/0.95 under the SAME learner, split and metric as the
           unknown-table runs, for a matched sample-cost ratio.
  xcheck : every E7b job (corrupt, ext, random, noise) plus the E7 cells E7b does not cover
           (inst 0-2: known @20k/100k/200k/1M; shared @100k/200k/1M; perbyte @20k/30k/40k/1M;
           3 restarts each for shared/perbyte).
Prediction (before any GPU run): outcome class agrees on >=95% of paired runs and per-cell
success rates stay inside the CPU 95% CIs; disagreements, if any, sit in threshold cells.
Writes experiments/encoding-induced-false-security-pufs-pilot-e7gpu.csv (append + resume)
cols: test,mode,variant,instance,N,restart,init,acc,acc_clean,val_acc,elapsed_s,device
"""
import csv, os, sys, time, numpy as np, torch

B = {'__name__': 'e7gpu_import'}
exec(open('experiments/encoding-induced-false-security-pufs-pilot-e7b.py').read(), B)
DEV = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
CSV = 'experiments/encoding-induced-false-security-pufs-pilot-e7gpu.csv'
SBOX_AES, G_AES, acc = B['SBOX_AES'], B['G_AES'], B['acc']
LR, BS, MIN_STEPS, MIN_EPOCHS = B['LR'], B['BS'], B['MIN_STEPS'], B['MIN_EPOCHS']


def _forward(self, xb):
    """E7 Structured.forward with the per-byte arange on the input's device."""
    T = self.table()
    if self.variant == 'perbyte':
        g = T[torch.arange(8, device=xb.device)[None, :], xb]
    else:
        g = T[xb]
    P = g[:, :, 0]
    suf = torch.flip(torch.cumprod(torch.flip(P, [1]), 1), [1])
    suf = torch.cat([suf[:, 1:], torch.ones_like(suf[:, :1])], 1)
    return self.lin((g * suf[:, :, None]).reshape(len(xb), 64)).squeeze(1)


B['Structured'].forward = _forward


def fit(model, gen, xtr, ytr, xva, yva):
    """E7 train() loop; randperm on the CPU generator, indices moved to the GPU."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(xtr)
    epochs = max(MIN_EPOCHS, -(-MIN_STEPS // max(1, n // BS)))
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen).to(DEV)
        for s in range(0, n, BS):
            idx = perm[s:s + BS]
            opt.zero_grad()
            lossf(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
    return model, acc(model, xva, yva)


def done():
    if not os.path.exists(CSV):
        return set()
    return {(r['mode'], r['variant'], int(r['instance']), int(r['N']), int(r['restart']), r['init'])
            for r in csv.DictReader(open(CSV))}


def run(mode, jobs):
    have = done()
    new = not os.path.exists(CSV)
    with open(CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['test', 'mode', 'variant', 'instance', 'N', 'restart', 'init',
                        'acc', 'acc_clean', 'val_acc', 'elapsed_s', 'device'])
        cache = {}
        for variant, s, N, r, init, sbox, G, nz in jobs:
            if (mode, variant, s, N, r, init) in have:
                continue
            key = (s, N, sbox.tobytes(), nz)
            if key not in cache:
                cache.clear()
                cache[key] = tuple(t.to(DEV) for t in B['data'](s, N, sbox, nz))
            xtr, ytr, xva, yva, xte, yte, yte_c = cache[key]
            t0 = time.time()
            p = None if init in ('random', 'fixed') else float(init.split('=')[1])
            m, gen = B['build'](variant, 1000 * s + r, G=G, corrupt=p)
            m, va = fit(m.to(DEV), gen, xtr, ytr, xva, yva)
            torch.cuda.synchronize()
            row = ['E7gpu', mode, variant, s, N, r, init, round(acc(m, xte, yte), 4),
                   round(acc(m, xte, yte_c), 4), round(va, 4), round(time.time() - t0, 1),
                   torch.cuda.get_device_name(0)]
            w.writerow(row)
            f.flush()
            print(*row, flush=True)


def jobs_for(mode):
    if mode == 'ratio':
        return [('known', s, N, 0, 'fixed', SBOX_AES, G_AES, 0.0) for s in range(10)
                for N in (250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000)]
    if mode == 'e7':
        J = []
        for s in range(3):
            for N in (20000, 100000, 200000, 1000000):
                J.append(('known', s, N, 0, 'fixed', SBOX_AES, G_AES, 0.0))
            for variant, grid in (('shared', (100000, 200000, 1000000)),
                                  ('perbyte', (20000, 30000, 40000, 1000000))):
                for N in grid:
                    J += [(variant, s, N, r, 'random', SBOX_AES, None, 0.0) for r in range(3)]
        return J
    return B['jobs_for'](mode)


if __name__ == '__main__':
    modes = sys.argv[1:] or ['ratio', 'xcheck']
    for mode in modes:
        for sub in (['corrupt', 'ext', 'random', 'noise', 'e7'] if mode == 'xcheck' else [mode]):
            run(sub, jobs_for(sub))
