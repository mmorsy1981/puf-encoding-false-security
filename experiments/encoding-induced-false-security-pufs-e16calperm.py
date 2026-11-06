"""E16: coordinate ascent with a permutation polish, gated on HELD-OUT instances.
Written and preregistered 2026-10-24, before any E16 run (only the import-time self-checks, which train nothing, ran).

Why this exists. E13's coordinate-ascent learner (cal) failed its preregistered identity-layer gate narrowly: best-of-3
validation-selected restarts solved 7/10 instances at 200k (gate: >= 8/10), so E13 correctly skipped the linear layer.
Looking at the per-restart results after the gate (a post-hoc look, disclosed) showed that most restarts end at 0.93-0.95
and several at exactly 1.000: the learner reaches almost-correct tables. cal updates an UNCONSTRAINED table (the true S-box
is a permutation), so a few colliding entries can persist. E16 adds the missing constraint and nothing else.

Learner cal+ (E13's cal plus a polish phase). Phase 1 = E13 cal, 1200 iterations, unchanged. Phase 2 starts from the
validation-best iterate of phase 1 and runs 400 iterations; each iteration refits the 65 weights (Newton) and computes the
full 256 x 256 candidate score matrix as in phase 1, then replaces the table by the best PERMUTATION: the linear assignment
(scipy linear_sum_assignment, maximising the summed score) over all entries at once. The final table is the validation-best
iterate over both phases. No other change (same loss, ridge, block size, seeds 1000*s + r).

Held-out design. To keep the learner from being tuned to the E13 data, E16 uses instances 10-19 (new PUF seeds, new data
rngs 11+100s / 12+100s) for every gate and result.
Parts (python ...-e16calperm.py <part>):
  id  : identity layer ('id'), N in {100k, 200k}, instances 10-19, restarts 0-2. Gate = best-of-3 validation-selected >= 0.95
        on >= 8/10 instances at 200k.
  lin : linear layer ('lin', public L, rng 6000), N in {200k, 500k}, instances 10-19, restarts 0-2. Runs ONLY IF the id gate
        passed (otherwise prints the gate and exits).

Predictions (before any E16 run; author's prior in brackets):
  G2 cal+ passes the id gate on the held-out instances.                                                     [0.55]
  L2 given G2, cal+ solves >= 5/10 lin instances (best-of-3, 0.95) at 500k.                                  [0.35]
  M2 polish helps: the per-restart rate of reaching 0.95 at id/200k is higher than E13 cal's 0.33 (instances
     0-9) by at least 0.15 [different instances, so a heuristic comparison].                                 [0.6]
Decision table: gate failed -> report cal and cal+ as inadequate and keep "resisted our learners"; gate passed and lin
solved -> the secret-S-box result extends to public GF(2)-linear mixing for n = 64 and the boundary moves to SPN and XOR;
gate passed and lin unsolved -> a learner bound with a verified learner behind it.
Writes experiments/encoding-induced-false-security-pufs-e16calperm.csv (append + resume)
cols: test,part,cfg,learner,instance,N,restart,acc,val_acc,elapsed_s,hw
"""
import csv, os, sys, time, numpy as np, torch
from scipy.optimize import linear_sum_assignment

E13 = {'__name__': 'e16_import'}
exec(open('experiments/encoding-induced-false-security-pufs-e13learners.py').read(), E13)
CAL, DEV, make_data = E13['CAL'], E13['DEV'], E13['make_data']
suffix_mask, mask_for, L_LIN = E13['suffix_mask'], E13['mask_for'], E13['L_LIN']
CSV = 'experiments/encoding-induced-false-security-pufs-e16calperm.csv'
COLS = ['test', 'part', 'cfg', 'learner', 'instance', 'N', 'restart', 'acc', 'val_acc', 'elapsed_s', 'hw']
KEY = ('cfg', 'learner', 'instance', 'N', 'restart')
HW = torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'
INST = range(10, 20)


class CALPerm(CAL):
    def polish(self, T, w, xtr, ytr, xva, yva, best, iters=400):
        for it in range(iters):
            w = self.fit_w(self.feats(T, xtr), ytr, w)
            sc = self.scores(T, xtr, ytr, w).cpu().numpy()
            rows, cols = linear_sum_assignment(-sc)
            T = torch.tensor(cols, dtype=torch.long, device=DEV)[torch.argsort(torch.tensor(rows, device=DEV))]
            if it % 20 == 19 or it == iters - 1:
                w = self.fit_w(self.feats(T, xtr), ytr, w)
                va = self.acc(T, w, xva, yva)
                if va > best[0]:
                    best = (va, T.clone(), w.clone())
        return best

    def run_plus(self, xtr, ytr, xva, yva, seed):
        va, T, w = self.run(xtr, ytr, xva, yva, seed)
        return self.polish(T.clone(), w.clone(), xtr, ytr, xva, yva, (va, T, w))


def _checks():
    xs = np.random.default_rng(993).integers(0, 256, (20000, 8)).astype(np.uint8)
    puf = E13['E11']['ArbiterPUF'](n=64, seed=6)
    y = torch.tensor(E13['E11']['labels'](puf, E13['E11']['internal_bits']('id', xs)).astype(np.float32), device=DEV)
    xt = torch.tensor(xs.astype(np.int64), device=DEV)
    cal = CALPerm(suffix_mask())
    Ttrue = torch.tensor(E13['E11']['SBOX'].astype(np.int64), device=DEV)
    w = cal.fit_w(cal.feats(Ttrue, xt), y, torch.zeros(65, device=DEV), iters=8)
    sc = cal.scores(Ttrue, xt, y, w).cpu().numpy()
    rows, cols = linear_sum_assignment(-sc)
    T = torch.tensor(cols, dtype=torch.long, device=DEV)[torch.argsort(torch.tensor(rows, device=DEV))]
    assert float((T == Ttrue).float().mean()) > 0.9, 'assignment does not recover the true table from the true weights'
    assert len(set(T.tolist())) == 256, 'polish output is not a permutation'


_checks()


def done():
    if not os.path.exists(CSV):
        return set()
    return {tuple(r[k] for k in KEY) for r in csv.DictReader(open(CSV))}


def gate():
    if not os.path.exists(CSV):
        return False, 0
    g = {}
    for r in csv.DictReader(open(CSV)):
        if r['cfg'] == 'id' and r['N'] == '200000':
            g.setdefault(int(r['instance']), []).append(r)
    k = sum(float(max(v, key=lambda r: float(r['val_acc']))['acc']) >= 0.95 for v in g.values())
    return (len(g) == 10 and k >= 8), k


def main(part):
    have = done()
    new = not os.path.exists(CSV)
    f = open(CSV, 'a', newline='')
    w = csv.writer(f)
    if new:
        w.writerow(COLS)
    cfg, Ns = ('id', (100000, 200000)) if part == 'id' else ('lin', (200000, 500000))
    if part == 'lin':
        ok, k = gate()
        print(f'E16 id gate: {k}/10 held-out instances solved at 200k -> {"PASS" if ok else "FAIL (lin skipped)"}', flush=True)
        if not ok:
            f.close()
            return
    cal = CALPerm(suffix_mask() if cfg == 'id' else mask_for(L_LIN))
    for s in INST:
        for N in Ns:
            data = None
            for r in range(3):
                if (cfg, 'coord-ascent+perm', str(s), str(N), str(r)) in have:
                    continue
                data = data or make_data(cfg, s, N)
                xtr, ytr, xva, yva, xte, yte, _ = data
                t0 = time.time()
                va, T, wt = cal.run_plus(xtr, ytr, xva, yva, 1000 * s + r)
                acc = cal.acc(T, wt, xte, yte)
                row = ['E16', part, cfg, 'coord-ascent+perm', s, N, r, round(acc, 4), round(va, 4), round(time.time() - t0, 1), HW]
                w.writerow(row)
                f.flush()
                print(*row, flush=True)
    f.close()


if __name__ == '__main__':
    for part in sys.argv[1:] or ['id', 'lin']:
        main(part)
