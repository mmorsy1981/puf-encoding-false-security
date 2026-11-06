"""E17: one last, bounded attempt at the linear-mixing question with the coordinate-ascent learner.
Written and preregistered 2026-10-25, before any E17 run (only the import-time self-checks, which train nothing, ran).

Why. E13 and E16 left the public GF(2)-linear layer after a secret S-box untested for new learners, because the
coordinate-ascent learner (cal) solved 7/10 identity-layer instances at 200k twice (E13 instances 0-9, E16 instances 10-19,
best-of-3, per-restart rate 0.33) and the preregistered gate needs 8/10. Its restarts end at 0.93-0.95 or at 1.000, and a
permutation polish (E16) did not change the rate. E17 tests the two cheapest remaining levers and nothing else:
more iterations and more restarts.

Learner: E13's cal, unchanged except the iteration count (2400 instead of 1200, i.e. twice the budget) and the number of
restarts per instance (best-of-5 instead of best-of-3). Same loss, ridge, block size (16), random initial table, seeds
1000*s + r (r = 0-4), validation-selected best iterate. Fresh held-out instances 20-29 (new PUF seeds, new data rngs
11+100s / 12+100s). GPU (CUDA, TF32 off), one process, every restart logged.

Parts (python ...-e17calmore.py <part>):
  id  : identity layer ('id'), N = 200k, instances 20-29, restarts 0-4. Gate = best-of-5 validation-selected >= 0.95 on
        >= 8/10 instances. 50 runs.
  lin : linear layer ('lin', public L, rng 6000), N in {200k, 500k}, instances 20-29, restarts 0-2 (best-of-3).
        Runs ONLY IF the id gate passed (otherwise prints the gate and exits). 60 runs.

Predictions (before any E17 run; author's prior in brackets):
  G3 cal passes the id gate with twice the iterations and best-of-5.                                          [0.35]
  M3 the per-restart rate of reaching 0.95 at id/200k is at least 0.50 (E13 and E16: 0.33).                  [0.35]
  L3 given G3, cal solves >= 5/10 lin instances (best-of-3, 0.95) at 500k.                                    [0.30]
Decision table and stop rule (fixed now). Gate failed -> the linear-layer question stays open, the paper keeps its wording
("resisted the one learner that passes the identity gate; four further learner families failed it"), and NO further learner
variant is tried. Gate passed and lin solved -> the secret-S-box result extends to public GF(2)-linear mixing at n = 64 and
the abstract and boundary section change. Gate passed and lin unsolved -> a learner bound with a verified second learner.
Writes experiments/encoding-induced-false-security-pufs-e17calmore.csv (append + resume)
cols: test,part,cfg,learner,instance,N,restart,acc,val_acc,elapsed_s,hw
"""
import csv, os, sys, time, torch

E13 = {'__name__': 'e17_import'}
exec(open('experiments/encoding-induced-false-security-pufs-e13learners.py').read(), E13)
CAL, DEV, make_data = E13['CAL'], E13['DEV'], E13['make_data']
suffix_mask, mask_for, L_LIN = E13['suffix_mask'], E13['mask_for'], E13['L_LIN']
CSV = 'experiments/encoding-induced-false-security-pufs-e17calmore.csv'
COLS = ['test', 'part', 'cfg', 'learner', 'instance', 'N', 'restart', 'acc', 'val_acc', 'elapsed_s', 'hw']
KEY = ('cfg', 'learner', 'instance', 'N', 'restart')
HW = torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'
INST = range(20, 30)
ITERS = 2400


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
    if part == 'lin':
        ok, k = gate()
        print(f'E17 id gate: {k}/10 held-out instances solved at 200k -> {"PASS" if ok else "FAIL (lin skipped, stop rule applies)"}', flush=True)
        if not ok:
            f.close()
            return
    cfg, Ns, R = ('id', (200000,), 5) if part == 'id' else ('lin', (200000, 500000), 3)
    cal = CAL(suffix_mask() if cfg == 'id' else mask_for(L_LIN))
    for s in INST:
        for N in Ns:
            data = None
            for r in range(R):
                if (cfg, 'coord-ascent-2x', str(s), str(N), str(r)) in have:
                    continue
                data = data or make_data(cfg, s, N)
                xtr, ytr, xva, yva, xte, yte, _ = data
                t0 = time.time()
                va, T, wt = cal.run(xtr, ytr, xva, yva, 1000 * s + r, iters=ITERS)
                row = ['E17', part, cfg, 'coord-ascent-2x', s, N, r, round(cal.acc(T, wt, xte, yte), 4), round(va, 4), round(time.time() - t0, 1), HW]
                w.writerow(row)
                f.flush()
                print(*row, flush=True)
    f.close()


if __name__ == '__main__':
    for part in sys.argv[1:] or ['id', 'lin']:
        main(part)
