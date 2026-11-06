"""E10roc: noise sweep for the ROC-SPUF EM-gated break (extends the single nz=0.1 point).
Written and preregistered 2026-10-20, before any run. Reuses pilot-e3grid.py's maps, data generator
and t2_em() unchanged (exec of the file up to its run loop); scheme 3-3, odd convention (all 8
scheme x convention cells agree within ~0.2pp in E3grid), puf seed 400+inst, seed s=0 data rngs
(2000+100*inst, 3000+100*inst), instances 0-9. CPU (sklearn liblinear).
noisiness in {0, 0.05, 0.1, 0.2, 0.3}; N in {20k, 50k}.
Ceiling per cell: accuracy of the noiseless composite (same weights, same gate) against the noisy
test labels, i.e. the best any model of the device can do on those labels.
Predictions:
  R1 at N=20k, gated accuracy is within 3pp of the ceiling on >=8/10 instances for nz <= 0.1 and
     within 5pp for nz = 0.2; N=50k narrows each gap.
  R2 the break never falls to the 0.75 random-selector cap for nz <= 0.2 (gated > 0.85 on 10/10).
  nz = 0.3 exploratory.
Rig check: (nz 0 and 0.1, inst 0, N 20k) must reproduce E3grid's 3-3-odd seed-0 rows exactly
(gated/ungated 0.9633/0.7554 and 0.931/0.752).
Writes experiments/encoding-induced-false-security-pufs-e10rocnoise.csv (append + resume)
cols: test,scheme,conv,nz,instance,N,acc_gat,acc_ung,ceiling,flip_gate,elapsed_s
"""
import csv, os, time, numpy as np
from pypuf.simulation import ArbiterPUF

src = open('experiments/encoding-induced-false-security-pufs-pilot-e3grid.py').read()
E3 = {'__name__': 'e10roc_import'}
exec(src.split('\nt0 = time.time()')[0], E3)
gen, t2_em, MAPS, labels = E3['gen'], E3['t2_em'], E3['MAPS'], E3['labels']
CSV = 'experiments/encoding-induced-false-security-pufs-e10rocnoise.csv'
SCHEME, CONV = '3-3', 'odd'


def done():
    if not os.path.exists(CSV):
        return set()
    return {(float(r['nz']), int(r['instance']), int(r['N'])) for r in csv.DictReader(open(CSV))}


if __name__ == '__main__':
    have = done()
    new = not os.path.exists(CSV)
    with open(CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['test', 'scheme', 'conv', 'nz', 'instance', 'N', 'acc_gat', 'acc_ung',
                        'ceiling', 'flip_gate', 'elapsed_s'])
        for nz in (0.0, 0.05, 0.1, 0.2, 0.3):
            for inst in range(10):
                for N in (20000, 50000):
                    if (nz, inst, N) in have:
                        continue
                    t0 = time.time()
                    puf = ArbiterPUF(n=64, seed=400 + inst, noisiness=nz)
                    clean = ArbiterPUF(n=64, seed=400 + inst, noisiness=0.0)
                    Ctr, ytr = gen(puf, N, np.random.default_rng(2000 + 100 * inst), SCHEME, CONV)
                    Cte, yte = gen(puf, 10000, np.random.default_rng(3000 + 100 * inst), SCHEME, CONV)
                    # noiseless composite on the same test challenges (gate and branch both clean)
                    rs_clean = labels(clean, Cte)
                    y_clean = labels(clean, MAPS[SCHEME](Cte, rs_clean, CONV))
                    ceiling = float((y_clean == yte).mean())
                    flip_gate = float((labels(puf, Cte) != rs_clean).mean())
                    u, g = t2_em(MAPS[SCHEME], CONV, Ctr, ytr, Cte, yte)
                    row = ['E10roc', SCHEME, CONV, nz, inst, N, g, u, round(ceiling, 4),
                           round(flip_gate, 4), round(time.time() - t0, 1)]
                    w.writerow(row)
                    f.flush()
                    print(*row, flush=True)
