"""E15: 30-instance extension of the E9 scaling grid (answers: headline cells rest on 10 instances, and the
first ten instances proved easier than the next twenty in E11-n30).
Written and preregistered 2026-10-24, before any E15 run.

Design: E9's nscale rig, learner, grids, seeds and success criterion UNCHANGED (this script execs the E9 file and
subclasses its Runner), instances 10-29 (PUF seed s, data rngs 11+100s and 12+100s, restart seed 1000*s + r), 3 restarts.
Differences, fixed now: the extension rule of E9 is switched off (fixed grids, so every instance has every N and the
cluster matrix is rectangular). Known-table runs for instances 10-29 use E9's known grid.
Analysis (article/make_figs_hw.py, automatic once the CSV exists): instances 0-29 pooled, N50 and cost ratios with the
instance-level bootstrap of article/stats_cluster.py.
GPU (CUDA, TF32 off), init and batch order drawn on the CPU as in E7gpu; one process; every restart logged.

Predictions (before any E15 run; author's prior in brackets):
  C1 Pooled 30-instance N50 lies inside the E9 10-instance bootstrap interval for every (variant, n) cell
     (shared 12.5k/33.9k/106.5k, per-byte 15.4k/76.3k/382.7k at n = 32/64/128).                          [0.5]
  C2 The growth with length survives: for both variants, the 30-instance N50 / known-table-N95 ratio at n = 128 exceeds
     the ratio at n = 32 by at least a factor of 1.5, with the bootstrap lower bound of that ratio of ratios above 1.   [0.85]
  C3 First-ten optimism repeats: at shared n = 64, N = 30k, the per-restart success of the 20 new instances is below
     0.40 (first ten: 0.40).                                                                              [0.6]
Writes experiments/encoding-induced-false-security-pufs-e15scale30.csv (append + resume; E9 column layout)
"""
import sys

G = {'__name__': 'e15_import'}
exec(open('experiments/encoding-induced-false-security-pufs-e9scale.py').read(), G)
Runner = G['Runner']
CSV = 'experiments/encoding-induced-false-security-pufs-e15scale30.csv'
NEW = range(10, 30)


class R30(Runner):
    def cell(self, mode, model, n, nz, variant, N, restarts, instances=NEW):
        return super().cell(mode, model, n, nz, variant, N, restarts, instances)


def main():
    R = R30(CSV)
    known = (100, 250, 500, 1000, 2000, 4000, 8000, 16000)
    shared = {32: (5000, 10000, 20000, 30000, 60000), 64: (10000, 20000, 30000, 40000, 60000),
              128: (10000, 20000, 40000, 60000, 100000, 200000)}
    perbyte = {32: (10000, 20000, 40000, 60000, 100000), 64: (40000, 60000, 100000, 150000, 200000),
               128: (100000, 200000, 300000, 400000, 800000)}
    for n in (32, 64, 128):
        for N in known:
            R.cell('nscale', 'pypuf', n, 0.0, 'known', N, 1)
    for n in (32, 64, 128):
        R.sweep('nscale', 'pypuf', n, 0.0, 'shared', shared[n], False)
    for n in (32, 64, 128):
        R.sweep('nscale', 'pypuf', n, 0.0, 'perbyte', perbyte[n], False)
    R.f.close()


if __name__ == '__main__':
    main()
