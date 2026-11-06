"""E20: noise control. The noise study's known-table grid started at N = 1000, so the denominator of "cost of the secret table /
cost of the informed attack" was censored at the floor of the grid. Re-run ONLY the known-table control on a fine grid.
Written and pre-specified 2026-10-30 (revised the same day after an independent oracle review), before any E20 run (only a CPU
syntax check and a no-training data/shape check ran).

Pre-specification statement: this file must be committed to git before the first run; record the commit hash in the first CSV row
of the run. (The script writes the hash into a 'start' row at every invocation, marked DIRTY if the file differs from the commit.)
Any later edit to this file after the first E20 row exists voids the pre-specification and must be disclosed as a deviation.
Oracle run order: G0 checks, E20 FIRST (its 'gocheck' mode, then 'known'), then E18 known + K1 gate, E18 E11 arm, E19 known + Q1
gate, E19 secret, then the E18 coordinate-ascent pilot at 1M, then the E18 coordinate-ascent runs.
No learner is changed: this script exec's E9's file and reuses its data generator make_data(), its Model('known') and its fit()
(E7 structured model with the true table fixed, only the linear layer trained; Adam LR 0.02, batch 4096, >= 6000 steps / >= 20
epochs, BCE, 10% validation), unchanged.

Why. E9 noise mode measured the informed attack only at N in {1k, 5k, 20k, 100k}, so the known-table N-to-0.95 could not be
resolved below 1000 and the headline ratio (secret-table cost / informed cost) rested on a floor value. The 30-instance
noiseless known grid (E9 nscale + E15) has a median N-to-0.95 of 675 at n = 64. E20 gives the noisy known-table curve the same
resolution so that the denominator is measured, not censored.

Rig (identical to E9 noise): pypuf ArbiterPUF(n=64, seed=s, noisiness=nz) for the training labels, challenge rng 11+100s train /
12+100s test (10k test), AES S-box applied bytewise, labels (r+1)/2, 10% validation (last tenth of N, noisy labels), learner seed
1000*s + r. Success is scored on NOISELESS labels (acc_clean, the test set labelled by the noiseless twin ArbiterPUF with the
same seed), as in E9. The flip rate (fraction of test labels that differ between the noisy and the noiseless device) is logged.
Success numerator for noise rows: ABSOLUTE 0.95 on CLEAN labels, i.e. acc_clean >= 0.95 (as in the E21 re-analysis); never accuracy
against the noisy test labels and never relative to a noisy ceiling.
Instances 0-9, the same ten as the E9 noise study; no new instances, so no seed block needed. Restarts r = 0-2 (seeds
1000*s + r; E9 used r = 0 only for known).
OVERLAP WITH E9 (corrected after the oracle review; the earlier text overstated it). E9 noise mode has known-table rows only at
N in {1000, 5000, 20000, 100000} (all six nz, instances 0-9, restart 0); E9 nscale mode (n = 64, nz = 0 only, instances 0-9,
restart 0) has known rows at N in {100, 250, 500, 1000, 2000, 4000, 8000, 16000}. The E20 grid points that overlap an E9 known row
are therefore: nz = 0: N in {100, 250, 500, 1000, 2000, 4000, 8000, 16000} (nscale) and {1000, 5000} (noise); nz > 0: N in
{1000, 5000} only. (20000 and 100000 are not on the E20 grid.) Reproduction is up to GPU non-determinism, not bitwise.
Grid: noisiness nz in {0, 0.05, 0.1, 0.2, 0.3, 0.5}; N in {100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 8000, 16000,
32000} (100-32000; 4000 and 32000 added by the oracle review). Runs: 6 x 10 x 13 x 3 = 2340.

Modes (python ...-e20noisectl.py <mode> ...; append + resume):
  gocheck : GO-CHECK against E9, run FIRST (needs GPU). nz = 0, restart 0, instances 0-9, at the 9 distinct N that overlap E9
            (100, 250, 500, 1000, 2000, 4000, 5000, 8000, 16000; 90 runs, written to the CSV as mode 'gocheck'). Compares acc with
            the existing E9 rows in encoding-induced-false-security-pufs-e9scale-nscale.csv (n = 64, variant known, restart 0) and
            in encoding-induced-false-security-pufs-e9scale-noise.csv (nz = 0, variant known, N = 1000, 5000), reports the max
            absolute difference and PASSES if it is <= 0.01 (otherwise 'known' refuses to start). Differences can only come
            from GPU non-determinism since data, seeds and learner are identical.
  known   : the grid above (refuses to start unless the go-check passed).
  summary : (CPU, no training, no pypuf needed) per noise level: mean flip rate; mean acc_clean (over instances and restarts)
            at each N; per-instance N-to-0.95 by TWO estimators, each a log-N linear interpolation of acc_clean between the
            last grid N below 0.95 and the first grid N at or above 0.95 (read at the first crossing; reported as 100 = "<=100"
            if already >= 0.95 at N = 100, ">32000" censored if never reached; censored values count as infinity in the median):
              (A) restart-mean over the 3 restarts (primary; the paper will use the restart-mean over the 10 instances for the
                  noise table), and
              (B) restart 0 only (the estimator behind the 675 median of the 30-instance noiseless known grid, instances 0-29,
                  E9 nscale + E15; the paper cites both).
            Primary summary = median over the 10 instances. Also prints the pooled-curve crossing (mean over instances), the
            comparison with 675 (descriptive) and R1-R3. Levels with missing instances/restarts print 'incomplete'.

Predictions (before any E20 run; author's prior in brackets):
  R1 noise 0: the median known-table N-to-0.95 lies in [500, 800] (the 30-instance noiseless known grid gives 675).     [0.80]
  R2 the median N-to-0.95 is non-decreasing in noisiness across the six levels (a tie is allowed).                      [0.75]
  R3 at noisiness 0.5 the median N-to-0.95 lies in [2000, 4000].                                                        [0.60]
  (R1-R3 are evaluated on estimator A; estimator B is reported next to it.)
Claims by outcome (fixed now; the paper uses the matching wording):
  R1 holds                          -> the noiseless denominator is confirmed at the measured resolution and the E9 noise
                                       ratios are recomputed with the new denominators, not the floor.
  R1 fails (median outside 500-800) -> reconcile with the 675 figure before any ratio is quoted; report both.
  R2 fails                          -> report the non-monotone levels as measured (noise level, flip rate, N-to-0.95).
  R3 fails                          -> report the measured value; if censored (> 32000) the noisy ratio is a lower bound only.
  Any level censored                -> reported as censored; no extrapolation.
  gocheck fails (max diff > 0.01)   -> stop; the E20 rows would not be comparable with the E9 rows; debug first.
No learner, hyperparameter or grid is changed after seeing results; no instance is dropped.

Runtime estimate (E9 noise known rows on the RTX 4070 laptop GPU: median 11-25 s per run, dominated by the >= 6000 optimiser
steps, nearly independent of N): 2340 runs x about 15-25 s = about 10-16 h, plus 90 go-check runs (about 0.5 h). The hardware is
recorded in the hw column of every row.
Start-of-run checks (modes 'gocheck' and 'known' need a GPU; 'summary' does not): torch.cuda.is_available() is asserted, the
device name and pypuf version are printed (pypuf must be 2.2.0), and the lock file experiments/.lock-<script name> is created with
the PID (refuses to start if the lock exists and that PID is alive; removed on normal exit).
Commands: python experiments/encoding-induced-false-security-pufs-e20noisectl.py gocheck ; ... known ; ... summary
Writes experiments/encoding-induced-false-security-pufs-e20noisectl.csv (append + resume)
cols: test,mode,nz,variant,instance,N,restart,acc,acc_clean,val_acc,flip,elapsed_s,hw,note
"""
import csv, os, subprocess, sys, time, numpy as np

SCRIPT = 'encoding-induced-false-security-pufs-e20noisectl'
CSV = 'experiments/encoding-induced-false-security-pufs-e20noisectl.csv'
COLS = ['test', 'mode', 'nz', 'variant', 'instance', 'N', 'restart', 'acc', 'acc_clean', 'val_acc', 'flip', 'elapsed_s', 'hw', 'note']
KEY = ('mode', 'nz', 'variant', 'instance', 'N', 'restart')
NOISE = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
NS = (100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 8000, 16000, 32000)
INST = range(10)
RESTARTS = 3
PYPUF_PIN = '2.2.0'
E9_NSCALE = 'experiments/encoding-induced-false-security-pufs-e9scale-nscale.csv'
E9_NOISE = 'experiments/encoding-induced-false-security-pufs-e9scale-noise.csv'
GO_NS = (100, 250, 500, 1000, 2000, 4000, 5000, 8000, 16000)      # N with an E9 known row at nz = 0
GO_TOL = 0.01
HW = 'not loaded'


# ---------------------------------------------------------------- start-of-run checks and lock
def pid_alive(pid):
    if sys.platform == 'win32':       # os.kill(pid, 0) is CTRL_C_EVENT on Windows, so use tasklist
        out = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH', '/FO', 'CSV'], capture_output=True, text=True).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


LOCK = f'experiments/.lock-{SCRIPT}'


def acquire_lock():
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip())
        except ValueError:
            pid = -1
        if pid > 0 and pid_alive(pid):
            raise SystemExit(f'refusing to start: {LOCK} exists and PID {pid} is alive')
        print(f'stale lock {LOCK} (PID {pid} not alive); replacing', flush=True)
    open(LOCK, 'w').write(str(os.getpid()))


def release_lock():
    try:
        if open(LOCK).read().strip() == str(os.getpid()):
            os.remove(LOCK)
    except OSError:
        pass


def git_note(mode):
    h = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', f'experiments/{SCRIPT}.py'], capture_output=True,
                           text=True).stdout.strip() != ''
    return f'start mode={mode} commit={h}{" DIRTY (script differs from the commit)" if dirty else ""}'


def load():
    import torch
    from importlib.metadata import version
    global E9, torch_, DEV, HW
    torch_ = torch
    assert torch.cuda.is_available(), 'CUDA is required for the training modes (gocheck, known)'
    assert version('pypuf') == PYPUF_PIN, f'pypuf {version("pypuf")} != pinned {PYPUF_PIN}'
    print('device:', torch.cuda.get_device_name(0), '| pypuf', version('pypuf'), '| torch', torch.__version__, flush=True)
    E9 = {'__name__': 'e20_import'}
    exec(open('experiments/encoding-induced-false-security-pufs-e9scale.py').read(), E9)
    DEV = E9['DEV']
    HW = f'{torch.cuda.get_device_name(0)}; torch {torch.__version__}; pypuf {version("pypuf")}'
    _checks()


def _checks():
    """No training: shapes, split sizes, and that the noisy twin differs from the noiseless one at nz > 0 only."""
    xtr, ytr, xva, yva, xte, yte, yte_c, flip = E9['make_data']('noise', 'pypuf', 64, 0.0, 0, 1000)
    assert xtr.shape == (900, 8) and xva.shape == (100, 8) and xte.shape == (10000, 8) and ytr.shape == (900,)
    assert flip == 0.0 and bool((yte == yte_c).all()), 'nz = 0 must have no flips'
    *_, flip5 = E9['make_data']('noise', 'pypuf', 64, 0.5, 0, 1000)
    assert flip5 > 0.0, 'nz = 0.5 must flip labels'


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

    def _write(self, row):
        self.w.writerow(row)
        self.f.flush()
        print(*row, flush=True)

    def start(self, mode):
        self._write(['E20', 'start', '', '', '', '', '', '', '', '', '', 0, HW, git_note(mode)])

    def done(self, mode, nz, s, N, r):
        return (mode, str(nz), 'known', str(s), str(N), str(r)) in self.have

    def run_one(self, mode, nz, s, N, r, data):
        xtr, ytr, xva, yva, xte, yte, yte_c, flip = data
        t0 = time.time()
        m, va = E9['fit']('known', 1000 * s + r, 8, xtr, ytr, xva, yva)
        self._write(['E20', mode, nz, 'known', s, N, r, round(E9['acc'](m, xte, yte), 4), round(E9['acc'](m, xte, yte_c), 4),
                     round(va, 4), round(flip, 4), round(time.time() - t0, 1), HW, ''])


def e9_reference():
    """(N, instance) -> E9 acc at nz = 0, restart 0, known, n = 64; nscale and noise rows both listed (key includes the source)."""
    ref = {}
    for src, path in (('nscale', E9_NSCALE), ('noise', E9_NOISE)):
        for r in csv.DictReader(open(path)):
            if r['variant'] == 'known' and r['restart'] == '0' and float(r['nz']) == 0.0 and r['n'] == '64':
                ref[(src, int(r['N']), int(r['instance']))] = float(r['acc'])
    return ref


def go_status(rows=None):
    """Max abs difference between the E20 gocheck rows and the E9 rows, and the number of compared pairs. None if incomplete."""
    rows = [r for r in csv.DictReader(open(CSV))] if rows is None and os.path.exists(CSV) else (rows or [])
    mine = {(int(r['N']), int(r['instance'])): float(r['acc']) for r in rows if r['mode'] == 'gocheck'}
    ref = e9_reference()
    diffs = []
    for (src, N, s), a in ref.items():
        if N in GO_NS and (src == 'nscale' or N in (1000, 5000)):
            if (N, s) not in mine:
                return None, 0
            diffs.append(abs(mine[(N, s)] - a))
    return (max(diffs) if diffs else None), len(diffs)


def mode_gocheck(W):
    for s in INST:
        for N in GO_NS:
            if W.done('gocheck', 0.0, s, N, 0):
                continue
            W.run_one('gocheck', 0.0, s, N, 0, E9['make_data']('noise', 'pypuf', 64, 0.0, s, N))
    mx, n = go_status()
    ok = mx is not None and mx <= GO_TOL
    print(f'GO-CHECK vs E9 (nz = 0, restart 0): {n} pairs, max abs difference {mx}, tolerance {GO_TOL} ->', 'PASS' if ok else 'FAIL', flush=True)
    if not ok:
        raise SystemExit('go-check failed: do not run the known mode')


def mode_known(W):
    mx, n = go_status()
    if mx is None or mx > GO_TOL:
        raise SystemExit(f'go-check not passed (max diff {mx}, {n} pairs): run the gocheck mode first')
    for nz in NOISE:
        for s in INST:
            for N in NS:
                data = None
                for r in range(RESTARTS):
                    if W.done('known', nz, s, N, r):
                        continue
                    data = data or E9['make_data']('noise', 'pypuf', 64, nz, s, N)
                    W.run_one('known', nz, s, N, r, data)


# ---------------------------------------------------------------- summary (no training, no heavy imports)
def n_to_95(ns, acc):
    """Log-N interpolated first crossing of 0.95; (value, tag) with tag in '', '<=', '>' (censored)."""
    if acc[0] >= 0.95:
        return ns[0], '<='
    for i in range(1, len(ns)):
        if acc[i] >= 0.95:
            t = (0.95 - acc[i - 1]) / (acc[i] - acc[i - 1])
            return float(np.exp(np.log(ns[i - 1]) + t * (np.log(ns[i]) - np.log(ns[i - 1])))), ''
    return float('inf'), '>'


def mode_summary():
    rows = [r for r in csv.DictReader(open(CSV)) if r['mode'] == 'known']
    mx, n = go_status()
    print('go-check vs E9:', 'not run / incomplete' if mx is None else f'{n} pairs, max abs diff {mx:.4f} -> {"PASS" if mx <= GO_TOL else "FAIL"}')
    medA, medB = {}, {}
    fmt = lambda x: round(x) if np.isfinite(x) else '>32000'
    for nz in NOISE:
        R = [r for r in rows if float(r['nz']) == nz]
        if not R:
            print(f'nz={nz}: no rows')
            continue
        acc = {}                                          # (instance, N) -> {restart: acc_clean}
        for r in R:
            acc.setdefault((int(r['instance']), int(r['N'])), {})[int(r['restart'])] = float(r['acc_clean'])
        full = lambda s: all(len(acc.get((s, N), {})) == RESTARTS for N in NS)
        if not all(full(s) for s in INST):
            print(f'nz={nz}: incomplete ({sum(full(s) for s in INST)}/10 instances have the full grid x 3 restarts)')
            continue
        flip = float(np.mean([float(r['flip']) for r in R]))
        perA = [n_to_95(list(NS), [float(np.mean(list(acc[(s, N)].values()))) for N in NS])[0] for s in INST]   # (A) restart-mean
        perB = [n_to_95(list(NS), [acc[(s, N)][0] for N in NS])[0] for s in INST]                              # (B) restart 0 only
        curve = [float(np.mean([np.mean(list(acc[(s, N)].values())) for s in INST])) for N in NS]
        medA[nz], medB[nz] = float(np.median(perA)), float(np.median(perB))
        print(f'nz={nz}: flip {flip:.3f}; median N95 (A, restart-mean) {fmt(medA[nz])}, (B, restart 0) {fmt(medB[nz])}; '
              f'pooled-curve crossing {n_to_95(list(NS), curve)[0]:.0f}')
        print('   per-instance A:', [fmt(x) for x in perA])
        print('   per-instance B:', [fmt(x) for x in perB])
        print('   mean acc_clean by N:', {N: round(a, 3) for N, a in zip(NS, curve)})
    msA = [medA.get(nz) for nz in NOISE]
    print('comparison: 30-instance noiseless known grid median N-to-0.95 (restart 0, instances 0-29) = 675')
    print('R1 noise-0 median in [500, 800] (A):', 'incomplete' if medA.get(0.0) is None else 500 <= medA[0.0] <= 800,
          '| (B):', 'incomplete' if medB.get(0.0) is None else 500 <= medB[0.0] <= 800)
    print('R2 median non-decreasing in noisiness (A):', 'incomplete' if None in msA else all(a <= b for a, b in zip(msA[:-1], msA[1:])))
    print('R3 noise-0.5 median in [2000, 4000] (A):', 'incomplete' if medA.get(0.5) is None else 2000 <= medA[0.5] <= 4000)


MODES = {'gocheck': mode_gocheck, 'known': mode_known}

if __name__ == '__main__':
    modes = sys.argv[1:]
    assert modes, 'usage: e20noisectl.py gocheck|known|summary ...'
    assert all(m in MODES or m == 'summary' for m in modes), f'unknown mode in {modes}'
    if modes == ['summary']:
        mode_summary()
    else:
        load()                      # asserts CUDA, pypuf 2.2.0; prints device
        acquire_lock()
        try:
            W = Writer()
            for mode in modes:
                if mode == 'summary':
                    mode_summary()
                    continue
                W.start(mode)
                MODES[mode](W)
            W.f.close()
        finally:
            release_lock()
