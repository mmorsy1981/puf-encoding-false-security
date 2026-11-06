r"""E22 — mixing ladder at 3.2M and 10M CRPs (pre-specification DRAFT v2, 2026-11-06)

Status: v2 APPROVED by the author 2026-11-06, after the oracle design gate (PASS with required changes, applied;
experiments/logs/e22-design-gate.md). Nothing has been run. Once approved, this text becomes the docstring of
`experiments/encoding-induced-false-security-pufs-e22mixscale.py`, and the author commits it before the first run.

## Question
E18 found the E11 mask-table learner at chance (≤0.515) on repeated w=4 and w=8 mixing up to 1M CRPs, and only partly learning
w=2 at 1M (best-of-5 6/10). At w=2 the first successes appeared at the top of the grid, so the w≥4 failures are censored by
the data budget (oracle O5). E22 asks: **does the same frozen learner learn w=4 (and w=8) with more data, and does w=2 reach the
cost criterion?** Under the frozen fit rule the number of training steps grows with N, so every claim is about data and steps
growing together (28 epochs / 6,160 steps at 1M; 20 epochs / 14,080 steps at 3.2M; 20 epochs / 43,960 steps at 10M).

## Design (frozen learner, no new variant: consistent with the E17 stop rule, as in E18)
- Layers: E18's repeated-block layers mix2, mix4, mix8 (mix8 = E11 L_LIN), loaded through E18's own code, unchanged.
  No independent arm, no w=1 (already solved at 200k).
- Learner: E11 mask-table learner (`MaskModel('shared')`, E11 `fit()`, Adam LR 0.02, batch 4096, ≥6000 steps / ≥20 epochs,
  BCE, 10% validation), exec'd unchanged. Coordinate ascent is NOT run (≈1000 s per run at 1M, ≈44 GPU-h at 3.2M; out of scope).
- Instances: 30–39, the E18 block (paired with E18 on the N axis; fixed before E18, nothing selected by outcome; the test set
  (rng 12+100s, 10k) is identical at every N).
- Restarts: 0–4, seeds 1000*s + r (E18 convention). Primary: validation-selected best-of-5; best-of-3 reported next to it.
- Grid: N = 3,200,000 for rep2, rep4, rep8 (150 runs); N = 10,000,000 for rep4 only (50 runs). 200 runs in total.
  Data path: E11 `make_data` (challenge rng 11+100s train / 12+100s test, 10k test, validation = last tenth of N).
- Known-table control: not rerun; E18's K1 gate (10/10 at 10k on every rung, these instances) and its known N-to-0.95 medians
  (rep2 670, rep4 652, rep8 673) are the controls and denominators.
- Success = test accuracy ≥ 0.95 (primary; 0.90 secondary) of the validation-selected restart.

## Go-checks (before the main grid)
- G1 reproducibility: re-run E18 e11 rep2 instance 38 restart 3 at N = 1M through the E22 code path (E18 test accuracy 0.9962);
  require ≥ 0.95 and |Δ| ≤ 0.005. Assert that the first 1M rows of the 3.2M (and 10M) training draw and both test arrays equal
  E18's exactly. Failure → stop and debug.
- G2 timing pilot: one run at 10M on rep4, instance 60, seed 777000 (not a grid seed); log elapsed time, peak GPU memory and
  CPU RAM; accuracy left blank (not a result). Replaces the runtime estimate below before the grid starts.

## Predictions (author's priors in brackets; approved by the author 2026-11-06, R1 raised from 0.55 to 0.70 by the author)
- R1 rep2 at 3.2M: best-of-5 solves ≥ 8/10 (w=2 reaches the cost criterion).                                   [0.70]
- R2 rep4 at 3.2M: best-of-5 solves 0/10 and every selected run stays ≤ 0.55 test accuracy.                    [0.65]
- R3 rep8 at 3.2M: best-of-5 solves 0/10 (replicates E11/E18 at 3.2× the data).                                [0.85]
- R4 monotone: solved count at 3.2M non-increasing in w (rep2 ≥ rep4 ≥ rep8).                                   [0.85]
- R5 rep4 at 10M: best-of-5 solves 0/10 and every selected run stays ≤ 0.55 test accuracy.                     [0.50]
The 0.55 band is new (E18 used 0.95/0.90 only): chance has a standard error of about 0.005 on 10k test rows, and E18's rep4
maximum was 0.515.

## Claims by outcome (fixed now; the paper uses the matching wording)
rep4 (judged at the largest N with a success, else at 10M):
- ≥ 8/10 at some N   → four-byte mixing is learnable by this learner at that N (data and steps growing together); informed-cost
                        ratio N / 652 (grid-bounded upper bound); thesis scope extends to w ≤ 4; abstract, discussion and
                        limitations are rewritten; w=8 status as measured.
- 1–7/10 at some N   → four-byte mixing is partly learned at that N (counts reported), no cost; "open" becomes "partly learned".
- 0/10, all ≤ 0.55   → the learner bound moves from 1M to 10M CRPs; w ≥ 4 stays open, not secure (no change in claim type).
- 0/10, some selected run in (0.55, 0.95) → reported as measured: above chance but unsolved; no claim beyond the counts.
rep2 at 3.2M:
- ≥ 8/10             → w=2 cost = 3.2M / 670 (≈ 4800×, single grid point, upper bound).
- < 8/10             → two-byte mixing partly learned at 3.2M (counts reported), no cost; a count below E18's 6/10 at 1M is
                        reported as measured.
rep8 at 3.2M:
- 0/10               → the learner bound on the full linear layer moves to 3.2M CRPs.
- any success        → reported as measured; it contradicts the E11/E18 record on the same layer at lower N and is checked against G1.
Any restart that reaches 0.95 but is not the validation-selected one → reported as exploratory, not counted.
Single-seed instability or a non-monotone count is reported as measured; no instance is dropped; nothing is retuned.

## Kill criteria
- K1: G1 fails → stop (code path not comparable with E18).
- K2: G2 shows the 10M run cannot fit in GPU memory or CPU RAM without changing the learner → drop the 10M point, report it, run
  the 3.2M grid; no learner change.
- No other stopping rule: all runs are reported.

## Environment and device
GPU: NVIDIA RTX 4070 Laptop (8 GB), `puf-clean` conda env, torch 2.8.0+cu129, pypuf 2.2.0 (pinned, unchanged). TF32 off,
CPU-side init/batch order as in E18. The feature tensor is built per batch and accuracy is computed in 20k-row chunks (E11 code),
so GPU memory is not expected to limit; the 10M training tensor is about 640 MB of int64.

## Runtime estimate (inferred from E18 timings and the fit rule, not measured)
E18 e11 at 1M: median 17 s per run (6,160 steps). 3.2M: 14,080 steps (2.29×) ≈ 40 s per run, 150 runs ≈ 1.7 h. 10M: 43,960 steps
(7.1×) ≈ 125 s per run, 50 runs ≈ 1.7 h. Total ≈ 3.5 h plus label generation (not included in E18's elapsed_s). G2 replaces this.
Long-run protocol: detached, progress file + exit-code file in experiments/logs/.

## Analysis plan
Summary per rung and N: solved counts (0.95 primary, 0.90 secondary) for best-of-3 and best-of-5; max and median selected test
accuracy; R1–R5; per-instance table paired with E18's 60k/200k/1M results. Anything else is labelled exploratory.

## Smallest effect that would matter
Any rep4 success (≥1/10 at ≥0.95) changes the paper's w=4 wording from "open (at chance to 1M)" to "partly learned".
"""
# Usage (from any directory; the script chdirs to the repository root):
#   python experiments/encoding-induced-false-security-pufs-e22mixscale.py g1 | g2 | grid | summary   (modes may be chained)
# Layers, learner and data path are reused unchanged from E18 (imported as a module, E18's own load()), which itself execs E13/E11.
import csv, importlib.util, os, subprocess, sys, time, statistics
import numpy as np

SCRIPT = 'encoding-induced-false-security-pufs-e22mixscale'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CSV = f'experiments/{SCRIPT}.csv'
E18_CSV = 'experiments/encoding-induced-false-security-pufs-e18mixladder.csv'
E18_FILE = 'experiments/encoding-induced-false-security-pufs-e18mixladder.py'
COLS = ['test', 'mode', 'arm', 'w', 'learner', 'instance', 'N', 'restart', 'acc', 'val_acc', 'ncols', 'peak_gpu_mb',
        'cpu_rss_mb', 'elapsed_s', 'hw', 'note']
KEY = ('mode', 'arm', 'w', 'learner', 'instance', 'N', 'restart')
LEARNER = 'e11-mask'
INST = range(30, 40)
RESTARTS = 5
GRID = ((2, 3_200_000), (4, 3_200_000), (8, 3_200_000), (4, 10_000_000))      # (w, N), arm 'rep'
G1 = dict(w=2, inst=38, r=3, N=1_000_000, e18_expect=0.9962)
G2 = dict(w=4, inst=60, seed=777000, N=10_000_000)
E18_NS = (60000, 200000, 1000000)
LOCK = f'experiments/.lock-{SCRIPT}'


def cfg_of(w):
    return f'mix{w}'


# ---------------------------------------------------------------- E18 reuse (no logic copied)
def import_e18():
    """Import E18 as a module (its __main__ block does not run); module level is pure numpy (LAYER, NCOLS, lock helpers)."""
    spec = importlib.util.spec_from_file_location('e18mixladder', E18_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['e18mixladder'] = mod
    spec.loader.exec_module(mod)
    return mod


def git_note(mode):
    h = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', f'experiments/{SCRIPT}.py'], capture_output=True,
                           text=True).stdout.strip() != ''
    return f'start mode={mode} commit={h}{" DIRTY (script differs from the commit)" if dirty else ""}'


def acquire_lock(e18):
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip())
        except ValueError:
            pid = -1
        if pid > 0 and e18.pid_alive(pid):
            raise SystemExit(f'refusing to start: {LOCK} exists and PID {pid} is alive')
        print(f'stale lock {LOCK} (PID {pid} not alive); replacing', flush=True)
    open(LOCK, 'w').write(str(os.getpid()))


def release_lock():
    try:
        if open(LOCK).read().strip() == str(os.getpid()):
            os.remove(LOCK)
    except OSError:
        pass


def rss_peak_mb():
    import psutil
    mi = psutil.Process().memory_info()
    return round(getattr(mi, 'peak_wset', mi.rss) / 2 ** 20, 1)


class Writer:
    def __init__(self, hw, ncols):
        self.hw, self.ncols = hw, ncols
        self.have = set()
        if os.path.exists(CSV):
            self.have = {tuple(r[k] for k in KEY) for r in csv.DictReader(open(CSV))}
        new = not os.path.exists(CSV)
        self.f = open(CSV, 'a', newline='')
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(COLS)
            self.f.flush()

    def done(self, *key):
        return tuple(map(str, key)) in self.have

    def _write(self, row):
        self.w.writerow(row)
        self.f.flush()
        print(*row, flush=True)

    def start(self, mode):
        self._write(['E22', 'start', '', '', '', '', '', '', '', '', '', '', '', 0, self.hw, git_note(mode)])

    def row(self, mode, w, s, N, r, acc, va, t0, peak='', rss='', note=''):
        rnd = lambda v: '' if v is None else round(v, 4)
        self._write(['E22', mode, 'rep', w, LEARNER, s, N, r, rnd(acc), rnd(va), self.ncols[cfg_of(w)], peak, rss,
                     round(time.time() - t0, 1), self.hw, note])
        self.have.add(tuple(map(str, (mode, 'rep', w, LEARNER, s, N, r))))


def read_rows(path=CSV):
    return [r for r in csv.DictReader(open(path))] if os.path.exists(path) else []


# ---------------------------------------------------------------- go-checks
def data_equality(e18, s, w):
    """G1 data assertion (data generation only): the first 1M rows of the training challenge draw at N = 3.2M and 10M, and both
    test arrays, equal the N = 1M draw exactly. Labels of the 3.2M draw's first 1M rows are checked as well."""
    cfg = cfg_of(w)
    base = e18.make_data(cfg, s, G1['N'])
    xb, yb, xteb, yteb = base[6]
    for N in (3_200_000, 10_000_000):
        xr = np.random.default_rng(11 + 100 * s).integers(0, 256, (N, 8)).astype(np.uint8)
        assert np.array_equal(xr[:G1['N']], xb), f'training challenge prefix differs at N={N}'
        del xr
    d32 = e18.make_data(cfg, s, 3_200_000)
    x32, y32, xte32, yte32 = d32[6]
    assert np.array_equal(x32[:G1['N']], xb), 'make_data challenge prefix differs at 3.2M'
    assert np.array_equal(y32[:G1['N']], yb), 'make_data label prefix differs at 3.2M'
    assert np.array_equal(xte32, xteb) and np.array_equal(yte32, yteb), 'test arrays differ at 3.2M'
    assert np.array_equal(d32[4].cpu().numpy(), base[4].cpu().numpy()) and np.array_equal(d32[5].cpu().numpy(),
                                                                                         base[5].cpu().numpy())
    print('G1 data equality (instance 38): 3.2M and 10M challenge prefixes, 3.2M label prefix, both test arrays: OK', flush=True)


def mode_g1(W, e18):
    if W.done('g1', 'rep', G1['w'], LEARNER, G1['inst'], G1['N'], G1['r']):
        print('g1 already done', flush=True)
        return
    t0 = time.time()
    try:
        data_equality(e18, G1['inst'], G1['w'])
        eq = 'data-equal=OK'
    except AssertionError as e:
        W.row('g1', G1['w'], G1['inst'], G1['N'], G1['r'], None, None, t0, note=f'FAIL data equality: {e}')
        return
    ref = [r for r in read_rows(E18_CSV) if (r['mode'], r['arm'], r['w'], r['instance'], r['N'], r['restart']) ==
           ('e11', 'rep', str(G1['w']), str(G1['inst']), str(G1['N']), str(G1['r']))]
    assert len(ref) == 1, 'E18 reference row not found'
    e18_acc = float(ref[0]['acc'])
    torch = e18.torch_
    cfg, s, r, N = cfg_of(G1['w']), G1['inst'], G1['r'], G1['N']
    t0 = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    data = e18.make_data(cfg, s, N)
    xtr, ytr, xva, yva, xte, yte, _ = data
    m, gen = e18.build('mask', 'shared', 1000 * s + r, A=e18.mask_for(e18.LAYER[cfg]))
    m, va = e18.fit(m, gen, xtr, ytr, xva, yva)
    acc = e18.accm(m, xte, yte)
    delta = acc - e18_acc
    ok = acc >= 0.95 and abs(delta) <= 0.005
    W.row('g1', G1['w'], s, N, r, acc, va, t0, peak=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), rss=rss_peak_mb(),
          note=f'{"PASS" if ok else "FAIL"} e18={e18_acc} e22={round(acc, 4)} delta={round(delta, 4)} {eq}')
    if not ok:
        print('K1 (G1 failed): stop and debug', flush=True)


def mode_g2(W, e18):
    s, N, w, seed = G2['inst'], G2['N'], G2['w'], G2['seed']
    if W.done('g2', 'rep', w, LEARNER, s, N, 0):
        print('g2 already done', flush=True)
        return
    torch = e18.torch_
    cfg, t0 = cfg_of(w), time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        data = e18.make_data(cfg, s, N)
        xtr, ytr, xva, yva, xte, yte, _ = data
        m, gen = e18.build('mask', 'shared', seed, A=e18.mask_for(e18.LAYER[cfg]))
        m, va = e18.fit(m, gen, xtr, ytr, xva, yva)
        e18.accm(m, xte, yte)                         # evaluated for the memory/time footprint only; not logged
        note = f'timing pilot seed={seed}; acc/val_acc intentionally blank (not a result)'
    except (torch.cuda.OutOfMemoryError, MemoryError) as e:
        note = f'FAILED (K2: drop the 10M point, report): {type(e).__name__}'
    W.row('g2', w, s, N, 0, None, None, t0, peak=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), rss=rss_peak_mb(),
          note=note)


def require_go(W):
    rows = read_rows()
    g1 = [r for r in rows if r['mode'] == 'g1' and r['note'].startswith('PASS')]
    g2 = [r for r in rows if r['mode'] == 'g2']
    if not g1 or not g2:
        raise SystemExit(f'refusing to start the grid: need a g1 PASS row (found {len(g1)}) and a g2 row (found {len(g2)})')
    k2 = any('FAILED' in r['note'] for r in g2) and not any('FAILED' not in r['note'] for r in g2)
    print('go-checks: g1 PASS, g2 present', '(g2 FAILED: K2, 10M point dropped)' if k2 else '', flush=True)
    return k2


def mode_grid(W, e18):
    drop10 = require_go(W)
    torch = e18.torch_
    for w, N in GRID:
        if N == 10_000_000 and drop10:
            print('K2: skipping the 10M rung', flush=True)
            continue
        cfg = cfg_of(w)
        A = e18.mask_for(e18.LAYER[cfg])
        print(f'{cfg} N={N}: {e18.NCOLS[cfg]} columns', flush=True)
        for s in INST:
            data = None
            for r in range(RESTARTS):
                if W.done('grid', 'rep', w, LEARNER, s, N, r):
                    continue
                data = data or e18.make_data(cfg, s, N)
                xtr, ytr, xva, yva, xte, yte, _ = data
                t0 = time.time()
                torch.cuda.reset_peak_memory_stats()
                m, gen = e18.build('mask', 'shared', 1000 * s + r, A=A)
                m, va = e18.fit(m, gen, xtr, ytr, xva, yva)
                W.row('grid', w, s, N, r, e18.accm(m, xte, yte), va, t0,
                      peak=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), rss=rss_peak_mb())
            del data
            torch.cuda.empty_cache()
            open(f'experiments/logs/{SCRIPT}.progress', 'w').write(
                f'{len(read_rows())} rows written, last {time.strftime("%Y-%m-%d %H:%M:%S")} (grid target 200 runs)\n')


# ---------------------------------------------------------------- summary (CPU only)
def selected(by_r, k):
    """Validation-selected restart among restarts 0..k-1 (ties -> lowest restart); None if any of them is missing."""
    if any(r not in by_r for r in range(k)):
        return None
    best = max(range(k), key=lambda r: (float(by_r[r]['val_acc']), -r))
    return best, float(by_r[best]['acc'])


def collect(rows, mode, w, N):
    out = {}
    for r in rows:
        if r['mode'] == mode and r['arm'] == 'rep' and r['w'] == str(w) and r['N'] == str(N) and r['acc'] != '':
            out.setdefault(int(r['instance']), {})[int(r['restart'])] = r
    return out


def cell(rows, mode, w, N, k):
    """-> dict of per-instance selected acc (all 10 instances required) or None (incomplete)."""
    d = collect(rows, mode, w, N)
    sel = {s: selected(d.get(s, {}), k) for s in INST}
    return None if any(v is None for v in sel.values()) else sel


def solved(sel, thr):
    return sum(a >= thr for _, a in sel.values())


def mode_summary():
    rows = read_rows()
    e18rows = read_rows(E18_CSV)
    print(f'E22 summary ({sum(r["mode"] == "grid" for r in rows)} of 200 grid rows present)')
    for r in rows:
        if r['mode'] in ('g1', 'g2'):
            print(f'  {r["mode"]}: acc={r["acc"] or "-"} elapsed_s={r["elapsed_s"]} peak_gpu_mb={r["peak_gpu_mb"]} '
                  f'cpu_rss_mb={r["cpu_rss_mb"]} note={r["note"]}')
    print('\nper rung and N: solved at 0.95 / 0.90 (best-of-3, best-of-5), max and median selected test acc (best-of-5)')
    res = {}
    for w, N in GRID:
        line = f'  mix{w} N={N:>10,}: '
        c3, c5 = cell(rows, 'grid', w, N, 3), cell(rows, 'grid', w, N, 5)
        if c5 is None or c3 is None:
            have = len(collect(rows, 'grid', w, N))
            line += f'incomplete ({sum(len(v) for v in collect(rows, "grid", w, N).values())} of 50 runs, {have} instances started)'
            res[(w, N)] = None
        else:
            accs = [a for _, a in c5.values()]
            res[(w, N)] = dict(s95=solved(c5, .95), mx=max(accs), accs=accs)
            line += (f'b3 {solved(c3, .95)}/10 @.95, {solved(c3, .90)}/10 @.90 | b5 {solved(c5, .95)}/10 @.95, '
                     f'{solved(c5, .90)}/10 @.90 | max {max(accs):.4f} median {statistics.median(accs):.4f}')
        print(line)

    def R(name, text, ok):
        print(f'  {name}: {text}: ' + ('incomplete' if ok is None else ('HOLDS' if ok else 'DOES NOT HOLD')))

    print('\nR1-R5 (as written in the spec)')
    g = lambda w, N: res.get((w, N))
    a, b, c, d = g(2, 3_200_000), g(4, 3_200_000), g(8, 3_200_000), g(4, 10_000_000)
    R('R1', 'rep2 at 3.2M best-of-5 solves >= 8/10', None if a is None else a['s95'] >= 8)
    R('R2', 'rep4 at 3.2M best-of-5 solves 0/10 and every selected run <= 0.55', None if b is None else (b['s95'] == 0 and b['mx'] <= 0.55))
    R('R3', 'rep8 at 3.2M best-of-5 solves 0/10', None if c is None else c['s95'] == 0)
    R('R4', 'solved count at 3.2M non-increasing in w (rep2 >= rep4 >= rep8)',
      None if None in (a, b, c) else (a['s95'] >= b['s95'] >= c['s95']))
    R('R5', 'rep4 at 10M best-of-5 solves 0/10 and every selected run <= 0.55', None if d is None else (d['s95'] == 0 and d['mx'] <= 0.55))

    print('\nclaim by outcome (spec wording)')
    # rep4: judged at the largest N with a success, else at 10M
    if d is None:
        print('  rep4: incomplete (10M cell not complete, so the largest N with a success is not determined)')
    else:
        if d['s95'] >= 1:
            at, x = 10_000_000, d
        elif b is not None and b['s95'] >= 1:
            at, x = 3_200_000, b
        elif b is None:
            at, x = None, None
        else:
            at, x = 10_000_000, d
        if x is None:
            print('  rep4: incomplete (3.2M cell not complete)')
        elif x['s95'] >= 8:
            print(f'  rep4 at N={at:,}: >= 8/10 -> four-byte mixing is learnable by this learner at that N (data and steps growing '
                  f'together); informed-cost ratio N / 652 = {at / 652:,.0f} (grid-bounded upper bound); thesis scope extends to w <= 4')
        elif x['s95'] >= 1:
            print(f'  rep4 at N={at:,}: {x["s95"]}/10 -> four-byte mixing is partly learned at that N (counts reported), no cost')
        elif x['mx'] <= 0.55:
            print('  rep4: 0/10, all <= 0.55 -> the learner bound moves from 1M to 10M CRPs; w >= 4 stays open, not secure')
        else:
            print('  rep4: 0/10, some selected run in (0.55, 0.95) -> reported as measured: above chance but unsolved; no claim beyond the counts')
    if a is None:
        print('  rep2 at 3.2M: incomplete')
    elif a['s95'] >= 8:
        print(f'  rep2 at 3.2M: {a["s95"]}/10 >= 8/10 -> w=2 cost = 3.2M / 670 = {3_200_000 / 670:,.0f}x (single grid point, upper bound)')
    else:
        print(f'  rep2 at 3.2M: {a["s95"]}/10 < 8/10 -> two-byte mixing partly learned at 3.2M (counts reported), no cost'
              + ('; below E18\'s 6/10 at 1M, reported as measured' if a['s95'] < 6 else ''))
    if c is None:
        print('  rep8 at 3.2M: incomplete')
    elif c['s95'] == 0:
        print('  rep8 at 3.2M: 0/10 -> the learner bound on the full linear layer moves to 3.2M CRPs')
    else:
        print(f'  rep8 at 3.2M: {c["s95"]}/10 -> reported as measured; contradicts the E11/E18 record on the same layer at lower N; check against G1')

    print('\nEXPLORATORY (not counted): restarts >= 0.95 that were not the validation-selected one (best-of-5 selection)')
    for w, N in GRID:
        d_ = collect(rows, 'grid', w, N)
        c5 = cell(rows, 'grid', w, N, 5)
        if c5 is None:
            print(f'  mix{w} N={N:>10,}: incomplete')
            continue
        n = sum(1 for s in INST for r, row in d_[s].items() if r != c5[s][0] and float(row['acc']) >= 0.95)
        print(f'  mix{w} N={N:>10,}: {n} unselected restarts >= 0.95 (of {sum(len(v) for v in d_.values()) - 10})')

    print('\nper-instance table, validation-selected best-of-5 test acc, paired with E18 e11 (60k / 200k / 1M)')
    for w in (2, 4, 8):
        Ns = [(n, 'e18') for n in E18_NS] + [(3_200_000, 'e22')] + ([(10_000_000, 'e22')] if w == 4 else [])
        print(f'  mix{w}:  instance  ' + '  '.join(f'{n:>10,}' for n, _ in Ns))
        cells = {}
        for n, src in Ns:
            cells[n] = cell(e18rows if src == 'e18' else rows, 'e11' if src == 'e18' else 'grid', w, n, 5)
        for s in INST:
            print(f'         {s:>8}  ' + '  '.join(f'{cells[n][s][1]:>10.4f}' if cells[n] else f'{"incomplete":>10}' for n, _ in Ns))


MODES = {'g1': mode_g1, 'g2': mode_g2, 'grid': mode_grid}

if __name__ == '__main__':
    modes = sys.argv[1:]
    assert modes, 'usage: e22mixscale.py g1|g2|grid|summary ...'
    assert all(m in MODES or m == 'summary' for m in modes), f'unknown mode in {modes}'
    if modes == ['summary']:
        mode_summary()
        raise SystemExit(0)
    e18 = import_e18()
    e18.load()                      # asserts CUDA and pypuf 2.2.0, execs E13/E11 unchanged, patches the layers, runs E18's checks
    torch = e18.torch_
    torch.backends.cuda.matmul.allow_tf32 = False       # TF32 off
    torch.backends.cudnn.allow_tf32 = False
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    acquire_lock(e18)
    try:
        W = Writer(e18.HW, e18.NCOLS)
        for mode in modes:
            if mode == 'summary':
                mode_summary()
                continue
            W.start(mode)
            MODES[mode](W, e18)
        W.f.close()
    finally:
        release_lock()
