"""E19: simulated higher-order XOR Arbiter PUFs behind the secret S-box. Is the measured 4-/5-XOR failure explained by XOR
order, or by hardware?
Written and pre-specified 2026-10-30 (revised the same day after an independent oracle review), before any E19 run (only a CPU
syntax check and a no-training data/shape check ran).

Pre-specification statement: this file must be committed to git before the first run; record the commit hash in the first CSV row
of the run. (The script writes the hash into a 'start' row at every invocation, marked DIRTY if the file differs from the commit.)
Any later edit to this file after the first E19 row exists voids the pre-specification and must be disclosed as a deviation.
Oracle run order: G0 checks, E20 first (its 'gocheck'), then E18 known + K1 gate, E18 E11 arm, then E19 known + Q1 gate (this
file), E19 secret, then the E18 coordinate-ascent pilot at 1M, then the E18 coordinate-ascent runs. The learner is FROZEN: E11/E12's XOR learner (E7 character-table
features -> k linear heads -> soft XOR as a product of tanh, E11 MaskModel(heads = k) with the suffix mask, trained by E11's
fit(): Adam LR 0.02, batch 4096, >= 6000 steps / >= 20 epochs, BCE, 10% validation), exec'd unchanged. No new variant, no
hyperparameter changed.

Why. E14 put the AES S-box in front of MEASURED 4-XOR and 5-XOR Arbiter PUFs (Mursi et al. 2020, 1M CRPs each). The known-table
(informed) attack succeeded (4-XOR 20k, 5-XOR 100k CRPs), and the secret-table learner stayed near the majority rate at up to
900k. Two explanations are open: (i) XOR order, i.e. the learner's optimisation fails for k >= 4 (E11 already saw k = 3 fail at
800k, 0/5, and k = 2 fail on some instances at 400k), or (ii) something about the measured hardware data. Simulated pypuf XOR
APUFs under the same protocol separate the two: if simulated k = 4, 5 also fail with the same learner and budget, the
measured failure is explained by XOR order within our learner budget. E12 gave k = 3 only 5 instances; Part B extends it to 10.

Rig. E11's data path (AES S-box as the shared secret table, labels via gate1 through E11 make_data, challenge rng 11+100s
train / 12+100s test, 10k test, 10% validation = last tenth of N), device = pypuf XORArbiterPUF(n=64, k, seed=s). E11's
make_data is reused unchanged; its internal_bits() and device_for() are wrapped so cfgs xor3/xor4/xor5 are generated (E11 only
knows xor2, xor3; every other cfg falls through to E11's own function). GPU with CPU-side init/batch order, TF32 off. Every
restart is logged. The test majority rate (maj) is logged per row.

Instances. 40-49 (PUF seed = instance number, same instance index for all k; data rngs 11+100s and 12+100s). Seed
collision check (2026-10-30; scan of every CSV instance column in experiments/ and grep of every script): the highest n = 64
instance index used anywhere is 29, E12/E11 xor used 0-4, the E18 block is 30-39, other PUF seeds in old scripts are >= 300.
Restart seeds: known 1000*s + r (E11 xor-known / E14 convention); secret 100000 + 1000*s + r (E12 xor convention, s >= 40 so no
overlap with E12's 100000-104019). Not verified (a caveat, not a design choice): whether pypuf's XORArbiterPUF with the same seed
and different k shares chain weights; if it does, the k = 3, 4, 5 devices at an instance are nested, not independent. load()
now CHECKS this for instances 40-49 (are the k = 3 pypuf XOR device weights a prefix of the k = 4 and k = 5 weights for the same
seed, and k = 4 a prefix of k = 5?) and prints/logs the result (the 'start' row note); it matters for pairing instances across k.

Modes (python ...-e19xorsim.py <mode> ...; one CSV; modes append and resume):
  known   : k in {3, 4, 5}, instances 40-49, N in {20k, 50k, 100k, 200k, 500k, 900k}, restarts 0-2 (informed attack cost on
            simulated devices; compare E14 known-table, measured: 4-XOR 20k, 5-XOR 100k). 540 runs. Per-instance result =
            validation-selected best of the 3.
  secretA : Part A, the measured-device protocol of E14: shared secret table, k in {3, 4, 5}, N in {200k, 500k, 900k},
            restarts 0-4 (validation-selected best-of-5), instances 40-49. 450 runs.
  secretB : Part B, k = 3 only, N in {800k, 1.6M, 3.2M}, restarts 0-9 (best-of-10), instances 40-49 (E12 had 5 instances,
            0-4). 300 runs.
  summary : (CPU, no training) per (mode, k, N): instances solved (validation-selected test acc >= 0.95), instances "failed"
            (selected test acc <= maj + 0.03), known-table N-to-0.95 per k (median over instances of the smallest grid N whose
            best-of-3 reaches 0.95), and Q1-Q3 and the claims table.
Success = validation-selected test accuracy >= 0.95. "Fails" = selected accuracy within 0.03 of the majority rate (maj + 0.03
or lower), the E14 criterion.

Predictions (before any E19 run; author's prior in brackets):
  Q1 known table (best-of-3) reaches >= 0.95 at N <= 100k for k = 4 and at N <= 200k for k = 5 on >= 8/10 instances.     [0.85]
  Q2 secret table, best-of-5 at N <= 900k: fails (within 0.03 of majority) for k = 4 and k = 5 on >= 9/10 instances at every N.
                                                                                                                  [0.80]
  Q3 k = 3, best-of-10: solves >= 5/10 instances at 1.6M.                                                          [0.35]
Kill / claims table (fixed now; the paper uses the matching wording):
  Q2 holds (simulated k = 4, 5 also fail)    -> the measured failure is explained by XOR order within our learner budget;
                                                reported as a learner bound, not as evidence that XOR PUFs resist the attack.
  simulated k = 4 succeeds (best-of-5 >= 0.95 on >= 5/10 instances at some N <= 900k)
                                             -> the measured failure is hardware-specific and is reported as UNEXPLAINED.
  k = 5 only succeeds, or k = 4 succeeds on 1-4/10 -> mixed; reported as measured with per-instance counts, no stronger claim.
  Q1 fails (known table does not reach 0.95 on simulated k = 4/5 at those N)   -> the Q1 kill is PER k: a failing known-table
                                                control for one k stops the secret modes only for that k (secretA/secretB skip
                                                it, with a printed message); the other k continue. The summary reports which k
                                                passed Q1. Debug that k before reading any secret result for it (the control is
                                                the primary result). For k = 3 the same criterion is applied at N <= 200k
                                                (an assumption: the spec gave limits for k = 4, 5 only); a k = 3 failure skips
                                                the secret k = 3 runs.
Single-seed instability is reported as measured; no instance is dropped, no learner is retuned.

Runtime estimate (from logged runs on the RTX 4070 laptop GPU; not measured for E19): about 25 s per known run (E14), about 76 s
per secret run at k = 4, 5 (E14, 8 s-class data load excluded), about 30-40 s at k = 3 up to 1.6M and about 78 s at 3.2M (E12
xor). known 540 runs about 4 h; secretA 450 runs about 10 h; secretB 300 runs about 5-6 h incl. label generation for up to 3.2M
challenges per instance. Total about 20-24 h.
Start-of-run checks (every mode except 'summary'; training modes need a GPU): torch.cuda.is_available() is asserted, the device
name and pypuf version are printed (pypuf must be 2.2.0), and the lock file experiments/.lock-<script name> is created with the PID
(refuses to start if the lock exists and that PID is alive; removed on normal exit).
Commands: python experiments/encoding-induced-false-security-pufs-e19xorsim.py known ; ... summary (Q1 per k) ; ... secretA secretB
Writes experiments/encoding-induced-false-security-pufs-e19xorsim.csv (append + resume)
cols: test,mode,k,variant,instance,N,restart,acc,val_acc,maj,elapsed_s,hw,note
"""
import csv, os, subprocess, sys, time, numpy as np

SCRIPT = 'encoding-induced-false-security-pufs-e19xorsim'

CSV = 'experiments/encoding-induced-false-security-pufs-e19xorsim.csv'
COLS = ['test', 'mode', 'k', 'variant', 'instance', 'N', 'restart', 'acc', 'val_acc', 'maj', 'elapsed_s', 'hw', 'note']
KEY = ('mode', 'k', 'variant', 'instance', 'N', 'restart')
INST = range(40, 50)
PYPUF_PIN = '2.2.0'
PLAN = {                                     # mode -> (variant, ks, Ns, restarts)
    'known': ('known', (3, 4, 5), (20000, 50000, 100000, 200000, 500000, 900000), 3),
    'secretA': ('shared', (3, 4, 5), (200000, 500000, 900000), 5),
    'secretB': ('shared', (3,), (800000, 1600000, 3200000), 10),
}


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


def weights_of(puf):
    """Chain weights of a pypuf XOR device as a 2-D array (k, n+1 or n) if an attribute is available."""
    for name in ('weight_array', 'weights'):
        if hasattr(puf, name):
            a = np.asarray(getattr(puf, name))
            return a.reshape(a.shape[0], -1) if a.ndim >= 2 else a.reshape(1, -1)
    raise AttributeError('no weight attribute on the pypuf device')


def prefix_check():
    """Are the k = 3 weights a prefix (first 3 chains) of the k = 4 and k = 5 weights for the same seed (and k = 4 of k = 5)?"""
    res = {}
    try:
        for s in INST:
            w = {k: weights_of(E11['XORArbiterPUF'](n=64, k=k, seed=s)) for k in (3, 4, 5)}
            res[s] = (bool(np.array_equal(w[3], w[4][:3])), bool(np.array_equal(w[3], w[5][:3])), bool(np.array_equal(w[4], w[5][:4])))
    except Exception as e:                                        # report, do not hide
        return f'prefix check FAILED to run: {type(e).__name__}: {e}'
    allp = all(all(v) for v in res.values())
    return ('k=3 weights are a prefix of k=4,5 (and k=4 of k=5) for ALL 10 instances: devices are NESTED across k' if allp else
            f'prefix property fails for some instance (k3<k4, k3<k5, k4<k5 by instance): {res}')


PREFIX = 'not checked'


def load():
    import torch
    from importlib.metadata import version
    assert torch.cuda.is_available(), 'CUDA is required for every mode except summary'
    print('device:', torch.cuda.get_device_name(0), '| pypuf', version('pypuf'), '| torch', torch.__version__, flush=True)
    global E11, torch_, DEV, HW, make_data, fit, accm, build, suffix_mask
    torch_ = torch
    E11 = {'__name__': 'e19_import'}
    exec(open('experiments/encoding-induced-false-security-pufs-e11gen.py').read(), E11)
    DEV, make_data, fit, accm, build, suffix_mask = E11['DEV'], E11['make_data'], E11['fit'], E11['accm'], E11['build'], E11['suffix_mask']
    assert version('pypuf') == PYPUF_PIN, f'pypuf {version("pypuf")} != pinned {PYPUF_PIN}'
    HW = f'{torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu"}; torch {torch.__version__}; pypuf {version("pypuf")}'
    orig_bits, orig_dev = E11['internal_bits'], E11['device_for']
    XCFG = ('xor3', 'xor4', 'xor5')
    E11['internal_bits'] = lambda cfg, xb: E11['sbits'](xb) if cfg in XCFG else orig_bits(cfg, xb)
    E11['device_for'] = lambda cfg, s: E11['XORArbiterPUF'](n=64, k=int(cfg[-1]), seed=s) if cfg in XCFG else orig_dev(cfg, s)
    _checks()
    global PREFIX
    PREFIX = prefix_check()
    print('XOR weights prefix check:', PREFIX, flush=True)


def _checks():
    """No training. xor3 through the wrapper equals E11's own xor3 device and bits; xor4/5 give +-1/0-1 labels and 64-bit
    internal challenges; the data have the expected shapes."""
    xb = np.random.default_rng(996).integers(0, 256, (200, 8)).astype(np.uint8)
    assert np.array_equal(E11['internal_bits']('xor4', xb), E11['sbits'](xb))
    for k in (3, 4, 5):
        puf = E11['device_for'](f'xor{k}', 40)
        y = E11['labels'](puf, E11['internal_bits'](f'xor{k}', xb))
        assert y.shape == (200,) and set(np.unique(y)) <= {0.0, 1.0}, f'xor{k} labels'
    a = E11['device_for']('xor3', 3).eval(2 * E11['sbits'](xb).astype(np.int8) - 1)
    b = E11['XORArbiterPUF'](n=64, k=3, seed=3).eval(2 * E11['sbits'](xb).astype(np.int8) - 1)
    assert np.array_equal(a, b), 'wrapper xor3 != E11 xor3'
    xtr, ytr, xva, yva, xte, yte, _ = make_data('xor4', 40, 1000)
    assert xtr.shape == (900, 8) and xva.shape == (100, 8) and xte.shape == (10000, 8) and ytr.shape == (900,)


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

    def _write(self, row):
        self.w.writerow(row)
        self.f.flush()
        print(*row, flush=True)

    def start(self, mode):
        self._write(['E19', 'start', '', '', '', '', '', '', '', '', 0, HW, git_note(mode) + ' | ' + PREFIX])

    def row(self, mode, k, variant, s, N, r, acc, va, maj, t0):
        self._write(['E19', mode, k, variant, s, N, r, round(acc, 4), round(va, 4), round(maj, 4), round(time.time() - t0, 1), HW, ''])


Q1_LIM = {3: 200000, 4: 100000, 5: 200000}       # per-k Q1: known table best-of-3 >= 0.95 at some N <= limit on >= 8/10 instances


def q1_by_k(rows):
    """k -> (instances reaching 0.95 within the limit by validation-selected best-of-3, instances complete on the known grid)."""
    g = {}
    for r in rows:
        if r['mode'] == 'known':
            g.setdefault((int(r['k']), int(r['N']), int(r['instance'])), []).append(r)
    out = {}
    for k, lim in Q1_LIM.items():
        hit = done = 0
        for s in INST:
            sel = {N: max(g[(k, N, s)], key=lambda r: float(r['val_acc'])) for N in PLAN['known'][2] if len(g.get((k, N, s), [])) >= 3}
            done += len(sel) == len(PLAN['known'][2])
            hit += any(float(r['acc']) >= 0.95 for N, r in sel.items() if N <= lim)
        out[k] = (hit, done)
    return out


def run_mode(W, mode):
    variant, ks, Ns, R = PLAN[mode]
    if mode != 'known':
        q1 = q1_by_k(list(csv.DictReader(open(CSV))) if os.path.exists(CSV) else [])
        keep = tuple(k for k in ks if q1[k][0] >= 8 and q1[k][1] == len(INST))
        for k in ks:
            print(f'Q1 gate k={k}: known-table {q1[k][0]}/10 reach 0.95 within N <= {Q1_LIM[k]} ({q1[k][1]}/10 instances complete) ->',
                  'PASS' if k in keep else f'FAIL or incomplete: {mode} SKIPS k={k}', flush=True)
        ks = keep
    for k in ks:
        for s in INST:
            for N in Ns:
                data = None
                for r in range(R):
                    if W.done(mode, k, variant, s, N, r):
                        continue
                    data = data or make_data(f'xor{k}', s, N)
                    xtr, ytr, xva, yva, xte, yte, _ = data
                    maj = float(max(yte.mean(), 1 - yte.mean()))
                    t0 = time.time()
                    seed = 1000 * s + r if mode == 'known' else 100000 + 1000 * s + r
                    m, gen = build('mask', variant, seed, A=suffix_mask(), heads=k)
                    m, va = fit(m, gen, xtr, ytr, xva, yva)
                    W.row(mode, k, variant, s, N, r, accm(m, xte, yte), va, maj, t0)
                del data
                torch_.cuda.empty_cache() if DEV.type == 'cuda' else None


def mode_summary():
    rows = [r for r in csv.DictReader(open(CSV)) if r['mode'] != 'start']
    best = {}                                                     # (mode, k, N, instance) -> (selected row)
    g = {}
    for r in rows:
        g.setdefault((r['mode'], int(r['k']), int(r['N']), int(r['instance'])), []).append(r)
    for key, v in g.items():
        if len(v) >= PLAN[key[0]][3]:
            best[key] = max(v, key=lambda r: float(r['val_acc']))
    cell = {}
    for (mode, k, N, s), r in best.items():
        c = cell.setdefault((mode, k, N), [0, 0, 0])
        c[0] += float(r['acc']) >= 0.95
        c[1] += float(r['acc']) <= float(r['maj']) + 0.03
        c[2] += 1
    for key in sorted(cell):
        print(key, 'solved %d, failed(<=maj+0.03) %d, of %d instances' % tuple(cell[key]))
    print('known-table N-to-0.95 (median over instances; None = not reached on grid):')
    for k in (3, 4, 5):
        n95 = []
        for s in INST:
            ok = [N for N in PLAN['known'][2] if best.get(('known', k, N, s)) and float(best[('known', k, N, s)]['acc']) >= 0.95]
            n95.append(min(ok) if ok else np.inf)
        med = float(np.median(n95))
        print(' k =', k, 'median', None if np.isinf(med) else med, 'per-instance', n95)
    sol = lambda mode, k, N: cell.get((mode, k, N), [0, 0, 0])[0]
    fl = lambda mode, k, N: cell.get((mode, k, N), [0, 0, 0])[1]
    ok = lambda k, lim: sum(1 for s in INST if any(best.get(('known', k, N, s)) and float(best[('known', k, N, s)]['acc']) >= 0.95
                                                   for N in PLAN['known'][2] if N <= lim)) >= 8
    print('Q1 known: k=4 <=100k and k=5 <=200k on >=8/10:', ok(4, 100000), ok(5, 200000))
    print('Q1 per k (kill is per k; instances reaching 0.95 within the limit, instances with the full known grid):',
          q1_by_k(rows), 'limits', Q1_LIM)
    print('Q2 secretA fails on >=9/10 at every N for k=4,5:', all(fl('secretA', k, N) >= 9 for k in (4, 5) for N in PLAN['secretA'][2]))
    print('Q3 k=3 best-of-10 solves >=5/10 at 1.6M:', sol('secretB', 3, 1600000) >= 5)
    print('simulated k=4 succeeds on >=5/10 at some N<=900k (=> hardware-specific, unexplained):',
          any(sol('secretA', 4, N) >= 5 for N in PLAN['secretA'][2]))


if __name__ == '__main__':
    modes = sys.argv[1:]
    assert modes, 'usage: e19xorsim.py known|secretA|secretB|summary ...'
    assert all(m in PLAN or m == 'summary' for m in modes), f'unknown mode in {modes}'
    if modes == ['summary']:
        mode_summary()
    else:
        load()                      # asserts CUDA, pypuf 2.2.0; prints device; checks the XOR weight-prefix property
        acquire_lock()
        try:
            W = Writer()
            for mode in modes:
                if mode == 'summary':
                    mode_summary()
                    continue
                W.start(mode)
                run_mode(W, mode)
            W.f.close()
        finally:
            release_lock()
