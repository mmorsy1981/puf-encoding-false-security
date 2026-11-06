"""E18: the mixing ladder. Where does the secret-S-box attack stop as a public GF(2)-linear layer mixes more bytes?
Written and pre-specified 2026-10-30 (revised the same day after an independent oracle review), before any E18 run (only
syntax checks, pure-numpy layer checks and no-training shape checks ran).

Pre-specification statement: this file must be committed to git before the first run; record the commit hash in the first CSV row
of the run. (The script writes the hash into a 'start' row of the CSV at every invocation, and marks it DIRTY if the file differs
from the commit.) Any later edit to this file after the first E18 row exists voids the pre-specification and must be disclosed as a
deviation. The learners are FROZEN: E11's mask-table learner and E13's coordinate-ascent learner, exec'd from their own files, no
new variant, no hyperparameter changed. The only wrapper is the chunked feature construction described under 'Disclosed wrapper'.

Run order fixed by the oracle review (commands at the end of this docstring): G0 checks (this file's load() checks, the other
scripts' checks), E20 first (its 'gocheck'), then E18 known + the K1 gate, then the E18 E11 arm, then E19 known + Q1 gate, E19
secret, then the coordinate-ascent pilot at 1M (E18 'timing'), then the coordinate-ascent runs (E18 'coord'). The e11 and coord modes
of THIS file refuse to start unless the K1 gate passes from the CSV; coord also refuses to start unless the pilot row exists.

Why. The paper's claim "a secret S-box is learned, not hidden" is proven only for byte-local layers. E11 (mix/lin, 0/180 runs,
max 0.510 at 200k), E12 (two bit-sign learners), E13 and E16/E17 (coordinate ascent) did not succeed on the full 64-bit public
linear layer while the known-table control succeeded at 1k. FACTUAL CORRECTION (oracle review): coordinate ascent never ran on the
linear layer in E13/E16/E17 at all (0 'lin' rows in those CSVs), because E13's gate rule (run the lin part only if the identity gate
passed) skipped it. A coordinate-ascent failure on the linear layer is therefore not on record; the record on the linear layer is E11's
and E12's learners only. E18 DELIBERATELY OVERRIDES E13's gate rule: a positive result needs no gate (if coordinate ascent solves a
mixing rung, that is informative regardless), and coordinate-ascent FAILURES are uninformative about security unless the
learner first passes a non-trivial gate. E11's identity-layer gate pass is by construction (the identity layer is the case the learner
was built for). The ladder asks for the missing structure: public linear layers that mix w = 1, 2, 4, 8 adjacent bytes. This does not violate the
E17 stop rule ("no further learner variant"): no learner variant is tried.

Layers (public, the same for all instances; all 64 x 64 over GF(2); internal challenge = L @ S(x), i.e. E11's convention
sbits(x) @ L.T). MAIN LADDER = REPEATED block: L_w applies the SAME invertible 8w x 8w GF(2) block to each of the 64/(8w) groups of
w adjacent bytes (block-diagonal with identical blocks), w in {1, 2, 4, 8}. The block is drawn by rejection (random 0/1 matrix,
redrawn until full rank, E11's algorithm generalised to size d) from np.random.default_rng(6100 + w) for w in {1, 2, 4}. w = 8 is
E11's L_LIN itself (rng 6000; the single block of size 64; asserted equal at load, as before). cfg names mix1, mix2, mix4, mix8
(mix8 == E11 'lin'; the equality of the internal bits is asserted at load). EXTRA ARM 'mix1ind' (arm 'ind', w = 1): INDEPENDENT random
invertible 8 x 8 blocks, one per byte (eight blocks drawn in turn from rng 6110). Both learners use the PUBLIC L (the mask matrix
A_k = XOR_{j>=k} L_j, E11 mask_for), so what is secret is the 256-entry S-box only, as in E11/E13.
Rationale for the extra arm (oracle review): repeated w = 1 is exactly the IDENTITY problem with the secret table B-compose-S (the
same public 8 x 8 map B after every byte, so the unknown table is just B o S), so P2 on the repeated w = 1 rung only checks the layer
code, not the learner. Independent w = 1 has many more free columns than repeated w = 1 (the oracle review counted 55 vs 8; the script computes 56 vs 8, see NCOLS) and is the real gate for the mask-table learner. E11's learner sees
the number of table columns, not w, so it may fail simply because the column count grows; the byte-boundary claim is made only
if it solves independent w = 1 (56 columns) but fails repeated w = 2 (30 columns, fewer). The number of columns (distinct non-zero byte masks
= columns of the learned character table) of every rung and arm is printed at load and in the summary and written in the 'ncols'
column of every CSV row.

Learners (both exec'd unchanged; E13's file execs E11's file):
  e11   : E11's MaskModel('shared') trained by E11's fit() (Adam LR 0.02, batch 4096, >= 6000 steps / >= 20 epochs, BCE, 10%
          validation), seed 1000*s + r. This is the E11 'mix' learner.
  coord : E13's CAL (coordinate ascent over a hard shared table, 1200 iterations, 16-entry blocks, Newton refit, ridge 1e-2,
          random initial table, validation-selected best iterate), seed 1000*s + r, exactly as configured in E13, WITHOUT the
          permutation polish (E16's cal+ and E17's 2400-iteration variant are NOT used).
  known : the true S-box and the true L known; E11's MaskModel('known') (only the 65 weights are trained), seeds 1000*s + r.
Disclosed wrapper (the only change to frozen code): CAL.feats(T, xb) builds an (N, 64, 8) float tensor (about 1.8 GB at 1M). It is
replaced at load by a wrapper that calls the ORIGINAL feats on 100,000-row chunks and concatenates the results. Each output row
depends only on its own input row, so the output is bit-identical to the unchunked function. At load a hard assertion compares the
wrapper (forced to 5,000-row chunks, so that several chunks are exercised) with the original on 20,000 rows (true table and a random
table) with torch.equal and fails loudly otherwise. CAL.scores already chunks at 40,000 rows; nothing else is changed.

Rig. E11's data path (AES S-box, labels via gate1 through E11 make_data, challenge rng 11+100s train / 12+100s test, 10k test,
10% validation = last tenth of N), PUF = pypuf ArbiterPUF(n=64, seed=s, noisiness=0.0), GPU with CPU-side init/batch order
(TF32 off). E11's make_data is reused unchanged; its internal_bits() is wrapped (new cfg names only, all other cfgs fall through
to E11's own function). Every restart is logged.

Instances. 30-39 (same ten for both learners and the known table: paired). PUF seed = instance number, data rngs 11+100s and
12+100s, restart seed 1000*s + r (r = 0-4, i.e. 30000-39004). Timing pilot: instance 60, seed 777000 (not any real seed). Collision
check (done 2026-10-30 by scanning the instance/seed columns of every CSV in experiments/ and grepping every script): the highest instance
index used anywhere for n = 64 is 29 (E15: 10-29, E17: 20-29, E7gpu n30: 0-29, E21 sweep: 0-29); other PUF seeds in old scripts are
300+10*inst+j and 400+inst (all >= 300); layer rngs are 6000/6001 (E11) and 6101, 6102, 6104, 6110 (here; grep of 6104/6110 found
no other use); E12 xor restart seeds are 100000+. So the block 30-39 and instance 60 are unused. (The E19 block is 40-49.)

Modes (python ...-e18mixladder.py <mode> ...; one CSV; modes append and resume). Arms: rep w=1,2,4,8 and ind w=1 (5 rungs/arms).
  known   : every rung and arm, instances 30-39, N in {100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000}, restarts 0-2.
            1500 runs. Per instance the selected run is the validation-best of the 3.
  e11     : every rung and arm, instances 30-39, N in {60k, 200k, 1M}, restarts 0-4 in one pass (best-of-5 by validation
            accuracy). 750 runs. Refuses to start unless K1 passed.
  coord   : same grid with the coordinate-ascent learner, 750 runs. Loop order: restarts OUTERMOST (restarts 0-2 for every cell
            first, then 3-4), so a best-of-3 table (comparable with E13) exists first; best-of-5 total. Refuses to start unless K1
            passed and the timing pilot row exists.
  timing  : memory/time pilot. ONE coordinate-ascent restart at N = 1,000,000 on instance 60 (outside the grid), w = 8 repeated
            (= E11 lin), seed 777000 (not any real restart seed). Logs elapsed time and peak GPU memory
            (torch.cuda.max_memory_allocated, in MiB, column peak_gpu_mb); accuracy and val_acc are LEFT BLANK (not a result).
  summary : (CPU, no training, no pypuf needed) per (mode, arm, N): solved counts at 0.95 (primary) and 0.90 (secondary, fixed
            now, not primary), best-of-3 first then best-of-5; known N-to-0.95 per rung; column counts; P1-P5, K1, K2.
            A cell missing any of the 10 instances (with the required number of restarts) prints 'incomplete', never zero counts.
Success = test accuracy >= 0.95 (primary) of the validation-selected restart; accuracy is logged, so the secondary 0.90 count is
computed from the same rows (secondary only).

Predictions (before any E18 run; author's prior in brackets):
  P1 known table solves 10/10 instances at every rung and arm at N = 10k (the control; equals the K1 gate).              [0.95]
  P2a E11 learner solves REPEATED w = 1 on >= 8/10 instances at 200k (known-learnable; only checks the layer code).       [0.85]
  P2b E11 learner solves INDEPENDENT w = 1 ('mix1ind') on >= 8/10 instances at 200k (the learner gate).                  [0.50]
  P3 the E11 success count is non-increasing in w over the repeated rungs at each N (w = 1 >= 2 >= 4 >= 8).               [0.70]
  P4 the E11 learner solves 0/10 at repeated w = 8 at every N (replicates E11's 0/10 on the same layer, new instances).     [0.90]
  P5 coordinate ascent solves repeated w = 1 on >= 6/10 instances at 200k (best-of-3; E13/E16/E17 identity layer: 7/10 at
     200k best-of-3). A REPLICATION check of earlier work, not a gate.                                                    [0.50]
Kill criteria (fixed now):
  K1 Every rung and arm must be 10/10 at N = 10k (known table, validation-best of 3), otherwise stop (e11 and coord refuse to
     run). The per-rung known-table N-to-0.95, interpolated on the grid (log-N linear interpolation of the restart-mean accuracy,
     per instance, median over the 10 instances), is the denominator of the informed cost. A rung that falls short at lower N
     but passes at 10k is not a kill: report the higher informed cost.
  K2 the E11 learner fails P2b -> the column count explains it (it is not evidence about mixing); no byte-boundary claim is made,
     its earlier identity-layer pass stays a by-construction pass, and the paper must say that no learner passes a non-trivial gate.
Claims by outcome (fixed now; the paper uses the matching wording):
  E11 solves independent w = 1 but fails repeated w = 2 (<= 32 columns)  -> the byte-boundary claim is earned (within E11's learner).
  E11 solves independent w = 1 and repeated w = 2 (or 4), fails above      -> the attack degrades with the number of mixed bytes; the
                                                      break is at the reported rung.
  E11 fails independent w = 1 (K2)                -> column count explains it and no boundary claim is made.
  coord solves any rung with w >= 2               -> that mixing is learnable, at the reported cost (the smallest N at which the
                                                      learner reaches 0.95 on >= 8/10) relative to the known-table N-to-0.95 at that rung.
  both learners fail every w >= 2                 -> a learner bound with a precise location (the rung), not evidence of security.
Single-seed instability or a non-monotone count is reported as measured; no instance is dropped, and no learner is retuned.

Runtime estimate (oracle review; earlier logged runs on the RTX 4070 laptop GPU, not measured for E18): coordinate ascent over the
5 rungs/arms about 6-8.5 days (the 1M cells alone about 118-167 h; the independent arm adds about 1.5 days of that), E11 learner about
10 h, known about 2 h. Risks: CAL has never been run at 1M (E13/E16/E17 stop at <= 500k): the chunked feats() wrapper bounds the largest
temporary to 100k rows, the 'timing' pilot measures peak memory at 1M; a CUDA out-of-memory error is NOT caught (the run stops and
is reported; completed rows are kept and the run resumes).

Start-of-run checks (every mode except 'summary'; the training modes need a GPU): torch.cuda.is_available() is asserted, the device
name and pypuf version are printed (pypuf must be 2.2.0), and the lock file experiments/.lock-<script name> is created with the PID
(refuses to start if the lock exists and that PID is alive; removed on normal exit).

Commands, in the oracle's order (from the repository root):
  python experiments/encoding-induced-false-security-pufs-e20noisectl.py gocheck     (E20 first)
  python experiments/encoding-induced-false-security-pufs-e18mixladder.py known; ... summary   (K1 gate)
  python experiments/encoding-induced-false-security-pufs-e18mixladder.py e11
  python experiments/encoding-induced-false-security-pufs-e19xorsim.py known; ... secretA secretB   (after the Q1 gate)
  python experiments/encoding-induced-false-security-pufs-e18mixladder.py timing      (coordinate-ascent pilot at 1M)
  python experiments/encoding-induced-false-security-pufs-e18mixladder.py coord
Writes experiments/encoding-induced-false-security-pufs-e18mixladder.csv (append + resume)
cols: test,mode,arm,w,learner,instance,N,restart,acc,val_acc,ncols,peak_gpu_mb,elapsed_s,hw,note
"""
import csv, os, subprocess, sys, time, numpy as np

SCRIPT = 'encoding-induced-false-security-pufs-e18mixladder'
CSV = 'experiments/encoding-induced-false-security-pufs-e18mixladder.csv'
COLS = ['test', 'mode', 'arm', 'w', 'learner', 'instance', 'N', 'restart', 'acc', 'val_acc', 'ncols', 'peak_gpu_mb',
        'elapsed_s', 'hw', 'note']
KEY = ('mode', 'arm', 'w', 'learner', 'instance', 'N', 'restart')
ARMS = (('rep', 1), ('rep', 2), ('rep', 4), ('rep', 8), ('ind', 1))
WS = (1, 2, 4, 8)
INST = range(30, 40)
KNOWN_NS = (100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000)
KNOWN_R = 3
NS = (60000, 200000, 1000000)
RESTARTS = 5
PYPUF_PIN = '2.2.0'
LEARNER = {'known': 'known-table', 'e11': 'e11-mask', 'coord': 'coord-ascent', 'timing': 'coord-ascent'}
PILOT_INST, PILOT_SEED, PILOT_N, CHUNK = 60, 777000, 1000000, 100000


def cfg_of(arm, w):
    return f'mix{w}ind' if arm == 'ind' else f'mix{w}'


# ---------------------------------------------------------------- the public layers (pure numpy)
def gf2_invertible_n(rng, d):
    """E11's gf2_invertible generalised to d x d (same draw and rejection order; d = 64 reproduces E11 exactly)."""
    while True:
        L = rng.integers(0, 2, (d, d)).astype(np.uint8)
        M, r = L.copy(), 0
        for c in range(d):
            piv = next((i for i in range(r, d) if M[i, c]), None)
            if piv is None:
                continue
            M[[r, piv]] = M[[piv, r]]
            for i in range(d):
                if i != r and M[i, c]:
                    M[i] ^= M[r]
            r += 1
        if r == d:
            return L


def layer(arm, w):
    """64 x 64 public invertible L. 'rep': the same 8w x 8w block on every group of w bytes (w = 8: E11's L_LIN).
    'ind' (w = 1): independent 8 x 8 blocks, one per byte."""
    if arm == 'rep' and w == 8:
        return gf2_invertible_n(np.random.default_rng(6000), 64)
    d, L = 8 * w, np.zeros((64, 64), dtype=np.uint8)
    if arm == 'rep':
        B = gf2_invertible_n(np.random.default_rng(6100 + w), d)
        for b in range(64 // d):
            L[b * d:(b + 1) * d, b * d:(b + 1) * d] = B
    else:
        assert w == 1
        rng = np.random.default_rng(6110)
        for b in range(8):
            L[b * 8:(b + 1) * 8, b * 8:(b + 1) * 8] = gf2_invertible_n(rng, 8)
    return L


LAYER = {cfg_of(a, w): layer(a, w) for a, w in ARMS}


def n_masks(L):
    """Number of distinct non-zero byte masks of A_k = XOR_{j>=k} L_j (columns of the learned character table)."""
    A = (np.cumsum(L[::-1], axis=0)[::-1] % 2).astype(np.uint8)
    m = np.packbits(A.reshape(64, 8, 8), axis=2)[:, :, 0].astype(int)
    return len({int(v) for v in m.ravel() if v})


NCOLS = {cfg: n_masks(L) for cfg, L in LAYER.items()}


def _gf2_rank(M):
    M, r = M.copy(), 0
    for c in range(M.shape[1]):
        piv = next((i for i in range(r, M.shape[0]) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r


def layer_checks():
    """Pure numpy (no torch, no pypuf): block structure, rank, repeated vs independent."""
    for arm, w in ARMS:
        L, d = LAYER[cfg_of(arm, w)], 8 * w
        assert L.shape == (64, 64)
        blocks = [L[b * d:(b + 1) * d, b * d:(b + 1) * d] for b in range(64 // d)]
        for b, blk in enumerate(blocks):
            assert _gf2_rank(blk) == d, f'block {b} of {arm} w={w} not invertible'
        off = np.ones((64, 64), dtype=bool)
        for b in range(64 // d):
            off[b * d:(b + 1) * d, b * d:(b + 1) * d] = False
        assert not L[off].any(), f'{arm} w={w} has entries outside its diagonal blocks'
        same = all(np.array_equal(blocks[0], x) for x in blocks)
        assert same == (arm == 'rep'), f'{arm} w={w}: repeated/independent structure wrong'
        assert _gf2_rank(L) == 64


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


# ---------------------------------------------------------------- heavy setup (training modes only)
def load():
    """Assert GPU and pypuf pin, exec the frozen learners, patch the layer generator into E11's data path, wrap CAL.feats
    (chunked), run the no-training self-checks."""
    import torch
    from importlib.metadata import version
    assert torch.cuda.is_available(), 'CUDA is required for every mode except summary'
    assert version('pypuf') == PYPUF_PIN, f'pypuf {version("pypuf")} != pinned {PYPUF_PIN}'
    print('device:', torch.cuda.get_device_name(0), '| pypuf', version('pypuf'), '| torch', torch.__version__, flush=True)
    global E13, E11, torch_, DEV, HW, make_data, fit, accm, build, CAL, mask_for, L_LIN, suffix_mask, feats_orig
    torch_ = torch
    E13 = {'__name__': 'e18_import'}
    exec(open('experiments/encoding-induced-false-security-pufs-e13learners.py').read(), E13)
    E11 = E13['E11']
    DEV, make_data, fit, accm, build = E13['DEV'], E13['make_data'], E11['fit'], E11['accm'], E11['build']
    CAL, mask_for, L_LIN, suffix_mask = E13['CAL'], E13['mask_for'], E13['L_LIN'], E13['suffix_mask']
    HW = f'{torch.cuda.get_device_name(0)}; torch {torch.__version__}; pypuf {version("pypuf")}'
    orig = E11['internal_bits']

    def internal_bits(cfg, xb):
        if cfg in LAYER:
            return (E11['sbits'](xb).astype(np.int64) @ LAYER[cfg].T.astype(np.int64) % 2).astype(np.uint8)
        return orig(cfg, xb)

    E11['internal_bits'] = internal_bits          # make_data looks it up in E11's globals at call time
    feats_orig = CAL.feats
    CAL.feats = chunked_feats                     # disclosed wrapper; see docstring
    _checks()
    print('layers: columns of the learned character table per rung/arm:', NCOLS, flush=True)


def chunked_feats(self, T, xb, chunk=CHUNK):
    """CAL.feats on chunk-row slices, concatenated. Bit-identical to the unchunked function (row-wise product)."""
    if len(xb) <= chunk:
        return feats_orig(self, T, xb)
    return torch_.cat([feats_orig(self, T, xb[s0:s0 + chunk]) for s0 in range(0, len(xb), chunk)], 0)


def _checks():
    """No training. Layers are invertible and block structured; mix8 is E11's lin; both learners' features with the true
    table equal the direct parity features of the mixed internal challenge; the chunked CAL.feats equals the original."""
    layer_checks()
    assert np.array_equal(LAYER['mix8'], L_LIN), 'w = 8 is not E11 L_LIN'
    xb = np.random.default_rng(995).integers(0, 256, (300, 8)).astype(np.uint8)
    assert np.array_equal(E11['internal_bits']('mix8', xb), E11['internal_bits']('lin', xb)), 'mix8 != lin'
    xt = torch_.tensor(xb.astype(np.int64), device=DEV)
    true_T = torch_.tensor(E11['SBOX'].astype(np.int64), device=DEV)
    for cfg, L in LAYER.items():
        ref = E11['parity_feats'](E11['internal_bits'](cfg, xb))[:, :64]
        A = mask_for(L)
        got = E11['MaskModel']('known', None, A).feats(torch_.tensor(xb.astype(np.int64))).numpy()
        assert np.allclose(got, ref), f'E11 known features != direct features ({cfg})'
        got = CAL(A).feats(true_T, xt).cpu().numpy()
        assert np.allclose(got, ref), f'CAL features with the true table != direct features ({cfg})'
    # chunked wrapper vs the original function, 20k rows, several chunks
    xs = torch_.tensor(np.random.default_rng(997).integers(0, 256, (20000, 8)).astype(np.int64), device=DEV)
    rt = torch_.randint(0, 256, (256,), generator=torch_.Generator().manual_seed(5)).to(DEV)
    cal = CAL(mask_for(LAYER['mix8']))
    for T in (true_T, rt):
        a = feats_orig(cal, T, xs)
        b = chunked_feats(cal, T, xs, chunk=5000)
        assert a.shape == b.shape and torch_.equal(a, b), 'chunked CAL.feats is NOT bit-identical to the original: stop'
    print('chunked CAL.feats == original on 20,000 rows (5,000-row chunks): OK', flush=True)


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

    def _write(self, row):
        self.w.writerow(row)
        self.f.flush()
        print(*row, flush=True)

    def start(self, mode):
        self._write(['E18', 'start', '', '', '', '', '', '', '', '', '', '', 0, HW, git_note(mode)])

    def row(self, mode, arm, w, s, N, r, acc, va, t0, peak='', note=''):
        rnd = lambda v: '' if v is None else round(v, 4)
        self._write(['E18', mode, arm, w, LEARNER[mode], s, N, r, rnd(acc), rnd(va), NCOLS[cfg_of(arm, w)], peak,
                     round(time.time() - t0, 1), HW, note])


def k1_status(rows):
    """arm -> (solved, instances present) at N = 10k, known table, validation-best of the 3 restarts."""
    out = {}
    for arm, w in ARMS:
        g = {}
        for r in rows:
            if r['mode'] == 'known' and r['arm'] == arm and int(r['w']) == w and int(r['N']) == 10000:
                g.setdefault(int(r['instance']), []).append(r)
        sel = [max(v, key=lambda r: float(r['val_acc'])) for v in g.values() if len(v) >= KNOWN_R]
        out[(arm, w)] = (sum(float(r['acc']) >= 0.95 for r in sel), len(sel))
    return out


def read_rows():
    return [r for r in csv.DictReader(open(CSV))] if os.path.exists(CSV) else []


def require_k1():
    st = k1_status(read_rows())
    ok = all(v == (10, 10) for v in st.values())
    print('K1 gate (known table 10/10 at N = 10k, every rung and arm):', st, '->', 'PASS' if ok else 'FAIL, stopping', flush=True)
    if not ok:
        raise SystemExit('K1 not passed (or known mode incomplete): run/fix the known mode first; e11 and coord refuse to run')


def mode_known(W):
    for arm, w in ARMS:
        cfg = cfg_of(arm, w)
        A = mask_for(LAYER[cfg])
        print(f'{cfg}: {NCOLS[cfg]} columns', flush=True)
        for s in INST:
            for N in KNOWN_NS:
                data = None
                for r in range(KNOWN_R):
                    if W.done('known', arm, w, 'known-table', s, N, r):
                        continue
                    data = data or make_data(cfg, s, N)
                    xtr, ytr, xva, yva, xte, yte, _ = data
                    t0 = time.time()
                    m, gen = build('mask', 'known', 1000 * s + r, A=A)
                    m, va = fit(m, gen, xtr, ytr, xva, yva)
                    W.row('known', arm, w, s, N, r, accm(m, xte, yte), va, t0)


def mode_e11(W):
    require_k1()
    for arm, w in ARMS:
        cfg = cfg_of(arm, w)
        A = mask_for(LAYER[cfg])
        print(f'{cfg}: {NCOLS[cfg]} columns', flush=True)
        for s in INST:
            for N in NS:
                data = None
                for r in range(RESTARTS):
                    if W.done('e11', arm, w, 'e11-mask', s, N, r):
                        continue
                    data = data or make_data(cfg, s, N)
                    xtr, ytr, xva, yva, xte, yte, _ = data
                    t0 = time.time()
                    m, gen = build('mask', 'shared', 1000 * s + r, A=A)
                    m, va = fit(m, gen, xtr, ytr, xva, yva)
                    W.row('e11', arm, w, s, N, r, accm(m, xte, yte), va, t0)
                del data
                torch_.cuda.empty_cache()


def mode_coord(W):
    require_k1()
    if not W.done('timing', 'rep', 8, 'coord-ascent', PILOT_INST, PILOT_N, 0):
        raise SystemExit('the timing pilot row (python ... timing) is missing; run the 1M pilot first')
    for phase in ((0, 1, 2), (3, 4)):                              # restarts OUTERMOST
        for N in NS:
            for arm, w in ARMS:
                cfg = cfg_of(arm, w)
                cal = CAL(mask_for(LAYER[cfg]))
                for s in INST:
                    data = None
                    for r in phase:
                        if W.done('coord', arm, w, 'coord-ascent', s, N, r):
                            continue
                        data = data or make_data(cfg, s, N)
                        xtr, ytr, xva, yva, xte, yte, _ = data
                        t0 = time.time()
                        va, T, wt = cal.run(xtr, ytr, xva, yva, 1000 * s + r)      # E13 defaults: 1200 iterations, block 16
                        W.row('coord', arm, w, s, N, r, cal.acc(T, wt, xte, yte), va, t0)
                    del data
                    torch_.cuda.empty_cache()


def mode_timing(W):
    """Time/memory pilot: ONE CAL restart, w = 8 (repeated = E11 lin), instance 60 (outside the grid), N = 1M, seed 777000.
    Logs elapsed time and peak GPU memory; accuracy is left blank (not a result)."""
    if W.done('timing', 'rep', 8, 'coord-ascent', PILOT_INST, PILOT_N, 0):
        print('timing row already present')
        return
    t0 = time.time()
    xtr, ytr, xva, yva, xte, yte, _ = make_data('mix8', PILOT_INST, PILOT_N)
    dgen = time.time() - t0
    print(f'data generation: {dgen:.1f} s', flush=True)
    cal = CAL(mask_for(LAYER['mix8']))
    torch_.cuda.empty_cache()
    torch_.cuda.reset_peak_memory_stats()
    t0 = time.time()
    cal.run(xtr, ytr, xva, yva, PILOT_SEED)
    torch_.cuda.synchronize()
    peak = round(torch_.cuda.max_memory_allocated() / 2 ** 20, 1)
    W.row('timing', 'rep', 8, PILOT_INST, PILOT_N, 0, None, None, t0, peak=peak,
          note=f'data generation {dgen:.1f} s; accuracy deliberately not logged; seed {PILOT_SEED}; chunk {CHUNK}')


# ---------------------------------------------------------------- summary (no training, no heavy imports)
def n_to_95(ns, acc):
    """Log-N interpolated first crossing of 0.95; inf if never reached on the grid."""
    if acc[0] >= 0.95:
        return float(ns[0])
    for i in range(1, len(ns)):
        if acc[i] >= 0.95:
            t = (0.95 - acc[i - 1]) / (acc[i] - acc[i - 1])
            return float(np.exp(np.log(ns[i - 1]) + t * (np.log(ns[i]) - np.log(ns[i - 1]))))
    return float('inf')


class Incomplete(Exception):
    pass


def mode_summary():
    rows = [r for r in read_rows() if r['mode'] in ('known', 'e11', 'coord')]
    print('columns of the learned character table per rung/arm:', {f'{a}{w}': NCOLS[cfg_of(a, w)] for a, w in ARMS})
    by = {}                                           # (mode, arm, w, N, instance) -> {restart: row}
    for r in rows:
        by.setdefault((r['mode'], r['arm'], int(r['w']), int(r['N']), int(r['instance'])), {})[int(r['restart'])] = r

    def cell(mode, arm, w, N, R):
        """(solved@0.95, solved@0.90) of the validation-best of restarts 0..R-1; Incomplete unless all 10 instances have them."""
        s95 = s90 = n = 0
        for s in INST:
            d = by.get((mode, arm, w, N, s), {})
            if all(r in d for r in range(R)):
                best = max((d[r] for r in range(R)), key=lambda r: float(r['val_acc']))
                s95 += float(best['acc']) >= 0.95
                s90 += float(best['acc']) >= 0.90
                n += 1
        if n < len(INST):
            raise Incomplete(f'{mode} {arm}{w} N={N} best-of-{R}: {n}/10 instances')
        return s95, s90

    for mode, Ns, Rs in (('known', KNOWN_NS, (KNOWN_R,)), ('e11', NS, (3, 5)), ('coord', NS, (3, 5))):
        for R in Rs:
            for arm, w in ARMS:
                out = []
                for N in Ns:
                    try:
                        a, b = cell(mode, arm, w, N, R)
                        out.append(f'{N}: {a}/10 ({b}/10 at 0.90)')
                    except Incomplete:
                        out.append(f'{N}: incomplete')
                print(f'{mode:5s} best-of-{R} {arm}{w} [{NCOLS[cfg_of(arm, w)]} cols]:', '; '.join(out))
    # known-table N-to-0.95 per rung (restart-mean accuracy per instance, median over instances)
    denom = {}
    for arm, w in ARMS:
        per = []
        for s in INST:
            accs = []
            for N in KNOWN_NS:
                d = by.get(('known', arm, w, N, s), {})
                accs.append(np.mean([float(d[r]['acc']) for r in range(KNOWN_R)]) if all(r in d for r in range(KNOWN_R)) else None)
            if None not in accs:
                per.append(n_to_95(KNOWN_NS, accs))
        if len(per) == len(INST):
            denom[(arm, w)] = float(np.median(per))
            print(f'known N-to-0.95 {arm}{w}: median {denom[(arm, w)]:.0f}  per-instance {[round(x) if np.isfinite(x) else ">10000" for x in per]}')
        else:
            print(f'known N-to-0.95 {arm}{w}: incomplete ({len(per)}/10 instances)')

    def check(name, fn):
        try:
            print(name, fn())
        except Incomplete as e:
            print(name, 'incomplete (' + str(e) + ')')

    check('P1 known 10/10 at 10k, every rung and arm (= K1 gate passed):',
          lambda: all(cell('known', a, w, 10000, KNOWN_R)[0] == 10 for a, w in ARMS))
    check('P2a e11 repeated w=1 >= 8/10 at 200k (layer-code check), best-of-5:', lambda: cell('e11', 'rep', 1, 200000, 5)[0] >= 8)
    check('P2b e11 independent w=1 >= 8/10 at 200k (learner gate), best-of-5:', lambda: cell('e11', 'ind', 1, 200000, 5)[0] >= 8)
    check('P3 e11 non-increasing over repeated w at each N, best-of-5:',
          lambda: {N: all(cell('e11', 'rep', a, N, 5)[0] >= cell('e11', 'rep', b, N, 5)[0] for a, b in zip(WS[:-1], WS[1:])) for N in NS})
    check('P4 e11 repeated w=8 0/10 at every N, best-of-5:', lambda: all(cell('e11', 'rep', 8, N, 5)[0] == 0 for N in NS))
    check('P5 (replication check, not a gate) coord repeated w=1 >= 6/10 at 200k, best-of-3:',
          lambda: cell('coord', 'rep', 1, 200000, 3)[0] >= 6)
    check('K1 stop (any known control < 10/10 at 10k):', lambda: any(cell('known', a, w, 10000, KNOWN_R)[0] != 10 for a, w in ARMS))
    check('K2 (e11 fails independent w=1 at 200k):', lambda: cell('e11', 'ind', 1, 200000, 5)[0] < 8)
    for mode in ('e11', 'coord'):                      # informed-cost ratio: smallest N with >= 8/10, best-of-5 then best-of-3
        for R in (5, 3):
            for arm, w in ARMS:
                try:
                    ok = [N for N in NS if cell(mode, arm, w, N, R)[0] >= 8]
                except Incomplete as e:
                    print(f'cost {mode} best-of-{R} {arm}{w}: incomplete'); continue
                d = denom.get((arm, w))
                print(f'cost {mode} best-of-{R} {arm}{w}: smallest N with >= 8/10 = {min(ok) if ok else "none on grid"}'
                      + (f'; ratio to known N-to-0.95 {min(ok) / d:.0f}x' if ok and d else ''))


MODES = {'known': mode_known, 'e11': mode_e11, 'coord': mode_coord, 'timing': mode_timing}

if __name__ == '__main__':
    modes = sys.argv[1:]
    assert modes, 'usage: e18mixladder.py known|e11|coord|timing|summary ...'
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
