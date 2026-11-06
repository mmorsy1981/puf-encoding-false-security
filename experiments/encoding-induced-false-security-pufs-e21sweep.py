"""Threshold-sweep re-analysis (no training). Reuses article/stats_cluster.py (imported read-only).
Run: python encoding-induced-false-security-pufs-e21sweep.py   (CPU, pandas/numpy only)"""
import os, math, time, importlib.util
import numpy as np, pandas as pd
T0 = time.time()
EXP = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(EXP, 'encoding-induced-false-security-pufs-')
spec = importlib.util.spec_from_file_location('sc', os.path.join(EXP, '..', 'article', 'stats_cluster.py'))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
SEED = 20261024
def reseed(): sc._RNG = np.random.default_rng(SEED)   # same seed for every configuration (paper used one running stream)
TS = [0.90, 0.93, 0.95, 0.97, 0.98]
def out(name, rows): pd.DataFrame(rows).to_csv(P + 'e21sweep-' + name + '.csv', index=False)
def r2(x, k=1): return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), k)

def cluster_mat(df, ckey):
    g = df.groupby([ckey, 'N']).succ.mean().unstack('N')
    return list(g.index), list(g.columns), g.values

def stat_row(Ns, mat, kn, **meta):
    reseed(); s = sc.n50_ci(Ns, mat, known=kn)
    kmed = float(np.nanmedian(kn)) if np.isfinite(kn).any() else float('nan')
    return dict(meta, known_median_N=r2(kmed), n50=r2(s['n50']), n50_lo=r2(s['lo']), n50_hi=r2(s['hi']),
                mult=r2(s.get('ratio', float('nan')), 2), mult_lo=r2(s.get('rlo', float('nan')), 2), mult_hi=r2(s.get('rhi', float('nan')), 2),
                boot_nan=s['nan'], known_nan=int(np.isnan(kn).sum()))

def ceil_of(Kl, N):
    return max((n, a) for n, a in Kl if n <= N)[1]

# ---------------- load E9/E15 ----------------
a = pd.concat([pd.read_csv(P + 'e9scale-nscale.csv'), pd.read_csv(P + 'e15scale30.csv')])
nd = a.duplicated(['n', 'variant', 'instance', 'N', 'restart']).sum(); a = a.drop_duplicates(['n', 'variant', 'instance', 'N', 'restart'])
print('E9+E15 rows', len(a), 'dups dropped', nd)
K = {(n, i): sorted(zip(g.N, g.acc)) for (n, i), g in a[a.variant == 'known'].groupby(['n', 'instance'])}
atk = a[a.variant != 'known'].copy()
atk['ceil'] = [ceil_of(K[(n, i)], N) for n, i, N in zip(atk.n, atk.instance, atk.N)]

rows = []
for n in (32, 64, 128):
    for var in ('shared', 'perbyte'):
        d = atk[(atk.n == n) & (atk.variant == var)]
        for t in TS:
            dd = d.assign(succ=(d.acc >= t).astype(float))
            cl, Ns, mat = cluster_mat(dd, 'instance')
            knt = np.array([sc.interp95(K[(n, i)], t) for i in cl])
            rows.append(stat_row(Ns, mat, knt, n=n, variant=var, rule='abs_t', t=t, instances=len(cl)))
        dd = d.assign(succ=(d.acc >= d.ceil - 0.05).astype(float)); cl, Ns, mat = cluster_mat(dd, 'instance')
        knt = np.array([sc.interp95(K[(n, i)], 0.95) for i in cl])
        rows.append(stat_row(Ns, mat, knt, n=n, variant=var, rule='paper_ceil-0.05', t=0.95, instances=len(cl)))
out('sweep', rows)

# ---------------- 2. per-instance known N-to-t, bimodality ----------------
rows = []
for (n, i), pr in sorted(K.items()):
    r = dict(n=n, instance=i, plateau_acc=round(max(x for _, x in pr), 4), max_N=max(N for N, _ in pr))
    for t in TS: r[f'N_to_{t}'] = r2(sc.interp95(pr, t))
    rows.append(r)
out('known_per_instance', rows)
rows = []
for n in (32, 64, 128):
    for var in ('shared', 'perbyte'):
        d = atk[(atk.n == n) & (atk.variant == var)]
        hi, lo = int((d.acc >= 0.954).sum()), int((d.acc < 0.70).sum())
        rows.append(dict(n=n, variant=var, runs=len(d), ge_0954=hi, lt_070=lo, between=len(d) - hi - lo,
                         frac_bimodal=round((hi + lo) / len(d), 4),
                         runs_acc_0p90_to_0p98=int(((d.acc >= 0.90) & (d.acc < 0.98)).sum())))
out('bimodal', rows)

# ---------------- 7. per-restart rates ----------------
rows = []
d = atk[(atk.n == 64) & (atk.variant == 'shared')]
for rule in ('abs_0.95', 'paper_ceil-0.05'):
    d2 = d.assign(succ=((d.acc >= 0.95) if rule == 'abs_0.95' else (d.acc >= d.ceil - 0.05)).astype(float))
    cl, Ns, mat = cluster_mat(d2, 'instance')
    for lab, sl in (('first10', slice(0, 10)), ('all30', slice(0, 30)), ('last20', slice(10, 30))):
        reseed(); m, lo, hi = sc.band(mat[sl])
        for N, x, l, h in zip(Ns, m, lo, hi):
            rows.append(dict(rule=rule, subset=lab, N=N, rate=round(float(x), 4), lo=round(float(l), 3), hi=round(float(h), 3), instances=mat[sl].shape[0]))
out('rates_shared64', rows)

# ---------------- 5. best-of-3 (validation-selected; ties -> lowest restart index) ----------------
rows = []
for n in (32, 64, 128):
    for var in ('shared', 'perbyte'):
        d = atk[(atk.n == n) & (atk.variant == var)].sort_values(['instance', 'N', 'restart'])
        recs = []
        for (i, N), g in d.groupby(['instance', 'N']):
            b = g.iloc[int(np.argmax(g.val_acc.values))]; recs.append(dict(instance=i, N=N, acc=b.acc, ceil=b.ceil))
        b = pd.DataFrame(recs)
        for rule in ('abs_0.95', 'paper_ceil-0.05'):
            b['succ'] = ((b.acc >= 0.95) if rule == 'abs_0.95' else (b.acc >= b.ceil - 0.05)).astype(float)
            cl, Ns, mat = cluster_mat(b, 'instance')
            knt = np.array([sc.interp95(K[(n, i)], 0.95) for i in cl])
            r = stat_row(Ns, mat, knt, n=n, variant=var, rule=rule, t=0.95, instances=len(cl))
            r['inst_success_by_N'] = '; '.join(f'{N}:{x:.2f}' for N, x in zip(Ns, mat.mean(0)))
            rows.append(r)
out('bestof3', rows)

# ---------------- 3. FPGA ----------------
e = pd.read_csv(P + 'e8fpga.csv'); e = e[e.sbox == 'aes']
print('e8 aes rows', len(e), 'dups', int(e.duplicated(['device', 'variant', 'N', 'restart', 'mode']).sum()), 'modes', list(e['mode'].unique()))
DEVS = ['top', 'b0', 'b1', 'b2', 'b3', 'b4']
KE = {k: sorted(zip(g.N, g.acc)) for k, g in e[e.variant == 'known'].groupby('device')}
rows = []
for var in ('shared', 'perbyte'):
    d = e[e.variant == var].copy(); d['ceil'] = [ceil_of(KE[x], N) for x, N in zip(d.device, d.N)]
    for rule, ts in (('abs_t', TS), ('paper_ceil-0.05', [0.95])):
        for t in ts:
            d['succ'] = (d.acc >= t).astype(float) if rule == 'abs_t' else (d.acc >= d.ceil - 0.05).astype(float)
            g = d.groupby(['device', 'N']).succ.mean().unstack('N').loc[DEVS]
            knt = np.array([sc.interp95(KE[x], t) for x in DEVS])
            r = stat_row(list(g.columns), g.values, knt, variant=var, rule=rule, t=t, devices=6)
            r['known_per_device'] = ' '.join(f'{x}:{v:.0f}' for x, v in zip(DEVS, knt)); rows.append(r)
out('fpga', rows)

# ---------------- 4. stress ----------------
s = pd.read_csv(P + 'e9scale-stress.csv')
rows = []
for mdl in ('gauss', 'ar09', 't3', 'bias70'):
    ds = s[s.model == mdl]; KS = {i: sorted(zip(g.N, g.acc)) for i, g in ds[ds.variant == 'known'].groupby('instance')}
    d = ds[ds.variant == 'shared'].copy(); d['ceil'] = [ceil_of(KS[i], N) for i, N in zip(d.instance, d.N)]
    for rule, ts in (('abs_t', TS), ('paper_ceil-0.05', [0.95])):
        for t in ts:
            d['succ'] = (d.acc >= t).astype(float) if rule == 'abs_t' else (d.acc >= d.ceil - 0.05).astype(float)
            cl, Ns, mat = cluster_mat(d, 'instance')
            knt = np.array([sc.interp95(KS[i], t) for i in cl])
            r = stat_row(Ns, mat, knt, model=mdl, rule=rule, t=t, instances=len(cl))
            r['known_never_reaches_t'] = int(np.isnan(knt).sum()); r['known_at_or_below_500'] = int((knt <= 500).sum())
            r['known_max_acc_median'] = round(float(np.median([max(x for _, x in KS[i]) for i in cl])), 3)
            rows.append(r)
out('stress', rows)
print('runtime_s', round(time.time() - T0, 1))
