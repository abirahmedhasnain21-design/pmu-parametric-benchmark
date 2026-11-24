import sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score

from generator_v3 import N_CYC, FS, F0
from gen_build import build_param
from partF_core import PHYS
import partF_build as pb

RS = 42
SEEDS = [0, 1, 2, 3, 4]
PAPER = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc', 'Fi', 'Fv', 'Pai']
PHASOR = ['Im', 'Vm', 'Pai', 'Pav', 'Fi', 'Fv']
C6 = ['LG', 'LL', 'LLG', 'LLL', 'LLLG', 'NF']
C5 = {'LG': 'LG', 'LL': 'LL', 'LLG': 'LLG', 'LLL': 'LLL/G', 'LLLG': 'LLL/G', 'NF': 'NF'}
OUT = 'results/cr_results.json'
try:
    RES = json.load(open(OUT))
except Exception:
    RES = {}


def save():
    json.dump(RES, open(OUT, 'w'), indent=1)


def ms(v):
    v = np.asarray(v, float)
    return f'{v.mean():6.2f} ± {v.std():4.2f}'


def labels(D, merge):
    s = D.fault_types.map(C5) if merge else D.fault_types
    return s.values


def forest(seed=RS):
    return RandomForestClassifier(100, random_state=seed, n_jobs=2)


def param_full(seed, n=50):
    g = build_param(n, seed=seed, sweep_phase=False)
    g['phase_rot'], g['meas'] = 0, 'bus'
    P = pb.add_pu_per_run(gen_physical_all(g))
    return P


def gen_physical_all(gen):
    gen = gen.copy()
    gen['run_id'] = (gen.time_stamp.diff().fillna(0) < 0).cumsum()
    out = []
    for rid, g in gen.groupby('run_id'):
        g = g.reset_index(drop=True)
        f = pb.phys_features(g)
        for c in set(PAPER + PHASOR):
            f[c] = g[c].values[N_CYC - 1:]
        f['fault_types'] = g.fault_types.iloc[0]
        f['run_id'] = f'gen_{rid}'
        f['t_fault'] = g.t_fault.iloc[0]
        for p in ['m', 'Rf', 'theta0', 'scr', 'load']:
            f[p] = g[p].iloc[0]
        out.append(f)
    return pd.concat(out, ignore_index=True)


def post(P):
    return P[P.time_stamp >= P.t_fault].reset_index(drop=True)


_CACHE = {}


def PARAM(seed, n=50):
    if (seed, n) not in _CACHE:
        _CACHE[(seed, n)] = param_full(seed, n)
    return _CACHE[(seed, n)]


BASE_RAW = pb.load_base()
BASE_PHYS = pb.add_pu_per_run(pb.base_physical(BASE_RAW))


def best_single(Xtr, ytr, Xva, yva, feats, seed=RS):
    sc = {}
    for f in feats:
        t = DecisionTreeClassifier(max_depth=4, random_state=seed).fit(Xtr[[f]], ytr)
        sc[f] = accuracy_score(yva, t.predict(Xva[[f]]))
    return max(sc, key=sc.get)


def diagnostic(D, feats, merge, by_run, seed=RS):
    y = labels(D, merge)
    X = D[feats].reset_index(drop=True)
    idx = np.arange(len(X))
    if by_run:
        g = D.run_id.values
        tr, te = next(GroupShuffleSplit(1, test_size=.3, random_state=seed).split(X, y, g))
        fi, va = next(GroupShuffleSplit(1, test_size=.25, random_state=seed)
                      .split(tr, y[tr], g[tr]))
    else:
        tr, te = train_test_split(idx, test_size=.2, random_state=seed, stratify=y)
        fi, va = train_test_split(np.arange(len(tr)), test_size=.25,
                                  random_state=seed, stratify=y[tr])
    fi, va = tr[fi], tr[va]
    acc = lambda m, c: accuracy_score(y[te], m.fit(X.iloc[tr][c], y[tr])
                                      .predict(X.iloc[te][c])) * 100
    f_ = acc(forest(seed), feats)
    d4 = acc(DecisionTreeClassifier(max_depth=4, random_state=seed), feats)
    bf = best_single(X.iloc[fi], y[fi], X.iloc[va], y[va], feats, seed)
    s1 = acc(DecisionTreeClassifier(max_depth=4, random_state=seed), [bf])
    return f_, d4, s1, bf


def E1():
    print('\n=== E1  difficulty diagnostic (single feature chosen on validation)')
    rows = {}
    for merge in (False, True):
        k = 5 if merge else 6
        r = diagnostic(BASE_RAW, PAPER, merge, False)
        rows[f'existing | 9 original | {k}'] = [list(r)]
        r = diagnostic(BASE_PHYS, PHYS, merge, False)
        rows[f'existing | 12 invariant | {k}'] = [list(r)]
        for feats, nm in ((PAPER, '9 original'), (PHYS, '12 invariant')):
            rows[f'param | {nm} | {k}'] = [list(diagnostic(post(PARAM(s)), feats, merge, True))
                                           for s in SEEDS]
            print('  done', nm, k, time.strftime('%X'), flush=True)
    print(f"{'dataset | features | classes':34}{'forest':>16}{'depth-4':>16}{'1 feature':>16}{'margin':>9}  best")
    for key, r in rows.items():
        a = np.array([x[:3] for x in r], float)
        print(f'{key:34}{ms(a[:,0]):>16}{ms(a[:,1]):>16}{ms(a[:,2]):>16}'
              f'{(a[:,0]-a[:,2]).mean():9.2f}  {sorted(set(x[3] for x in r))}')
    RES['E1'] = rows
    save()


def E2():
    print('\n=== E2  regime-shifted splits on the parametric set (forest, 5 classes)')
    out = {}
    for s in SEEDS:
        D = post(PARAM(s))
        y = labels(D, True)
        for feats, nm in ((PAPER, '9 original'), (PHYS, '12 invariant')):
            X = D[feats]
            tr, te = next(GroupShuffleSplit(1, test_size=.3, random_state=RS)
                          .split(X, y, D.run_id.values))
            m = forest().fit(X.iloc[tr], y[tr])
            p = m.predict(X.iloc[te])
            early = (D.time_stamp.values[te] - D.t_fault.values[te]) < 1 / F0
            out.setdefault(f'{nm} | by run, all', []).append(accuracy_score(y[te], p) * 100)
            out.setdefault(f'{nm} | by run, first cycle', []).append(
                accuracy_score(y[te][early], p[early]) * 100)
            out.setdefault(f'{nm} | by run, settled', []).append(
                accuracy_score(y[te][~early], p[~early]) * 100)
            for lab, trm in (('Rf<=5 -> Rf>=10', D.Rf.values <= 5),
                             ('m<0.5 -> m>=0.5', D.m.values < .5),
                             ('SCR>=10 -> SCR<10', D.scr.values >= 10)):
                m = forest().fit(X[trm], y[trm])
                out.setdefault(f'{nm} | {lab}', []).append(
                    accuracy_score(y[~trm], m.predict(X[~trm])) * 100)
        print('  seed', s, time.strftime('%X'), flush=True)
    for k, v in out.items():
        print(f'  {k:40}{ms(v):>16}')
    RES['E2'] = out
    save()


def E3():
    print('\n=== E3  forest accuracy (by run, 5 classes) against runs per class')
    out = {}
    for n in (25, 50, 100, 200):
        for s in SEEDS[:3]:
            D = post(PARAM(s, n)) if n != 50 else post(PARAM(s))
            y = labels(D, True)
            tr, te = next(GroupShuffleSplit(1, test_size=.3, random_state=RS)
                          .split(D, y, D.run_id.values))
            tr = tr[::4] if n >= 100 else tr
            for feats, nm in ((PAPER, '9 original'), (PHYS, '12 invariant')):
                m = forest().fit(D.iloc[tr][feats], y[tr])
                out.setdefault(f'{nm} | n={n}', []).append(
                    accuracy_score(y[te], m.predict(D.iloc[te][feats])) * 100)
            if n != 50:
                _CACHE.pop((s, n), None)
        print('  n', n, time.strftime('%X'), flush=True)
    for k, v in out.items():
        print(f'  {k:28}{ms(v):>16}')
    RES['E3'] = out
    save()


def declare(pred, ok_mask, M):
    run = 0
    for i, o in enumerate(ok_mask):
        run = run + 1 if o else 0
        if run >= M:
            return i
    return None


def E4():
    print('\n=== E4  detection latency on held-out runs (5 classes)')
    Ms = (1, 3, 5, 10, 20)
    out = {}
    for s in SEEDS:
        P = PARAM(s)
        y = np.where(P.time_stamp.values >= P.t_fault.values, labels(P, True), 'NF')
        tr, te = next(GroupShuffleSplit(1, test_size=.3, random_state=RS)
                      .split(P, y, P.run_id.values))
        for feats, nm in ((PAPER, '9 original'), (PHASOR, 'phasor only'),
                          (PHYS, '12 invariant')):
            mdl = forest().fit(P.iloc[tr][feats], y[tr])
            T = P.iloc[te].assign(pred=mdl.predict(P.iloc[te][feats]))
            rng = np.random.default_rng(s)
            for rid, g in T.groupby('run_id', sort=False):
                cls = C5[g.fault_types.iloc[0]]
                t, t0, pr = g.time_stamp.values, g.t_fault.iloc[0], g.pred.values
                pre = t < t0 if cls != 'NF' else np.ones(len(t), bool)
                for rate, step in (('sample', 1), ('50fps', N_CYC)):
                    off = 0 if step == 1 else int(rng.integers(0, step))
                    sel = np.arange(off, len(t), step)
                    tt, pp, ppre = t[sel], pr[sel], pre[sel]
                    for M in (Ms if step == 1 else (1, 2, 3)):
                        key = f'{nm} | {rate} | M={M}'
                        d = out.setdefault(key, dict(lat=[], wrong=0, miss=0,
                                                     fa_runs=0, n_runs=0, n_fault=0))
                        d['n_runs'] += 1
                        if declare(pp[ppre], pp[ppre] != 'NF', M) is not None:
                            d['fa_runs'] += 1
                        if cls == 'NF':
                            continue
                        d['n_fault'] += 1
                        q = pp[~ppre]
                        i_any = declare(q, q != 'NF', M)
                        i_ok = declare(q, q == cls, M)
                        if i_ok is None:
                            d['miss'] += 1
                        else:
                            d['lat'].append((tt[~ppre][i_ok] - t0) * 1000)
                        if i_any is not None and (i_ok is None or i_any < i_ok):
                            d['wrong'] += 1
        print('  seed', s, time.strftime('%X'), flush=True)
    print(f"  {'features | rate | rule':34}{'median':>8}{'p90':>8}{'max':>8}"
          f"{'missed':>9}{'wrong 1st':>11}{'false decl.':>13}")
    for k, d in out.items():
        L = np.array(d['lat'])
        print(f"  {k:34}{np.median(L):8.2f}{np.percentile(L,90):8.2f}{L.max():8.2f}"
              f"{d['miss']:>5}/{d['n_fault']:<4}{d['wrong']:>6}/{d['n_fault']:<4}"
              f"{d['fa_runs']:>6}/{d['n_runs']}")
        d['lat'] = [float(x) for x in L]
    RES['E4'] = out
    save()


def cap(X, y, n=15000, seed=0):
    if len(X) <= n:
        return X, y
    i = np.random.RandomState(seed).choice(len(X), n, replace=False)
    return X[i], y[i]


def E5():
    print('\n=== E5  cross-dataset transfer, 12 invariant features, forest')
    out = {}
    yB6 = labels(BASE_PHYS, False)
    Xb = BASE_PHYS[PHYS].values
    for s in SEEDS:
        P = post(PARAM(s))
        Xg, yG6 = P[PHYS].values, labels(P, False)
        b1, b2, yb1, yb2 = train_test_split(Xb, yB6, test_size=.2, random_state=RS, stratify=yB6)
        tr, te = next(GroupShuffleSplit(1, test_size=.3, random_state=RS)
                      .split(Xg, yG6, P.run_id.values))
        for tag, (Xtr, ytr, Xte, yte) in {
                'existing -> existing': (b1, yb1, b2, yb2),
                'param -> param (by run)': (Xg[tr], yG6[tr], Xg[te], yG6[te]),
                'existing -> param': (Xb, yB6, Xg, yG6),
                'param -> existing': (Xg, yG6, Xb, yB6)}.items():
            for merge in (False, True):
                ytr_ = pd.Series(ytr).map(C5).values if merge else ytr
                yte_ = pd.Series(yte).map(C5).values if merge else yte
                Xc, yc = cap(Xtr, ytr_)
                Xe, ye = cap(Xte, yte_, seed=1)
                p = forest().fit(Xc, yc).predict(Xe)
                out.setdefault(f'{tag} | {5 if merge else 6} cls (trained {5 if merge else 6})',
                               []).append(accuracy_score(ye, p) * 100)
        print('  seed', s, time.strftime('%X'), flush=True)
    for k, v in out.items():
        print(f'  {k:52}{ms(v):>16}')
    RES['E5'] = out
    save()


def noisy_param(seed, snr_db, n=50):
    import gen_build
    orig = gen_build.run_case
    rng = np.random.default_rng(1000 + seed)

    def noisy(kind, **kw):
        r = orig(kind, **kw)
        pre = r['t'] < r['t_fault']
        for key in ('I', 'V'):
            x = r[key].copy()
            rms = np.sqrt((x[pre] ** 2).mean(axis=0))
            x += rng.normal(size=x.shape) * rms / 10 ** (snr_db / 20)
            r[key] = x
        return r
    gen_build.run_case = noisy
    try:
        g = build_param(n, seed=seed, sweep_phase=False)
    finally:
        gen_build.run_case = orig
    g['phase_rot'], g['meas'] = 0, 'bus'
    return pb.add_pu_per_run(gen_physical_all(g))


def latency_table(getP, feat_sets, rules, tag):
    out = {}
    for s in SEEDS:
        P = getP(s)
        y = np.where(P.time_stamp.values >= P.t_fault.values, labels(P, True), 'NF')
        tr, te = next(GroupShuffleSplit(1, test_size=.3, random_state=RS)
                      .split(P, y, P.run_id.values))
        for feats, nm in feat_sets:
            mdl = forest().fit(P.iloc[tr][feats], y[tr])
            T = P.iloc[te].assign(pred=mdl.predict(P.iloc[te][feats]))
            acc_post = accuracy_score(y[te][P.iloc[te].time_stamp.values >= P.iloc[te].t_fault.values],
                                      T.pred.values[T.time_stamp.values >= T.t_fault.values]) * 100
            out.setdefault(f'{nm} | post-fault accuracy', dict(v=[]))['v'].append(acc_post)
            rng = np.random.default_rng(s)
            for rid, g in T.groupby('run_id', sort=False):
                cls = C5[g.fault_types.iloc[0]]
                t, t0, pr = g.time_stamp.values, g.t_fault.iloc[0], g.pred.values
                pre = t < t0 if cls != 'NF' else np.ones(len(t), bool)
                for rate, step, Ms_ in rules:
                    off = 0 if step == 1 else int(rng.integers(0, step))
                    sel = np.arange(off, len(t), step)
                    tt, pp, ppre = t[sel], pr[sel], pre[sel]
                    for M in Ms_:
                        d = out.setdefault(f'{nm} | {rate} | M={M}', dict(
                            lat=[], wrong=0, miss=0, fa_runs=0, n_runs=0, n_fault=0))
                        d['n_runs'] += 1
                        if declare(pp[ppre], pp[ppre] != 'NF', M) is not None:
                            d['fa_runs'] += 1
                        if cls == 'NF':
                            continue
                        d['n_fault'] += 1
                        q = pp[~ppre]
                        i_any, i_ok = declare(q, q != 'NF', M), declare(q, q == cls, M)
                        if i_ok is None:
                            d['miss'] += 1
                        else:
                            d['lat'].append(float((tt[~ppre][i_ok] - t0) * 1000))
                        if i_any is not None and (i_ok is None or i_any < i_ok):
                            d['wrong'] += 1
        print('  seed', s, time.strftime('%X'), flush=True)
    print(f"  {tag}\n  {'features | rate | rule':34}{'median':>8}{'p90':>8}{'max':>8}"
          f"{'missed':>9}{'wrong 1st':>11}{'false decl.':>13}")
    for k, d in out.items():
        if 'v' in d:
            print(f'  {k:34}{ms(d["v"])}')
            continue
        L = np.array(d['lat']) if d['lat'] else np.array([np.nan])
        print(f"  {k:34}{np.median(L):8.2f}{np.percentile(L,90):8.2f}{L.max():8.2f}"
              f"{d['miss']:>5}/{d['n_fault']:<4}{d['wrong']:>6}/{d['n_fault']:<4}"
              f"{d['fa_runs']:>6}/{d['n_runs']}")
    return out


def E8():
    print('\n=== E8  latency with measurement noise on the waveforms')
    rules = (('sample', 1, (1, 5, 20)), ('50fps', N_CYC, (1, 2)))
    for snr in (40, 30):
        RES[f'E8_{snr}dB'] = latency_table(
            lambda s: noisy_param(s, snr), ((PAPER, '9 original'), (PHYS, '12 invariant')),
            rules, f'SNR {snr} dB')
        save()


def E9():
    print('\n=== E9  transfer (12 invariant, 5 classes, trained 5) vs runs per class')
    out = {}
    Xb, yb = BASE_PHYS[PHYS].values, labels(BASE_PHYS, True)
    for n in (50, 200):
        for s in SEEDS:
            g = build_param(n, seed=s, sweep_phase=False)
            g['phase_rot'], g['meas'] = 0, 'bus'
            P = post(pb.add_pu_per_run(pb.gen_physical(g)))
            Xg, yg = P[PHYS].values, labels(P, True)
            for cap_n in (15000, 60000):
                Xc, yc = cap(Xg, yg, n=cap_n)
                p = forest().fit(Xc, yc).predict(Xb)
                out.setdefault(f'param -> existing | n={n} | cap {cap_n}', []).append(
                    accuracy_score(yb, p) * 100)
                Xc, yc = cap(Xb, yb, n=cap_n)
                Xe, ye = cap(Xg, yg, n=60000, seed=1)
                p = forest().fit(Xc, yc).predict(Xe)
                out.setdefault(f'existing -> param | n={n} | cap {cap_n}', []).append(
                    accuracy_score(ye, p) * 100)
            print('  n', n, 'seed', s, time.strftime('%X'), flush=True)
    for k, v in out.items():
        print(f'  {k:44}{ms(v):>16}   {np.round(v,1)}')
    RES['E9'] = out
    save()


def E7():
    print('\n=== E7  reverse transfer, five vs seven swept parameters')
    out = {}
    Xb, yb = BASE_PHYS[PHYS].values, labels(BASE_PHYS, True)
    for s in SEEDS:
        for sweep in (False, True):
            g = build_param(50, seed=s, sweep_phase=sweep)
            if not sweep:
                g['phase_rot'], g['meas'] = 0, 'bus'
            P = post(pb.add_pu_per_run(pb.gen_physical(g)))
            Xc, yc = cap(P[PHYS].values, labels(P, True))
            p = forest().fit(Xc, yc).predict(Xb)
            out.setdefault('seven' if sweep else 'five', []).append(accuracy_score(yb, p) * 100)
    for k, v in out.items():
        print(f'  {k:8}{ms(v):>16}')
    RES['E7'] = out
    save()


if __name__ == '__main__':
    todo = sys.argv[1:] or ['E1', 'E2', 'E3', 'E4', 'E5']
    for e in todo:
        t = time.time()
        globals()[e]()
        print(f'  [{e} took {time.time()-t:.0f} s]', flush=True)
