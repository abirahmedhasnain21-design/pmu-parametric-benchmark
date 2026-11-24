import numpy as np, pandas as pd, time, os
from partF_core import (F0, FS, N_CYC, CLASSES, PHYS, to_uniform,
                        phys_features, add_pu)

WAVE  = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
PAPER = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc', 'Fi', 'Fv', 'Pai']
ANG   = ['Pai', 'Pav']
RS = 42


def load_base(path=None):
    if path is None:
        import os
        for p in ['PMU_Fault_Data.csv', 'data/PMU_Fault_Data.csv',
                  '/content/PMU_Fault_Data.csv', '/mnt/project/PMU_Fault_Data.csv']:
            if os.path.exists(p):
                path = p
                break
        else:
            raise FileNotFoundError(
                'PMU_Fault_Data.csv not found. Put it next to this notebook, '
                'or call load_base(path=...) with an explicit path.')
    raw = pd.read_csv(path)
    d = raw.drop_duplicates().dropna().reset_index(drop=True)
    d = d[~((d.Vm == 1.0) & (d.Im == 1.0))].reset_index(drop=True)
    return d


def base_physical(d):
    out = []
    for cls in CLASSES:
        g = d[d.fault_types == cls].sort_values('time_stamp')
        u = to_uniform(g, WAVE + ['Fi', 'Fv'])
        idx = np.searchsorted(g.time_stamp.values, u.time_stamp.values).clip(0, len(g)-1)
        for c in ANG:
            u[c] = g[c].values[idx]
        f = phys_features(u)
        for c in PAPER:
            f[c] = u[c].values[N_CYC-1:]
        f['fault_types'] = cls
        f['run_id'] = f'base_{cls}'
        f['t_fault'] = -np.inf
        out.append(f)
    return pd.concat(out, ignore_index=True)


def gen_physical(gen):
    gen = gen.copy()
    gen['run_id'] = (gen.time_stamp.diff().fillna(0) < 0).cumsum()
    out = []
    for rid, g in gen.groupby('run_id'):
        g = g.reset_index(drop=True)
        f = phys_features(g)
        for c in PAPER:
            f[c] = g[c].values[N_CYC-1:]
        f['fault_types'] = g.fault_types.iloc[0]
        f['run_id'] = f'gen_{rid}'
        f['t_fault'] = g.t_fault.iloc[0]
        for p in ['m', 'Rf', 'theta0', 'scr', 'load', 'phase_rot', 'meas']:
            if p in g:
                f[p] = g[p].iloc[0]
        out.append(f)
    return pd.concat(out, ignore_index=True)


def nominals(f):
    nf = f[f.fault_types == 'NF']
    return float(np.median(nf.I1_abs)), float(np.median(nf.V1_abs))


def add_pu_per_run(F):
    In_d, Vn_d = nominals(F)
    out = []
    for rid, g in F.groupby('run_id', sort=False):
        pre = g[g.time_stamp < g.t_fault]
        In = float(np.median(pre.I1_abs)) if len(pre) > 20 else In_d
        Vn = float(np.median(pre.V1_abs)) if len(pre) > 20 else Vn_d
        g = g.copy()
        g['I1_pu'] = g.I1_abs / In
        g['V1_pu'] = g.V1_abs / Vn
        g['Z1_pu'] = g.Z1_abs / (Vn / In)
        out.append(g)
    return pd.concat(out, ignore_index=True)


if __name__ == '__main__':
    t0 = time.time()
    d = load_base()
    print('base rows after cleaning :', len(d))
    B = base_physical(d)
    print('base physical            :', B.shape, '%.1f s' % (time.time() - t0))

    import gen_build
    gen = gen_build.build(n_per_class=60, seed=0)
    G = gen_physical(gen)
    print('parametric physical      :', G.shape, G.run_id.nunique(), 'runs')

    print('nominals  base  I=%.1f V=%.1f  (no pre-fault window: NF run used)'
          % nominals(B))
    B = add_pu_per_run(B)
    G = add_pu_per_run(G)

    B.to_parquet('base_phys.parquet') if False else B.to_pickle('base_phys.pkl')
    G.to_pickle('gen_phys.pkl')
    print('\nsanity — mean physical features per class')
    for tag, F in [('BASE', B), ('GEN(post-fault)', G[G.time_stamp >= G.t_fault])]:
        print('\n', tag)
        print(F.groupby('fault_types')[PHYS].mean().round(3).to_string())
    print('\ntotal %.1f s' % (time.time() - t0))
