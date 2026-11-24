import os
import numpy as np, pandas as pd
from scipy.io import loadmat
from generator_v3 import run_case

F0 = 50.0
CLASSES = ['LG', 'LL', 'LLG', 'LLL', 'LLLG']
GRID_M = [0.05, 0.5, 0.95]
GRID_RF = [0, 10, 50]
BASE = dict(theta0=0., scr=10., load=1., dur=.25, t_fault=.06)


def rms_after(t, x, t0, skip_cycles=4):
    lo = t0 + skip_cycles / F0
    w = (t >= lo) & (t <= lo + 4 / F0)
    return float(np.sqrt(np.mean(x[w] ** 2)))


def dc_tau(t, x, t0):
    N = max(8, int(round((1 / F0) / np.median(np.diff(t)))))
    off = pd.Series(x).rolling(N, center=True).mean().values
    w = (t > t0 + .002) & (t < t0 + .12) & np.isfinite(off)
    tt, oo = t[w] - t0, np.abs(off[w])
    keep = oo > .02 * np.nanmax(oo)
    if keep.sum() < 10:
        return np.nan
    s = np.polyfit(tt[keep], np.log(oo[keep]), 1)[0]
    return float(-1 / s) if s < 0 else np.nan


rows = []
for cls in CLASSES:
    for m in GRID_M:
        for rf in GRID_RF:
            f = f'emt_{cls}_m{m}_Rf{rf}.mat'
            if not os.path.exists(f):
                print('missing', f)
                continue
            M = loadmat(f)
            te = np.asarray(M['tout']).ravel()
            Ie = np.asarray(M['Idata'])
            if Ie.shape[0] != te.size:
                Ie = Ie.T
            r = run_case(cls, m=m, Rf=float(rf), **BASE)
            t0 = BASE['t_fault']
            for k, ph in enumerate('abc'):
                a, e = rms_after(r['t'], r['I'][:, k], t0), rms_after(te, Ie[:, k], t0)
                rows.append(dict(cls=cls, m=m, Rf=rf, phase=ph, analytic=a, emt=e,
                                 diff_pct=100 * (a - e) / e))
            j = int(np.argmax([np.ptp(Ie[te > t0, k]) for k in range(3)]))
            rows[-3 + j]['tau_analytic_ms'] = 1e3 * dc_tau(r['t'], r['I'][:, j], t0)
            rows[-3 + j]['tau_emt_ms'] = 1e3 * dc_tau(te, Ie[:, j], t0)

R = pd.DataFrame(rows)
if len(R):
    R['faulted'] = R.emt > 1.5 * R.groupby(['cls', 'm', 'Rf']).emt.transform('min')
    print(R.pivot_table(index=['cls', 'm'], columns='Rf', values='diff_pct',
                        aggfunc=lambda v: np.abs(v).max()).round(2))
    print(f'\nall phases : mean |diff| {R.diff_pct.abs().mean():.2f}%  '
          f'max {R.diff_pct.abs().max():.2f}%')
    if 'tau_emt_ms' in R:
        T = R.dropna(subset=['tau_emt_ms'])
        print('\ndc time constant (ms), worst phase')
        print(T[['cls', 'm', 'Rf', 'tau_analytic_ms', 'tau_emt_ms']]
              .to_string(index=False, float_format=lambda v: f'{v:7.2f}'))
    R.to_csv('emt_grid.csv', index=False)
    print('\nsaved emt_grid.csv')
