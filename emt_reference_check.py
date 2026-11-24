# Compare the analytic generator with the Simscape EMT runs at the paper's
# reference operating point (m = 0.5, Rf = 1 ohm, SCR = 10, load = 1.0).
# Needs emt_<CLASS>.mat (Idata: N x 3 amps, tout: seconds) in this folder.
# Simscape fault block: Ron = 1 ohm per phase, Rg = 0.001 ohm, grounded-star load.
import numpy as np
from scipy.io import loadmat
import generator_v2
import generator_v3

CASE = dict(m=0.5, Rf=1.0, theta0=0.0, scr=10.0, load=1.0, dur=0.25, t_fault=0.06)
F0 = 50.0
CLASSES = ['LG', 'LL', 'LLG', 'LLL', 'LLLG', 'NF']


def rms_after(t, x, t0):
    start = t0 + 4 / F0                      # skip 4 cycles of dc transient
    window = (t >= start) & (t <= start + 4 / F0)
    return np.sqrt(np.mean(x[window] ** 2))


for name, gen in [('generator_v2', generator_v2), ('generator_v3', generator_v3)]:
    errors = []
    for cls in CLASSES:
        mat = loadmat(f'emt_{cls}.mat')
        t_emt = np.asarray(mat['tout']).ravel()
        i_emt = np.asarray(mat['Idata'])
        if i_emt.shape[0] != t_emt.size:
            i_emt = i_emt.T
        run = gen.run_case(cls, **CASE)
        for k in range(3):
            a = rms_after(run['t'], run['I'][:, k], CASE['t_fault'])
            e = rms_after(t_emt, i_emt[:, k], CASE['t_fault'])
            errors.append(100 * (a - e) / e)
    errors = np.abs(errors)
    print(f'{name}: mean |error| {errors.mean():.2f}%   max {errors.max():.2f}%')
