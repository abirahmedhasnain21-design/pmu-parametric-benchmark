import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

F0, FS = 50.0, 3200.0
N_CYC = int(FS / F0)
A_OP = np.exp(2j * np.pi / 3)
A_INV = np.array([[1, 1, 1],
                  [1, A_OP, A_OP ** 2],
                  [1, A_OP ** 2, A_OP]]) / 3.0

CLASSES = ['LG', 'LL', 'LLG', 'LLL', 'LLLG', 'NF']

PHYS = ['I1_pu', 'V1_pu',
        'I2_I1_re', 'I2_I1_im', 'I0_I1_re', 'I0_I1_im',
        'V2_V1_re', 'V2_V1_im', 'V0_V1_re', 'V0_V1_im',
        'Z1_pu', 'Z1_ang']


def to_uniform(g, cols, fs=FS):
    t = g['time_stamp'].values.astype(float)
    keep = np.concatenate([[True], np.diff(t) > 0])
    t = t[keep]
    n = int(np.floor((t[-1] - t[0]) * fs)) + 1
    tu = t[0] + np.arange(n) / fs
    out = {'time_stamp': tu}
    for c in cols:
        out[c] = np.interp(tu, t, g[c].values.astype(float)[keep])
    return pd.DataFrame(out)


def sliding_phasor(x, n_cyc=N_CYC):
    k = np.exp(-2j * np.pi * np.arange(n_cyc) / n_cyc)
    Xk = (2.0 / n_cyc) * (sliding_window_view(np.asarray(x, float), n_cyc) @ k)
    ph = np.full(len(x), np.nan + 0j)
    ph[n_cyc - 1:] = Xk / np.sqrt(2)
    return ph


def sequence_components(Ia, Ib, Ic):
    P = np.vstack([sliding_phasor(Ia), sliding_phasor(Ib), sliding_phasor(Ic)])
    return A_INV @ P


def phys_features(run):
    I0, I1, I2 = sequence_components(run['Ia'].values, run['Ib'].values, run['Ic'].values)
    V0, V1, V2 = sequence_components(run['Va'].values, run['Vb'].values, run['Vc'].values)
    eps = 1e-12
    rI2, rI0 = I2 / (I1 + eps), I0 / (I1 + eps)
    rV2, rV0 = V2 / (V1 + eps), V0 / (V1 + eps)
    Z1 = V1 / (I1 + eps)
    f = pd.DataFrame({
        'time_stamp': run['time_stamp'].values,
        'I1_abs': np.abs(I1), 'V1_abs': np.abs(V1),
        'I2_I1_re': rI2.real, 'I2_I1_im': rI2.imag,
        'I0_I1_re': rI0.real, 'I0_I1_im': rI0.imag,
        'V2_V1_re': rV2.real, 'V2_V1_im': rV2.imag,
        'V0_V1_re': rV0.real, 'V0_V1_im': rV0.imag,
        'Z1_abs': np.abs(Z1), 'Z1_ang': np.angle(Z1, deg=True),
    })
    return f.iloc[N_CYC - 1:].reset_index(drop=True)


def add_pu(f, I_nom, V_nom):
    f = f.copy()
    f['I1_pu'] = f['I1_abs'] / I_nom
    f['V1_pu'] = f['V1_abs'] / V_nom
    f['Z1_pu'] = f['Z1_abs'] / (V_nom / I_nom)
    return f
