import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from generator_v3 import F0, FS, N_CYC, run_case

CLASSES = ['LG', 'LL', 'LLG', 'LLL', 'LLLG', 'NF']


def pmu_stream(x, fs=FS, f0=F0):
    n_samples = len(x)
    window_size = int(fs / f0)
    kernel = np.exp(-2j * np.pi * np.arange(window_size) / window_size)
    windows = sliding_window_view(np.asarray(x, float), window_size)
    phasors = (2 / window_size) * (windows @ kernel)

    valid = slice(window_size - 1, None)
    phase_series = np.zeros(n_samples, complex)
    phase_series[valid] = phasors
    magnitude = np.full(n_samples, np.nan)
    angle = np.full(n_samples, np.nan)
    magnitude[valid] = np.abs(phasors) / np.sqrt(2)
    angle[valid] = np.angle(phasors, deg=True)

    frequency_offset = np.gradient(np.unwrap(np.angle(phase_series))) * fs / (2 * np.pi)
    return magnitude, angle, f0 + np.nan_to_num(frequency_offset)


def make_run(kind, **kw):
    case = run_case(kind, **kw)
    columns = {'time_stamp': case['t']}
    for index, phase in enumerate('abc'):
        columns['I' + phase] = case['I'][:, index]
        columns['V' + phase] = case['V'][:, index]

    columns['Im'], columns['Pai'], columns['Fi'] = pmu_stream(case['I'][:, 0])
    columns['Vm'], columns['Pav'], columns['Fv'] = pmu_stream(case['V'][:, 0])

    frame = pd.DataFrame(columns)
    frame['fault_types'] = kind
    frame['t_fault'] = case['t_fault']
    for name, value in kw.items():
        frame[name] = value
    return frame.iloc[N_CYC - 1:].reset_index(drop=True)


def build_param(n_per_class=60, seed=0, sweep_phase=True):
    rng = np.random.default_rng(seed)
    runs = []
    for kind in CLASSES:
        for _ in range(n_per_class):
            run = make_run(
                kind,
                m=rng.uniform(0.05, 0.95),
                Rf=float(rng.choice([0, 1, 5, 10, 25, 50], p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1])),
                theta0=rng.uniform(0, 2 * np.pi),
                scr=rng.uniform(3, 25),
                load=rng.uniform(0.3, 1.2),
                phase_rot=int(rng.integers(0, 3)) if sweep_phase else 0,
                meas=str(rng.choice(['bus', 'fault'])) if sweep_phase else 'bus',
            )
            runs.append(run)
    return pd.concat(runs, ignore_index=True)


build = build_param
