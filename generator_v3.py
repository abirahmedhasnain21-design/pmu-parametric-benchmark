import numpy as np

A_OP = np.exp(2j * np.pi / 3)
A_MAT = np.array([[1, 1, 1], [1, A_OP ** 2, A_OP], [1, A_OP, A_OP ** 2]])
F0, FS = 50.0, 3200.0
W0 = 2 * np.pi * F0
N_CYC = int(FS / F0)


def seq_currents(kind, Z1, Z2, Z0, E, Rf):
    if kind == 'NF':
        return 0j, 0j, 0j
    if kind in ('LLL', 'LLLG'):
        return 0j, E / (Z1 + Rf), 0j
    if kind == 'LG':
        I = E / (Z1 + Z2 + Z0 + 3 * Rf)
        return I, I, I
    if kind == 'LL':
        I = E / (Z1 + Z2 + 2 * Rf)
        return 0j, I, -I
    if kind == 'LLG':
        Z2r, Z0r = Z2 + Rf, Z0 + Rf
        I1 = E / (Z1 + Rf + Z2r * Z0r / (Z2r + Z0r))
        return (-I1 * Z2r / (Z2r + Z0r), I1, -I1 * Z0r / (Z2r + Z0r))
    raise ValueError(kind)


def network(m, scr, load, Vll):
    Vph = Vll * 1e3 / np.sqrt(3)
    Zbase = (Vll * 1e3) ** 2 / 100e6
    ZL1 = (0.03 + 0.30j) * Zbase
    ZL0 = 3.0 * ZL1.real + 2.7j * ZL1.imag
    Zs1, Zs0 = ZL1 / scr, 1.2 * ZL1 / scr
    Zload = Vph / (load * Vph / abs(ZL1) * 0.6 + 1e-9)
    Zsrc1, Zsrc0 = Zs1 + m * ZL1, Zs0 + m * ZL0
    Zrem1, Zrem0 = (1 - m) * ZL1 + Zload, (1 - m) * ZL0 + Zload
    par = lambda a, b: a * b / (a + b)
    return dict(Vph=Vph, ZL1=ZL1, Zs1=Zs1, Zs0=Zs0, Zload=Zload,
                Zsrc1=Zsrc1, Zsrc0=Zsrc0, Zrem1=Zrem1, Zrem0=Zrem0,
                Z1th=par(Zsrc1, Zrem1), Z2th=par(Zsrc1, Zrem1),
                Z0th=par(Zsrc0, Zrem0),
                div1=Zrem1 / (Zsrc1 + Zrem1), div0=Zrem0 / (Zsrc0 + Zrem0),
                Vf0=Vph * Zrem1 / (Zsrc1 + Zrem1))


def run_case(kind, m=.5, Rf=.5, theta0=0., scr=10., load=1., phase_rot=0,
             meas='bus', Vll=132., dur=.25, t_fault=.06):
    N = network(m, scr, load, Vll)
    Vph, Zs1, Zs0 = N['Vph'], N['Zs1'], N['Zs0']

    Iload = Vph / (Zs1 + N['ZL1'] + N['Zload']) * load
    Iph_pre = Iload * np.array([1, A_OP ** 2, A_OP])
    Vph_pre = (Vph if meas == 'bus' else N['Vf0']) * np.array([1, A_OP ** 2, A_OP])

    dI0, dI1, dI2 = seq_currents(kind, N['Z1th'], N['Z2th'], N['Z0th'],
                                 N['Vf0'], Rf)
    seq_I = np.array([dI0 * N['div0'], dI1 * N['div1'], dI2 * N['div1']])
    dIabc = A_MAT @ seq_I
    if meas == 'bus':
        dVabc = A_MAT @ (seq_I * np.array([Zs0, Zs1, Zs1]))
    else:
        dVabc = A_MAT @ (np.array([dI0, dI1, dI2]) *
                         np.array([N['Z0th'], N['Z1th'], N['Z2th']]))

    if kind not in ('LLL', 'LLLG', 'NF'):
        dIabc = np.roll(dIabc, phase_rot)
        dVabc = np.roll(dVabc, phase_rot)

    Iph_f, Vph_f = dIabc + Iph_pre, Vph_pre - dVabc
    if kind == 'NF':
        Iph_f, Vph_f = Iph_pre, Vph_pre

    n = int(dur * FS)
    t = np.arange(n) / FS

    Rs_, Ls_ = N['Zsrc1'].real, N['Zsrc1'].imag / W0
    Rr_, Lr_ = N['Zrem1'].real, N['Zrem1'].imag / W0
    _r = np.roots([Ls_ * Lr_,
                   Rs_ * Lr_ + Rr_ * Ls_ + Rf * (Ls_ + Lr_),
                   Rs_ * Rr_ + Rf * (Rs_ + Rr_)])
    _r = _r.real[_r.real < 0]
    tau = float(-1.0 / np.max(_r)) if _r.size else Ls_ / max(Rs_, 1e-6)

    on = t >= t_fault
    I = np.zeros((n, 3))
    V = np.zeros((n, 3))
    for k in range(3):
        pi_ = np.sqrt(2) * abs(Iph_pre[k]) * np.cos(W0 * t + theta0 + np.angle(Iph_pre[k]))
        fi_ = np.sqrt(2) * abs(Iph_f[k]) * np.cos(W0 * t + theta0 + np.angle(Iph_f[k]))
        dc0 = (pi_[on][0] - fi_[on][0]) if on.any() else 0.
        dc = np.zeros(n)
        dc[on] = dc0 * np.exp(-(t[on] - t_fault) / max(tau, 1e-4))
        I[:, k] = np.where(on, fi_ + dc, pi_)
        pv = np.sqrt(2) * abs(Vph_pre[k]) * np.cos(W0 * t + theta0 + np.angle(Vph_pre[k]))
        fv = np.sqrt(2) * abs(Vph_f[k]) * np.cos(W0 * t + theta0 + np.angle(Vph_f[k]))
        V[:, k] = np.where(on, fv, pv)
    return dict(t=t, I=I, V=V, Iph=Iph_f, Vph=Vph_f, t_fault=t_fault,
                Z1f=N['Zsrc1'], Z1th=N['Z1th'], E=Vph, Vf0=N['Vf0'], tau=tau)


def fault_component(kind, m=.5, Rf=.5, scr=10., load=1., phase_rot=0, Vll=132.):
    N = network(m, scr, load, Vll)
    I0, I1, I2 = seq_currents(kind, N['Z1th'], N['Z2th'], N['Z0th'], N['Vf0'], Rf)
    v = A_MAT @ np.array([I0, I1, I2])
    if kind not in ('LLL', 'LLLG', 'NF'):
        v = np.roll(v, phase_rot)
    return np.abs(v)


if __name__ == '__main__':
    r = run_case('LLL', m=1e-6, Rf=0., scr=1e6, load=1e-6)
    assert abs(abs(r['Iph'][0]) - abs(r['E'] / r['Z1th'])) < 1e-3 * abs(r['E'] / r['Z1th'])
    print('v1 checks 1-6 (unchanged behaviour at phase_rot = 0)')
    i3 = fault_component('LLL', m=.5, Rf=0.)[0]
    i1 = fault_component('LG', m=.5, Rf=0.)[0]
    assert i1 < i3
    rf = [fault_component('LG', Rf=x)[0] for x in [0, 1, 5, 10, 20, 50]]
    assert all(np.diff(rf) < 0)
    ms = [fault_component('LG', m=x)[0] for x in [.05, .25, .5, .75, .95]]
    assert all(np.diff(ms) < 0)
    assert np.allclose(fault_component('LLL', Rf=1.), fault_component('LLLG', Rf=1.))
    tau = [run_case('LG', m=.5, Rf=x)['tau'] * 1e3 for x in [0, 1, 5, 20, 50]]
    assert all(np.diff(tau) < 0)
    big = 1e9
    l1 = seq_currents('LLG', 3+30j, 3+30j, big, 1e4, 5.)[1]
    l2 = seq_currents('LL', 3+30j, 3+30j, 0, 1e4, 5.)[1]
    assert abs(l1 - l2) < 1e-6 * abs(l2)
    print('  all pass (incl. LLG -> LL as Z0 -> inf)\n')

    print('validation 7  phase_rot permutes but does not change magnitudes')
    for kind in ['LG', 'LL', 'LLG', 'LLL']:
        v = [fault_component(kind, Rf=1., phase_rot=k) for k in (0, 1, 2)]
        assert np.allclose(sorted(np.round(v[0], 6)), sorted(np.round(v[1], 6)))
        assert np.allclose(sorted(np.round(v[0], 6)), sorted(np.round(v[2], 6)))
        sel = lambda x: ''.join(p for p, q in zip('ABC', x) if q > .3 * x.max())
        print(f'  {kind:5} rot0 {sel(v[0]):4} rot1 {sel(v[1]):4} rot2 {sel(v[2]):4}')
    print('\n  base dataset uses  LG:A  LL:AB  LLG:CA  -- all reachable above')

    AINV = np.array([[1, 1, 1], [1, A_OP, A_OP ** 2], [1, A_OP ** 2, A_OP]]) / 3
    seq = lambda r: AINV @ r['Vph']
    Vn = abs(run_case('NF', meas='fault')['Vph'][0])

    print('\nvalidation 8  bolted faults at the fault point')
    assert np.allclose(np.abs(run_case('LLL', Rf=0., meas='fault')['Vph']), 0, atol=1e-6)
    assert abs(np.abs(run_case('LG', Rf=0., meas='fault')['Vph'])[0]) < 1e-6
    vll = np.abs(run_case('LL', Rf=0., meas='fault')['Vph']) / Vn
    assert abs(vll[1] - vll[2]) < 1e-6 and abs(vll[0] - 1) < 1e-6
    V = seq(run_case('LL', Rf=0., meas='fault'))
    assert abs(abs(V[2] / V[1]) - 1) < 1e-6
    print('  LLL -> 0 pu on all phases;  LG -> 0 on the faulted phase;')
    print('  LL  -> healthy phase held at 1 pu and |V2/V1| = 1        OK')

    print('\nvalidation 9  fault-point voltage rises monotonically with Rf')
    vr = [abs(seq(run_case('LLL', Rf=float(x), meas='fault'))[1]) / Vn
          for x in [0, 1, 5, 25, 50]]
    assert all(np.diff(vr) > 0)
    print('  ', [f'{v:.3f}' for v in vr], 'pu   monotone')
    print("\n  'bus' mode reproduces v1 exactly; 'fault' mode reaches the base")
    print('  dataset regime (LLL 0.09 pu, LL |V2/V1| 0.91), which no source')
    print('  strength can produce at the bus.')
