# -*- coding: utf-8 -*-
"""
Core optical functions for zone plate simulation.
"""
import numpy as np
from scipy.special import j0
from scipy.integrate import trapezoid
from parameters import *


def HT(x, f, R, dr, kr):
    """Compute the Hankel transform."""
    x_c = x * np.exp(0j)
    N = len(x)
    ht = np.zeros(N, dtype=np.complex128)  # blank array to calculate result onto
    for i in range(N):
        jav = j0(kr[i] * x) * np.exp(0j)  # Bessel function
        ht[i] = trapezoid(f * jav * x_c, dx=dr)  # integral
    return ht


def iHT(x, f, R, dr, kr):
    """Compute the inverse Hankel transform."""
    kr_c = kr * np.exp(0j)
    N = len(x)
    dkr = kr[1] - kr[0]
    ht = np.zeros(N // fac, dtype=np.complex128)
    for i in range(N // fac):
        jav = j0(kr * x[i * fac]) * np.exp(0j)
        ht[i] = trapezoid(f * jav * kr_c, dx=dkr)
    return ht


def prop(x, HTr, z, lamda, dr, kr):
    """Propagate a field using the angular spectrum method."""
    k = 2 * np.pi / lamda  # wavenumber
    mask = kr <= k
    exp_fac = np.zeros_like(kr, dtype=np.complex128)
    exp_fac[mask] = np.exp(1j * z * np.sqrt(k**2 - kr[mask] ** 2))
    exp_fac[~mask] = 0
    return HTr * exp_fac


def ZP_cD(x, NZ, R, lamda):
    """Create a binary zone plate."""
    f = 1.4e-3
    h = np.zeros(len(x), dtype=np.complex128)
    r = [np.sqrt(j * lamda * f + 0.25 * (j * lamda) ** 2) for j in range(NZ + 1)]
    r = np.array(r)
    r = (R) * r / np.max(r)
    # print(r)
    for i in range(0, len(r), 2):
        for q in range(len(x)):
            if r[i - 1] <= abs(x[q]) <= r[i]:
                h[q] = 1

    fz = ((R) ** 2 - (NZ * lamda / 2) ** 2) / (NZ * lamda)
    return h, r, fz


def Gauss_spec(wav, wav_u, dlamda):
    """Gaussian spectral distribution."""
    return np.exp(-((wav - wav_u) ** 2) / (2 * (dlamda) ** 2))


def ZP_cD_ph(x, NZ, R, lamda):
    """Create a phase zone plate with attenuation."""
    f = 1.4e-3
    h = np.ones(len(x), dtype=np.complex128)
    r = [np.sqrt(j * lamda * f + 0.25 * (j * lamda) ** 2) for j in range(NZ + 1)]
    r = np.array(r)
    r = (R) * r / np.max(r)
    for i in range(1, len(r), 2):
        for q in range(len(x)):
            if r[i - 1] <= abs(x[q]) <= r[i]:
                h[q] -= 0.05
                h[q] *= np.exp(-1j * np.pi)
    ap1 = x < r[-1]
    ap2 = x > r[1]
    fz = ((R) ** 2 - (NZ * lamda / 2) ** 2) / (NZ * lamda)
    return h * ap1 * ap2, r, fz


def mtf(f, fc):
    """Analytical modulation transfer function."""
    # MTF calculation for f <= fc
    if abs(f) <= fc:
        return (
            2
            * (np.arccos(abs(f) / fc) - (abs(f) / fc) * np.sqrt(1 - (abs(f) / fc) ** 2))
            / np.pi
        )
    # MTF is 0 for f > fc
    else:
        return 0


def MTF_curve(freqs, fc):
    """Evaluate the MTF over an array of frequencies."""
    mtf_values = [mtf(f, fc) for f in freqs]
    return mtf_values


def rect(x):
    """Rectangle function."""
    return abs(x) <= 1 / 2


def circ(r):
    """Circle function."""
    return abs(r) <= 1


def sort(n, t):
    """Distribute tasks across processors."""
    """
    n - number of processors
    t - number of tasks
    Sort the number of tasks.
    """
    td = t // n  # the number of tasks per processor
    r = t % n  # should be the leftover tasks
    l = td * np.ones(n)  # make an array for the allegation of tasks
    for i in range(r):  # for loop to add the extra tasks to some processors
        l[i] = l[i] + 1
    return l
