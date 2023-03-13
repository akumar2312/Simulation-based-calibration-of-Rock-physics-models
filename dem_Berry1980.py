"""
Adapted from:    
https://github.com/fvizeus/PyDEM

This DEM implementation is based on
Berryman 1980.
"""

import numpy as np
from scipy.optimize import fsolve


# DEM part
def fun_theta(alpha):
    return alpha * (np.arccos(alpha) - alpha * np.sqrt(1.0 - alpha * alpha)) / (1.0 - alpha * alpha) ** (3.0 / 2.0)


def fun_f(alpha, theta):
    return alpha * alpha * (3.0 * theta - 2.0) / (1.0 - alpha * alpha)


def shape_params(A, B, R, theta, f):
    F1 = 1.0 + A * (1.5 * (f + theta) - R * (1.5 * f + 2.5 * theta - 4.0 / 3.0))
    F2 = 1.0 + A * (1.0 + 1.5 * (f + theta) - R * (1.5 * f + 2.5 * theta)) + B * (3.0 - 4.0 * R) + A * (A + 3.0 * B) * (
                1.5 - 2.0 * R) * (f + theta - R * (f - theta + 2.0 * theta * theta))
    F3 = 1.0 + A * (1.0 - f - 1.5 * theta + R * (f + theta))
    F4 = 1.0 + (A / 4.0) * (f + 3.0 * theta - R * (f - theta))
    F5 = A * (-f + R * (f + theta - 4.0 / 3.0)) + B * theta * (3.0 - 4.0 * R)
    F6 = 1.0 + A * (1.0 + f - R * (f + theta)) + B * (1.0 - theta) * (3.0 - 4.0 * R)
    F7 = 2.0 + (A / 4.0) * (3.0 * f + 9.0 * theta - R * (3.0 * f + 5.0 * theta)) + B * theta * (3.0 - 4.0 * R)
    F8 = A * (1.0 - 2.0 * R + (f / 2.0) * (R - 1.0) + (theta / 2.0) * (5.0 * R - 3.0)) + B * (1.0 - theta) * (
                3.0 - 4.0 * R)
    F9 = A * ((R - 1.0) * f - R * theta) + B * theta * (3.0 - 4.0 * R)

    P = 3.0 * F1 / F2
    Q = 2.0 / F3 + 1.0 / F4 + (F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4)
    return P, Q


def elast(Km, Gm, Ki, Gi, ci, theta, f):
    A = Gi / Gm - 1.0
    B = (Ki / Km - Gi / Gm) / 3.0
    R = Gm / (Km + (4.0 / 3.0) * Gm)
    Fm = (Gm / 6.0) * (9.0 * Km + 8.0 * Gm) / (Km + 2.0 * Gm)
    P, Q = shape_params(A, B, R, theta, f)

    K = Km - (Km + (4.0 / 3.0) * Gm) * ci * (Km - Ki) * P / 3.0 / (Km + (4.0 / 3.0) * Gm + ci * (Km - Ki) * P / 3.0)
    G = Gm - (Gm + Fm) * ci * (Gm - Gi) * Q / 5.0 / (Gm + Fm + ci * (Gm - Gi) * Q / 5.0)

    return K, G


def solve_DEM(alphai, phii, Ki, Gi, Km=77.0, Gm=32.0, phi0=0.0, r=1000, phitol=1.0E-30, gamma=0.01):
    phi = np.sum(phii)
    fraci = phii / np.sum(phi)
    ci = fraci * alphai / r
    n = np.int(np.ceil((np.log(1.0 - phi) - np.log(1.0 - phi0)) / np.sum(np.log(1.0 - ci))))
    if n > 500:
        n = 500
    elif n < 100:
        n = 100
    m = len(alphai)

    def func(r):
        f = np.empty(m)
        f[0] = np.log(alphai[0] / r[0]) + np.log(1.0 - phi0 / phi) - np.log(
            1 - ((1.0 - phi) / (1.0 - phi0)) ** (1.0 / n))
        for j in range(1, m):
            f[j] = f[j - 1] + np.log(alphai[j] / r[j]) + np.log(r[j - 1] / alphai[j - 1] - fraci[j - 1])
        return f

    def fderiv(r):
        jac = np.diag(-1.0 / r)
        for j in range(0, m - 1):
            jac[j + 1:, j] = -1.0 / r[j] + 1.0 / (r[j] - fraci[j] * alphai[j])

        return jac

    r0 = r * np.ones(m)

    ri = fsolve(func, r0, fprime=fderiv, factor=0.1)

    ci = fraci * alphai / ri

    thetai = fun_theta(alphai)
    fi = fun_f(alphai, thetai)

    K = np.empty(n)
    G = np.empty(n)
    phi = np.empty(n)

    K_ = Km
    G_ = Gm
    phi_ = phi0

    for i in range(n):
        dphi = ci[0] * (1.0 - phi_)
        K_, G_ = elast(K_, G_, Ki[0], Gi[0], ci[0], thetai[0], fi[0])
        phi_ = phi_ + dphi
        for j in range(1, m):
            dphi = dphi * ci[j] * (1.0 - ci[j - 1]) / ci[j - 1]
            K_, G_ = elast(K_, G_, Ki[j], Gi[j], ci[j], thetai[j], fi[j])
            phi_ = phi_ + dphi
        K[i] = K_
        G[i] = G_
        phi[i] = phi_

    return K, G, phi


# Gassmann part
def Gassmann_Ks(Kd, Km, Kf, phi):
    gamma = 1.0 - phi - Kd / Km
    return Kd + (gamma + phi) ** 2 / (gamma / Km + phi / Kf)


def Gassmann_Kd(Ks, Km, Kf, phi):
    gamma = phi * (Km / Kf - 1.0)
    return (Ks * (gamma + 1.0) - Km) / (gamma - 1.0 + Ks / Km)
