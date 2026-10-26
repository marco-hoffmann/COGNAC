"""mod. UNIFAC (Dortmund) from an exported parameter table, in plain NumPy.

    from cognac.unifac import excess_enthalpy, ln_gamma
    ln_gamma("ethanol_water.json", x=[0.3, 0.7], T=298.15)
    excess_enthalpy("ethanol_water.json", x=[0.5, 0.5], T=298.15)   # kJ/mol

The table comes from ``Predictor.unifac_table`` (a dict or its JSON file). These are
the textbook equations with no neural network involved; for COGNAC they reproduce the
model to float precision. The file depends on NumPy only and can be copied on its own.

    ln gamma_i   = ln gamma_i^C + ln gamma_i^R
    ln gamma_i^C = 1 - V'_i + ln V'_i - 5 q_i (1 - V_i/F_i + ln(V_i/F_i))
                   V_i = r_i / sum_j x_j r_j,  V'_i = r_i^(3/4) / sum_j x_j r_j^(3/4),  F_i = q_i / sum_j x_j q_j
    ln gamma_i^R = sum_k nu_ki [ln Gamma_k - ln Gamma_k^(i)]
    ln Gamma_k   = Q_k [1 - ln(sum_m theta_m Psi_mk) - sum_m theta_m Psi_km / sum_n theta_n Psi_nm]
    Psi_mn       = exp(-(a_mn + b_mn T + c_mn T^2) / T)
    h^E          = -R T^2 sum_i x_i d(ln gamma_i^R)/dT
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: kJ/(mol K); the value COGNAC was trained with, so h^E here matches the model's.
R_GAS = 0.008314


def load_table(table: dict | str | Path) -> dict:
    """A parameter table as a dict; strings and paths are read as JSON files."""
    return table if isinstance(table, dict) else json.loads(Path(table).read_text())


def _parameters(table: dict | str | Path):
    table = load_table(table)
    R = np.array([g["R"] for g in table["groups"]])
    Q = np.array([g["Q"] for g in table["groups"]])
    nu = np.zeros((len(table["molecules"]), len(R)))
    for i, molecule in enumerate(table["molecules"]):
        nu[i, molecule["groups"]] = 1.0
    a, b, c = (np.array(table[k], dtype=float) for k in ("a", "b", "c"))
    return R, Q, nu, a, b, c


def _points(x) -> tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=float)
    return np.atleast_2d(x), x.ndim == 1


def _residual(Q, nu, a, b, c, x: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray]:
    """Residual ln gamma_i and its temperature derivative, each [P, n]."""
    psi = np.exp(-(a + b * T + c * T**2) / T)
    dpsi = psi * (a / T**2 - c)  # dPsi/dT

    def ln_Gamma(groups: np.ndarray, X: np.ndarray):  # X: [P, len(groups)] group mole fractions
        theta = Q[groups] * X
        theta /= theta.sum(axis=1, keepdims=True)
        p, dp = psi[np.ix_(groups, groups)], dpsi[np.ix_(groups, groups)]
        S, dS = theta @ p, theta @ dp
        value = Q[groups] * (1.0 - np.log(S) - (theta / S) @ p.T)
        slope = Q[groups] * (-dS / S - (theta / S) @ dp.T + (theta * dS / S**2) @ p.T)
        return value, slope

    X_mix = x @ nu
    ln_mix, d_mix = ln_Gamma(np.arange(Q.size), X_mix / X_mix.sum(axis=1, keepdims=True))
    ln_r, d_r = np.empty_like(x), np.empty_like(x)
    for i in range(len(nu)):
        own = np.flatnonzero(nu[i])
        ln_pure, d_pure = ln_Gamma(own, np.full((1, len(own)), 1.0 / len(own)))
        ln_r[:, i] = (ln_mix[:, own] - ln_pure).sum(axis=1)
        d_r[:, i] = (d_mix[:, own] - d_pure).sum(axis=1)
    return ln_r, d_r


def ln_gamma(table: dict | str | Path, x, T: float) -> np.ndarray:
    """ln gamma of every component at mole fractions ``x`` ([n] or [P, n]) and ``T`` in K."""
    R, Q, nu, a, b, c = _parameters(table)
    x, single = _points(x)
    r, q = nu @ R, nu @ Q
    V, F = r / (x @ r)[:, None], q / (x @ q)[:, None]
    V_dash = r**0.75 / (x @ r**0.75)[:, None]
    ln_c = 1.0 - V_dash + np.log(V_dash) - 5.0 * q * (1.0 - V / F + np.log(V / F))
    out = ln_c + _residual(Q, nu, a, b, c, x, T)[0]
    return out[0] if single else out


def excess_enthalpy(table: dict | str | Path, x, T: float) -> np.ndarray:
    """h^E in kJ/mol: h^E = -R T^2 sum_i x_i d(ln gamma_i^R)/dT, with
    d ln Gamma_k/dT = Q_k [-S'_k/S_k - sum_m theta_m (Psi'_km S_m - Psi_km S'_m) / S_m^2],
    S_k = sum_m theta_m Psi_mk, Psi'_mn = Psi_mn (a_mn/T^2 - c_mn). The combinatorial part
    does not depend on T."""
    _, Q, nu, a, b, c = _parameters(table)
    x, single = _points(x)
    h = -R_GAS * T**2 * (x * _residual(Q, nu, a, b, c, x, T)[1]).sum(axis=1)
    return h[0] if single else h
