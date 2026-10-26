"""Liquid-liquid equilibria with the convex envelope method (CEM).

A mixture splits into liquid phases where the Gibbs energy of mixing
dg_mix / RT = sum_i x_i (ln x_i + ln gamma_i) lies above its lower convex envelope:
the envelope's facets that span non-neighboring grid points connect the coexisting
phases. The grid result is then refined to the root of the isoactivity equations.

Any number of components: `phase_split`, and `binary_split` for binaries. Their grids
match the paper's evaluation (1/128 for binary and ternary mixtures); a finer binary
grid reports numerical slivers on nearly ideal mixtures.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial import ConvexHull

LnGamma = Callable[[np.ndarray], np.ndarray]  # compositions [K, n] -> ln gamma [K, n]


def lower_hull_gaps(x1: np.ndarray, dg: np.ndarray) -> list[tuple[float, float]]:
    """Miscibility gaps (x1', x1'') of a binary from dg_mix sampled on the grid ``x1``."""
    try:
        hull = ConvexHull(np.column_stack([x1, dg]))
    except Exception:  # degenerate (e.g. collinear) point set: no gap
        return []
    lower = sorted({v for simplex, eq in zip(hull.simplices, hull.equations) if eq[1] < -1e-10 for v in simplex})
    return [(float(x1[a]), float(x1[b])) for a, b in zip(lower[:-1], lower[1:]) if b - a > 1]


def simplex_grid(n: int, m: int) -> np.ndarray:
    """All compositions of n components with mole fractions in steps of 1/m."""

    def compositions(k: int, total: int):
        if k == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for rest in compositions(k - 1, total - i):
                yield (i, *rest)

    return np.array(list(compositions(n, m)), dtype=float) / m


def dg_mix(X: np.ndarray, ln_gamma: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(X > 0, X * (np.log(np.where(X > 0, X, 1.0)) + ln_gamma), 0.0).sum(1)


def _lower_facets(X: np.ndarray, dg: np.ndarray):
    hull = ConvexHull(np.column_stack([X[:, :-1], dg]))
    lower = hull.equations[:, -2] < -1e-10
    return hull.simplices[lower]


def phase_split(ln_gamma: LnGamma, feed, m: int | None = None, refine: bool = True) -> np.ndarray | None:
    """Coexisting liquid phases ([P, n]) that the mixture ``feed`` splits into, or None.

    The facet of the lower convex envelope of dg_mix above ``feed`` has vertices in P
    groups of nearby grid points; each group is one phase, its composition the group's
    vertices weighted by the feed's barycentric coordinates. A facet within one group
    lies in a stable region. ``m`` is the grid
    resolution (default 128 up to three components, 32 for four, 12 beyond).
    """
    feed = np.asarray(feed, dtype=float) / np.sum(feed)
    n = feed.size
    m = m or {2: 128, 3: 128, 4: 32}.get(n, 12)
    X = simplex_grid(n, m)
    dg = dg_mix(X, ln_gamma(X))
    for simplex in _lower_facets(X, dg):
        V = X[simplex]
        try:  # barycentric coordinates of the feed in this facet
            w = np.linalg.solve(np.vstack([V.T[:-1], np.ones(len(V))]), np.append(feed[:-1], 1.0))
        except np.linalg.LinAlgError:
            continue
        if w.min() < -1e-9:
            continue
        # Same phase: closer than 3 grid steps, or than a quarter of the facet's extent (the
        # vertices of one phase can lie several steps apart along the binodal).
        extent = max(np.abs(a - b).max() for a in V for b in V)
        groups = _groups(V, max(3.0 / m, 0.25 * extent))
        phases = [(w[g, None] * V[g]).sum(0) / w[g].sum() for g in groups if w[g].sum() > 1e-12]
        if len(phases) < 2:
            return None
        phases = np.array(phases)
        return refine_split(ln_gamma, feed, phases) if refine else phases
    return None


def _groups(V: np.ndarray, distance: float) -> list[list[int]]:
    """Single-linkage groups of facet vertices closer than ``distance`` (max-norm)."""
    groups: list[list[int]] = []
    for i in range(len(V)):
        hits = [g for g in groups if any(np.abs(V[i] - V[j]).max() <= distance * 1.0001 for j in g)]
        groups = [g for g in groups if g not in hits] + [[i] + [j for g in hits for j in g]]
    return groups


def binary_split(ln_gamma: LnGamma, m: int = 128, refine: bool = True) -> tuple[float, float] | None:
    """(x1', x1'') of the widest miscibility gap of a binary, or None."""
    X = simplex_grid(2, m)  # x1 ascending
    gaps = lower_hull_gaps(X[:, 0], dg_mix(X, ln_gamma(X)))
    if not gaps:
        return None
    a, b = max(gaps, key=lambda g: g[1] - g[0])
    phases = np.array([[a, 1 - a], [b, 1 - b]])
    if refine:
        phases = refine_split(ln_gamma, phases.mean(axis=0), phases)
    return tuple(sorted((float(phases[0, 0]), float(phases[1, 0]))))


def refine_split(
    ln_gamma: LnGamma,
    feed,
    phases,
    tol: float = 1e-7,
    accept_tol: float = 1e-5,
    max_iter: int = 50,
    fd_step: float = 1e-4,
    max_du: float = 1.0,
) -> np.ndarray:
    """Newton-refine a P-phase split to the isothermal flash root; keep ``phases`` on failure.

    Unknowns u = ln x (P x n) and phase fractions beta (P). Equations: equal
    ln x_i + ln gamma_i in every phase, sum_i x_i = 1 per phase, sum_k beta_k x^(k) = feed.
    The Jacobian is a central finite difference with backtracking on the residual.
    """
    feed, X0 = np.asarray(feed, float), np.asarray(phases, float)
    P, n = X0.shape

    def lng(X):
        X = np.clip(X, 1e-300, None)
        return ln_gamma(X / X.sum(1, keepdims=True))

    def residual(u, beta, lg):
        x = np.exp(u)
        return np.concatenate([((u + lg) - (u[0] + lg[0]))[1:].ravel(), x.sum(1) - 1.0, beta @ x - feed])

    try:
        trace = X0 < 1e-10  # seed absent components from isoactivity instead of ln(1e-12)
        if trace.any():
            Xc = np.clip(X0, 1e-10, None)
            g = np.exp(lng(Xc))
            a = np.where(trace, 0.0, Xc * g).max(0)
            X0 = np.where(trace, np.clip(a / g, 1e-12, 1e-2), X0)
        X0 = np.clip(X0, 1e-12, None)
        X0 = X0 / X0.sum(1, keepdims=True)
        u, beta = np.log(X0), np.linalg.lstsq(X0.T, feed, rcond=None)[0]
        lg = lng(np.exp(u))
        F = residual(u, beta, lg)
        res = np.abs(F).max()
        for _ in range(max_iter):
            if res < tol:
                break
            pert = np.array(
                [np.exp(u[k] + s * fd_step * np.eye(n)[i]) for k in range(P) for i in range(n) for s in (1, -1)]
            )
            lg_p = lng(pert).reshape(P, n, 2, n)
            J = np.zeros((P * n + P, P * n + P))
            for k in range(P):
                for i in range(n):
                    cols = []
                    for s_idx, s in enumerate((1, -1)):
                        up, lp = u.copy(), lg.copy()
                        up[k, i] += s * fd_step
                        lp[k] = lg_p[k, i, s_idx]
                        cols.append(residual(up, beta, lp))
                    J[:, k * n + i] = (cols[0] - cols[1]) / (2 * fd_step)
            J[-n:, P * n :] = np.exp(u).T
            step = np.linalg.lstsq(J, -F, rcond=None)[0]
            du, db = step[: P * n].reshape(P, n), step[P * n :]
            scale = min(1.0, max_du / max(np.abs(du).max(), 1e-300))
            for _ in range(12):
                u_new, b_new = u + scale * du, beta + scale * db
                lg_new = lng(np.exp(u_new))
                F_new = residual(u_new, b_new, lg_new)
                if np.all(np.isfinite(F_new)) and np.abs(F_new).max() < res:
                    break
                scale *= 0.5
            else:
                break
            u, beta, lg, F = u_new, b_new, lg_new, F_new
            res = np.abs(F).max()
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return np.asarray(phases, float)
    X = np.exp(u)
    X = X / X.sum(1, keepdims=True)
    distinct = min(np.abs(X[a] - X[b]).max() for a in range(P) for b in range(a + 1, P)) >= 1e-4
    return X if res < accept_tol and distinct and beta.min() >= -1e-6 else np.asarray(phases, float)
