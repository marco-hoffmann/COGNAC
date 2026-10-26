"""User-facing access to the trained models: predictions and model insights.

from cognac import load_model
model = load_model("COGNAC")                        # or "COGNAC-IMP"
model.ln_gamma(["CCO", "O"], x=[0.3, 0.7], T=298.15)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq

from .chem import smiles_to_graph
from .data import batch_of
from .lle import binary_split, phase_split
from .model import COGNAC, State, reduced_to_raw_abc

CHECKPOINTS = Path(__file__).parent / "checkpoints"
MODELS = ("COGNAC", "COGNAC-IMP")
R_KJ = 0.008314  # kJ / (mol K)


def load_model(name: str = "COGNAC", device: str = "cpu") -> Predictor:
    """A trained model: ``name`` is "COGNAC", "COGNAC-IMP" or a path to a ``.pt`` file.

    The two named models are the paper's full-data models.
    """
    if name in MODELS:
        path = CHECKPOINTS / name / "full_data.pt"
    else:
        path = Path(name)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = COGNAC(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    return Predictor(model.to(device).eval(), name=str(name))


def _as_points(x, n: int) -> tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    x = np.atleast_2d(x)
    if x.shape[1] != n:
        raise ValueError(f"x has {x.shape[1]} mole fractions per point, the mixture {n} components")
    if np.any(x < 0) or np.any(np.abs(x.sum(1) - 1) > 1e-6):
        raise ValueError("mole fractions must be non-negative and sum to 1")
    return x, single


def _atom_labels(smiles: str) -> list[str]:
    """Element + 1-based node index, polar H with their parent atom: C1, C2, O3, H4(O3)."""
    graph = smiles_to_graph(smiles)
    heavy = [s for s in graph.symbols if s != "H"]
    labels = [f"{s}{i + 1}" for i, s in enumerate(heavy)]
    parents = {}
    src, dst = graph.edge_index
    for a, b in zip(src.tolist(), dst.tolist()):
        if graph.symbols[a] == "H" and a >= len(heavy):
            parents[a] = b
    return labels + [f"H{i + 1}({labels[parents[i]]})" for i in range(len(heavy), graph.num_nodes)]


class Predictor:
    """Wraps a `COGNAC` network with NumPy in / NumPy (or pandas) out.

    Compositions ``x`` are one point ``[n]`` or many ``[P, n]``; temperatures are
    in K, pressures in bar, enthalpies in kJ/mol.
    """

    def __init__(self, model: COGNAC, name: str = "COGNAC") -> None:
        self.model, self.name = model, name

    def __repr__(self) -> str:
        return f"Predictor({self.name}, residual={self.model.residual}, device={self.model.device})"

    def state(self, smiles: Sequence[str], T: float | Sequence[float]) -> State:
        """Network pass for one mixture at one or several temperatures."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        return self.model.prepare(batch_of([list(smiles)] * len(T)), torch.tensor(T, dtype=torch.float32))

    # ------------------------------------------------------------ predictions
    @torch.no_grad()
    def ln_gamma(self, smiles: Sequence[str], x, T: float) -> np.ndarray:
        """ln gamma_i of every component, shape like ``x``."""
        x, single = _as_points(x, len(smiles))
        out = self.model.ln_gamma(self.state(smiles, T), torch.tensor(x)[None])[0].cpu().numpy()
        return out[0] if single else out

    def ln_gamma_inf(self, solute: str, solvent: str, T: float) -> float:
        """ln gamma of ``solute`` infinitely diluted in ``solvent``."""
        return float(self.ln_gamma([solvent, solute], [1.0, 0.0], T)[1])

    def gE(self, smiles: Sequence[str], x, T: float) -> np.ndarray:
        """Excess Gibbs energy g^E / RT = sum_i x_i ln gamma_i."""
        x, single = _as_points(x, len(smiles))
        g = (x * self.ln_gamma(smiles, x, T)).sum(1)
        return g[0] if single else g

    def excess_enthalpy(self, smiles: Sequence[str], x, T: float) -> np.ndarray:
        """h^E = -R T^2 d(g^E/RT)/dT in kJ/mol, by automatic differentiation."""
        x, single = _as_points(x, len(smiles))
        out = []
        for chunk in np.array_split(x, max(1, len(x) // 256)):
            temperature = torch.full((len(chunk),), float(T), requires_grad=True)
            state = self.model.prepare(batch_of([list(smiles)] * len(chunk)), temperature)
            xt = torch.tensor(chunk, dtype=torch.float32, device=self.model.device)
            g = (xt * self.model.ln_gamma(state, xt)).sum(1)
            (dgdT,) = torch.autograd.grad(g.sum(), temperature)
            out.append((-R_KJ * float(T) ** 2 * dgdT).cpu().numpy())
        h = np.concatenate(out)
        return h[0] if single else h

    def vle(self, smiles: Sequence[str], x, T: float, ps_bar: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Bubble pressure p (bar) and vapor composition y by modified Raoult's law,
        p = sum_i x_i gamma_i ps_i, for given pure-component vapor pressures ``ps_bar``."""
        x, single = _as_points(x, len(smiles))
        partial = x * np.exp(self.ln_gamma(smiles, x, T)) * np.asarray(ps_bar, dtype=float)
        p = partial.sum(1)
        y = partial / p[:, None]
        return (p[0], y[0]) if single else (p, y)

    def lle(self, smiles: Sequence[str], T: float, feed=None) -> np.ndarray | None:
        """Coexisting liquid phases ``[P, n]`` at ``T``, or None if the mixture is stable.

        For a binary the widest miscibility gap is returned and ``feed`` is optional;
        for three or more components give the overall composition ``feed``.
        """
        with torch.no_grad():
            state = self.state(smiles, T)

        @torch.no_grad()
        def ln_gamma(X: np.ndarray) -> np.ndarray:  # in chunks: memory grows with atom pairs x points
            X = torch.tensor(X, dtype=torch.float32)
            return torch.cat([self.model.ln_gamma(state, c[None])[0].cpu() for c in X.split(2048)]).double().numpy()

        if len(smiles) == 2 and feed is None:
            split = binary_split(ln_gamma)
            return None if split is None else np.array([[split[0], 1 - split[0]], [split[1], 1 - split[1]]])
        if feed is None:
            raise ValueError("give the overall composition `feed` for three or more components")
        return phase_split(ln_gamma, feed)

    def sle(self, smiles: Sequence[str], T: float, Tm_K: float, hm_kJ_mol: float, solid: int = 0) -> float | None:
        """Solubility x of the pure solid ``smiles[solid]`` in a binary at ``T``:
        ln(x gamma) = -dh_m/R (1/T - 1/T_m). None above the melting point."""
        if T >= Tm_K:
            return None
        rhs = -hm_kJ_mol / R_KJ * (1.0 / T - 1.0 / Tm_K)
        with torch.no_grad():
            state = self.state(smiles, T)

        def f(xs: float) -> float:
            x = [xs, 1 - xs] if solid == 0 else [1 - xs, xs]
            with torch.no_grad():
                lg = self.model.ln_gamma(state, torch.tensor([x], dtype=torch.float32))[0, solid]
            return np.log(xs) + float(lg) - rhs

        try:
            return float(brentq(f, 1e-12, 1.0 - 1e-12, xtol=1e-10))
        except ValueError:  # no sign change: no solubility on this branch
            return None

    # --------------------------------------------------------------- insights
    @torch.no_grad()
    def atoms(self, smiles: str) -> pd.DataFrame:
        """Per atom of one molecule: cavity volume (Å^3) and area (Å^2), and the UNIFAC
        size parameters R_k, Q_k (relative to a CH2 group)."""
        s = self.state([smiles], 298.15)
        return pd.DataFrame(
            {
                "atom": _atom_labels(smiles),
                "volume_A3": s.volume.double().cpu().numpy(),
                "area_A2": s.area.double().cpu().numpy(),
                "R": s.R.double().cpu().numpy(),
                "Q": s.Q.double().cpu().numpy(),
            }
        )

    @torch.no_grad()
    def interactions(self, smiles: Sequence[str], T: float = 298.15) -> pd.DataFrame:
        """Every atom pair (k, m) of a mixture, k < m: mod. UNIFAC interaction parameters.

        ``A, B, C`` (K) are the model's polynomial a_km(T) = A + B tau + C tau^2 with
        tau = T/298.15 - 1; ``a, b, c`` the same polynomial in the published mod. UNIFAC
        form a + b T + c T^2; ``a_T`` its value at ``T`` and ``Psi_T = exp(-a_T / T)``.
        Pairs within one molecule enter the pure-component reference.
        """
        s = self.state(smiles, T)
        labels = [(i, label) for i, smi in enumerate(smiles) for label in _atom_labels(smi)]
        src, dst = s.pair_index.cpu().numpy()
        keep = src < dst
        src, dst = src[keep], dst[keep]
        A, B, C = s.coeffs[keep].cpu().double().numpy().T
        a, b, c = reduced_to_raw_abc(A, B, C)
        return pd.DataFrame(
            {
                "molecule_k": [labels[i][0] for i in src],
                "atom_k": [labels[i][1] for i in src],
                "molecule_m": [labels[i][0] for i in dst],
                "atom_m": [labels[i][1] for i in dst],
                "A": A,
                "B": B,
                "C": C,
                "a": a,
                "b": b,
                "c": c,
                "a_T": s.a[keep].double().cpu().numpy(),
                "Psi_T": np.exp(s.log_psi[keep].double().cpu().numpy()),
            }
        )

    @torch.no_grad()
    def unifac_table(self, smiles: Sequence[str], path: str | Path | None = None) -> dict:
        """The mixture as a mod. UNIFAC (Dortmund) parameter table; written to ``path`` as JSON if given.

        Every atom is one group (occurring once in its molecule). ``groups`` lists them with
        their size parameters R and Q; ``molecules`` names the groups of each component;
        ``a``, ``b``, ``c`` are the N x N interaction matrices over all groups, in K, -, 1/K,
        for Psi_mn = exp(-(a_mn + b_mn T + c_mn T^2) / T). With these, any mod. UNIFAC
        implementation (Dortmund combinatorial term, z = 10) reproduces COGNAC. COGNAC-IMP
        uses the same parameters in a self-consistent residual term, which a standard
        implementation does not have.
        """
        s = self.state(smiles, 298.15)  # the coefficients do not depend on T
        n = s.Q.numel()
        coeffs = np.zeros((3, n, n))
        src, dst = s.pair_index.cpu().numpy()
        coeffs[:, src, dst] = s.coeffs.double().cpu().numpy().T
        a, b, c = reduced_to_raw_abc(*coeffs)
        labels = [label for smi in smiles for label in _atom_labels(smi)]
        slot = s.node_slot.cpu().numpy()
        R, Q = s.R.double().cpu().numpy(), s.Q.double().cpu().numpy()
        table = {
            "model": self.name,
            "residual": self.model.residual,
            "form": "mod. UNIFAC (Dortmund): Psi_mn = exp(-(a_mn + b_mn*T + c_mn*T^2)/T), T in K",
            "molecules": [
                {"smiles": smi, "groups": [int(k) for k in np.flatnonzero(slot == i)]} for i, smi in enumerate(smiles)
            ],
            "groups": [
                {"id": k, "molecule": int(slot[k]), "atom": labels[k], "R": float(R[k]), "Q": float(Q[k])}
                for k in range(n)
            ],
            "a": a.tolist(),
            "b": b.tolist(),
            "c": c.tolist(),
        }
        if path is not None:
            Path(path).write_text(json.dumps(table, indent=1) + "\n")
        return table

    @torch.no_grad()
    def decompose(self, smiles: Sequence[str], x, T: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ln gamma at one composition split into its parts.

        Returns (components, atoms): per component ln gamma = combinatorial + residual;
        per atom k its residual contribution Q_k [ln Gamma_k(mix) - ln Gamma_k(pure)],
        which sum to each component's residual part.
        """
        x, _ = _as_points(x, len(smiles))
        s = self.state(smiles, T)
        total, parts = self.model.ln_gamma(s, torch.tensor(x[:1]), return_parts=True)
        components = pd.DataFrame(
            {
                "smiles": list(smiles),
                "x": x[0],
                "ln_gamma": total[0].cpu().numpy(),
                "combinatorial": parts["combinatorial"][0].cpu().numpy(),
                "residual": parts["residual"][0].cpu().numpy(),
            }
        )
        atoms = pd.DataFrame(
            {
                "molecule": s.node_slot.cpu().numpy(),
                "atom": [label for smi in smiles for label in _atom_labels(smi)],
                "theta": parts["theta"][:, 0].cpu().numpy(),
                "residual_contribution": parts["atom_residual"][:, 0].cpu().numpy(),
            }
        )
        return components, atoms
