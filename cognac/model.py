"""COGNAC: a graph neural network that predicts mod. UNIFAC (Dortmund) parameters per atom.

Every atom is a UNIFAC "group". A message-passing network over the molecular graph
gives each atom an embedding that depends on its neighborhood. Two heads read it:

* the geometry head gives the atom's cavity volume and surface area, which become
  the UNIFAC size parameters R_k = V_k / V_ref and Q_k = A_k / A_ref;
* the interaction head gives, for every pair of atoms (k, m), the coefficients of the
  mod. UNIFAC temperature polynomial a_km(T) = A + B tau + C tau^2,
  tau = T / 298.15 K - 1.

These are evaluated in the unchanged mod. UNIFAC equations: ln gamma = combinatorial
(Dortmund r^(3/4) form) + residual (group activity coefficients). The residual
closure is either the closed-form UNIFAC sum (``residual="explicit"``, COGNAC) or
its self-consistent COSMOSPACE fixed point (``residual="implicit"``, COGNAC-IMP).

`prepare` runs the network once per mixture and temperature; `ln_gamma` then
evaluates the thermodynamics at any composition, which is what makes phase
equilibrium calculations cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chem import DOMAIN_ATOMIC_NUMBERS, DOMAIN_ELEMENTS, Batch

#: Cavity volume (Å^3) and area (Å^2) of a CH2 group in a long alkane (COSMO, CHAOS).
V_REF = 20.18
A_REF = 21.17
#: Reference temperature of the reduced temperature basis tau = T / T_REF - 1.
T_REF = 298.15
#: The exponent |a/T| of Psi = exp(-a/T) is the identity up to KNEE and saturates
#: smoothly towards EXP_MAX above it, so it stays bounded with a non-zero gradient.
EXP_KNEE = 40.0
EXP_MAX = 60.0
#: UNIFAC coordination number.
Z = 10.0
EPS = 1e-8


def scatter_sum(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    return src.new_zeros((size, *src.shape[1:])).index_add_(0, index, src)


def scatter_logsumexp(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """log(sum(exp(src))) over rows with equal *index*; ``src`` is [E, P]."""
    expanded = index.view(-1, 1).expand_as(src)
    m = src.new_full((size, src.shape[1]), -torch.inf).scatter_reduce(0, expanded, src, "amax")
    m = m.detach().nan_to_num(neginf=0.0)
    return m + torch.log(scatter_sum(torch.exp(src - m[index]), index, size) + EPS)


def reduced_to_raw_abc(A, B, C, t_ref: float = T_REF):
    """(A, B, C) of a + b T + c T^2 written as A + B tau + C tau^2 -> published (a, b, c)."""
    return A - B + C, (B - 2.0 * C) / t_ref, C / t_ref**2


class AtomEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.element = nn.Embedding(len(DOMAIN_ATOMIC_NUMBERS), hidden)
        row = torch.full((119,), -1, dtype=torch.long)
        row[list(DOMAIN_ATOMIC_NUMBERS)] = torch.arange(len(DOMAIN_ATOMIC_NUMBERS))
        self.register_buffer("element_row", row, persistent=False)
        self.degree = nn.Embedding(8, hidden)
        self.formal_charge = nn.Embedding(9, hidden)
        self.hybridization = nn.Embedding(6, hidden)
        self.aromatic = nn.Embedding(2, hidden)
        self.num_h = nn.Embedding(6, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = self.element_row[x[:, 0]]
        if bool((rows < 0).any()):
            raise ValueError(f"Element outside the model domain ({', '.join(DOMAIN_ELEMENTS)}).")
        h = (
            self.element(rows)
            + self.degree(x[:, 1])
            + self.formal_charge(x[:, 2])
            + self.hybridization(x[:, 3])
            + self.aromatic(x[:, 4])
            + self.num_h(x[:, 5])
        )
        return self.norm(h)


class BondEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.bond_type = nn.Embedding(5, hidden)
        self.conjugated = nn.Embedding(2, hidden)
        self.in_ring = nn.Embedding(2, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.norm(self.bond_type(e[:, 0]) + self.conjugated(e[:, 1]) + self.in_ring(e[:, 2]))


def mlp(*dims: int, dropout: float | None = 0.0) -> nn.Sequential:
    """Linear-SiLU stack, with dropout after the first activation unless ``dropout`` is None."""
    layers: list[nn.Module] = []
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(d_in, d_out))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
            if i == 0 and dropout is not None:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MessagePassing(nn.Module):
    """h_i <- LayerNorm(h_i + upd(h_i, sum_j msg(h_i, h_j, e_ij))) over bonded neighbors j."""

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.msg = mlp(3 * hidden, hidden, hidden, dropout=dropout)
        self.upd = mlp(2 * hidden, hidden, hidden, dropout=dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return self.norm(h)
        src, dst = edge_index
        m = scatter_sum(self.msg(torch.cat([h[dst], h[src], e], dim=-1)), dst, h.shape[0])
        return self.norm(h + self.upd(torch.cat([h, m], dim=-1)))


@dataclass
class State:
    """The temperature- and molecule-dependent part of a batch: everything except x."""

    n_mixtures: int
    n_components: int
    node_mixture: torch.Tensor  # [N] mixture of each atom
    node_slot: torch.Tensor  # [N] component position of each atom
    node_component: torch.Tensor  # [N] molecule of each atom
    component_mixture: torch.Tensor
    component_slot: torch.Tensor
    R: torch.Tensor  # [N] atom volume parameter
    Q: torch.Tensor  # [N] atom area parameter
    r: torch.Tensor  # [B, n] molecule volume parameter
    q: torch.Tensor  # [B, n] molecule area parameter
    n_atoms: torch.Tensor  # [B, n] atoms per molecule
    pair_index: torch.Tensor  # [2, E] all atom pairs of a mixture
    coeffs: torch.Tensor  # [E, 3] (A, B, C) per pair, a_kk = 0
    a: torch.Tensor  # [E] a_km(T) in K
    log_psi: torch.Tensor  # [E] ln Psi_km = -a_km / T
    ln_Gamma_pure: torch.Tensor  # [N, 1] group activity coefficient in the pure molecule
    volume: torch.Tensor  # [N] atom cavity volume (Å^3)
    area: torch.Tensor  # [N] atom cavity area (Å^2)


class COGNAC(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        layers: int = 4,
        dropout: float = 0.0,
        residual: str = "explicit",
        solver_damping: float = 0.7,
        solver_tol: float = 5e-3,
        solver_max_iter: int = 25,
        solver_grad_steps: int = 10,
    ) -> None:
        super().__init__()
        if residual not in ("explicit", "implicit"):
            raise ValueError("residual must be 'explicit' or 'implicit'")
        self.residual = residual
        self.solver_damping = float(solver_damping)
        self.solver_tol = float(solver_tol)
        self.solver_max_iter = int(solver_max_iter)
        self.solver_grad_steps = int(solver_grad_steps)

        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        self.layers = nn.ModuleList(MessagePassing(hidden_dim, dropout) for _ in range(layers))
        self.geometry_head = mlp(hidden_dim, hidden_dim, hidden_dim // 2, 2, dropout=None)
        self.interaction_head = mlp(3 * hidden_dim, hidden_dim, hidden_dim // 2, 3, dropout=dropout)
        # Start temperature-independent (B = C = 0); the data adds T-dependence where needed.
        with torch.no_grad():
            self.interaction_head[-1].weight[1:].zero_()
            self.interaction_head[-1].bias[1:].zero_()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ----------------------------------------------------------------- network
    def interaction_coefficients(self, h: torch.Tensor, pair_index: torch.Tensor) -> torch.Tensor:
        """(A, B, C) per atom pair, symmetric and zero for k = m.

        The head is symmetric in its two atoms. Subtracting half of each atom's self
        term, c_km - (c_kk + c_mm) / 2, makes a_kk = 0 (Psi_kk = 1) at every T.
        Every node must appear once as a self-pair in ``pair_index``.
        """
        src, dst = pair_index
        hs, hd = h[src], h[dst]
        c = self.interaction_head(torch.cat([hs + hd, (hs - hd).abs(), hs * hd], dim=-1))
        self_pair = src == dst
        c_self = c.new_zeros((h.shape[0], 3))
        c_self[src[self_pair]] = c[self_pair]
        return c - 0.5 * (c_self[src] + c_self[dst])

    def prepare(self, batch: Batch, temperature: torch.Tensor) -> State:
        """Run the network for a batch of mixtures at temperatures ``[B]`` (K)."""
        batch = batch.to(self.device)
        temperature = temperature.to(self.device, torch.float32)
        B, n = batch.n_mixtures, batch.n_components
        n_comp = B * n

        h = self.atom_encoder(batch.x)
        e = self.bond_encoder(batch.edge_attr)
        for layer in self.layers:
            h = layer(h, batch.edge_index, e)

        raw = self.geometry_head(h)
        volume = F.softplus(raw[:, 0]) + EPS
        area = F.softplus(raw[:, 1]) + EPS
        R, Q = volume / V_REF, area / A_REF

        def per_molecule(values: torch.Tensor) -> torch.Tensor:
            out = values.new_zeros(B, n)
            out[batch.component_mixture, batch.component_slot] = scatter_sum(values, batch.node_component, n_comp)
            return out

        node_mixture = batch.component_mixture[batch.node_component]
        coeffs = self.interaction_coefficients(h, batch.pair_index)
        T = temperature[node_mixture[batch.pair_index[0]]]
        tau = T / T_REF - 1.0
        a = coeffs[:, 0] + coeffs[:, 1] * tau + coeffs[:, 2] * tau.pow(2)
        exponent = -a / (T + EPS)
        over = (exponent.abs() - EXP_KNEE).clamp_min(0.0) / (EXP_MAX - EXP_KNEE)
        exponent = torch.sign(exponent) * (
            exponent.abs().clamp_max(EXP_KNEE) + (EXP_MAX - EXP_KNEE) * over / (1.0 + over)
        )
        log_psi = torch.log(torch.exp(exponent).clamp_min(EPS))

        # Pure-component reference: each molecule alone, atoms at equal mole fraction.
        n_atoms_component = scatter_sum(torch.ones_like(Q), batch.node_component, n_comp)
        theta_raw = Q * (1.0 / (n_atoms_component[batch.node_component] + EPS))
        theta_pure = theta_raw / (scatter_sum(theta_raw, batch.node_component, n_comp)[batch.node_component] + EPS)
        same = batch.pair_same_component
        ln_Gamma_pure = self.group_ln_gamma(Q, theta_pure[:, None], batch.pair_index[:, same], log_psi[same])

        return State(
            n_mixtures=B,
            n_components=n,
            node_mixture=node_mixture,
            node_slot=batch.component_slot[batch.node_component],
            node_component=batch.node_component,
            component_mixture=batch.component_mixture,
            component_slot=batch.component_slot,
            R=R,
            Q=Q,
            r=per_molecule(R),
            q=per_molecule(Q),
            n_atoms=per_molecule(torch.ones_like(Q)),
            pair_index=batch.pair_index,
            coeffs=coeffs,
            a=a,
            log_psi=log_psi,
            ln_Gamma_pure=ln_Gamma_pure,
            volume=volume,
            area=area,
        )

    # ---------------------------------------------------------- thermodynamics
    @staticmethod
    def combinatorial(x: torch.Tensor, r: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """mod. UNIFAC (Dortmund) combinatorial ln gamma; x is [B, P, n], r and q [B, n]."""
        x = x.clamp_min(EPS)
        x = x / (x.sum(-1, keepdim=True) + EPS)
        r, q = r[:, None, :], q[:, None, :]
        r34 = r.clamp_min(EPS).pow(0.75)
        V = r / ((r * x).sum(-1, keepdim=True) + EPS)
        V_dash = r34 / ((r34 * x).sum(-1, keepdim=True) + EPS)
        F_ = q / ((q * x).sum(-1, keepdim=True) + EPS)
        V_over_F = V / (F_ + EPS)
        return 1.0 - V_dash + torch.log(V_dash + EPS) - 0.5 * Z * q * (1.0 - V_over_F + torch.log(V_over_F + EPS))

    def group_ln_gamma(
        self, Q: torch.Tensor, theta: torch.Tensor, pairs: torch.Tensor, log_psi: torch.Tensor
    ) -> torch.Tensor:
        """ln Gamma_k of every atom ([N, P]) for area fractions ``theta`` ([N, P]).

        explicit: ln Gamma_k = Q_k [1 - ln S_k - sum_m theta_m Psi_km / S_m],
                  S_k = sum_m theta_m Psi_mk  (UNIFAC)
        implicit: ln Gamma_k = Q_k g_k with the fixed point
                  g_k = -ln sum_m theta_m exp(g_m) Psi_mk  (COSMOSPACE),
                  solved by damped iteration from the explicit value.
        """
        src, dst = pairs
        N = Q.shape[0]
        log_theta = torch.log(theta.clamp_min(EPS))
        log_psi = log_psi[:, None]
        log_S = scatter_logsumexp(log_theta[src] + log_psi, dst, N)
        correction = scatter_sum(torch.exp(log_theta[dst] + log_psi - log_S[dst]), src, N)
        g = 1.0 - log_S - correction
        if self.residual == "implicit":
            g = self._cosmospace(g.detach(), log_theta[src] + log_psi, src, dst, N)
        return Q[:, None] * g

    def _cosmospace(
        self, g: torch.Tensor, edge_terms: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, N: int
    ) -> torch.Tensor:
        """Damped fixed-point iteration, differentiated through its last steps only.

        It iterates without gradient until max|dg| <= tol (or max_iter), then takes
        ``solver_grad_steps`` more steps with gradient: a truncated Neumann series of
        the implicit-function gradient. Those steps also move the value, so they are
        part of the model, not only of its training.
        """
        alpha = self.solver_damping

        def step(g_in: torch.Tensor) -> torch.Tensor:
            return (1.0 - alpha) * g_in - alpha * scatter_logsumexp(edge_terms + g_in[src], dst, N)

        with torch.no_grad():
            for _ in range(self.solver_max_iter):
                g_next = step(g)
                done = float((g_next - g).abs().max()) <= self.solver_tol
                g = g_next
                if done:
                    break
        for _ in range(self.solver_grad_steps):
            g = step(g)
        return g

    def ln_gamma(self, state: State, x: torch.Tensor, return_parts: bool = False):
        """ln gamma at compositions ``x``: [B, n] (one per mixture) or [B, P, n].

        Returns a tensor of the same shape as ``x``; with ``return_parts`` also a dict
        with the combinatorial and residual parts and the per-atom group terms.
        """
        squeeze = x.dim() == 2
        x = x.to(self.device, torch.float32)
        if squeeze:
            x = x[:, None, :]
        B, P, n = x.shape
        ln_c = self.combinatorial(x, state.r, state.q)

        xc = x.clamp_min(EPS)
        xc = xc / (xc.sum(-1, keepdim=True) + EPS)
        denom = (xc * state.n_atoms[:, None, :]).sum(-1)  # [B, P]
        x_atom = xc[state.node_mixture, :, state.node_slot] / (denom[state.node_mixture] + EPS)
        theta_raw = state.Q[:, None] * x_atom  # [N, P]
        theta = theta_raw / (scatter_sum(theta_raw, state.node_mixture, B)[state.node_mixture] + EPS)
        ln_Gamma_mix = self.group_ln_gamma(state.Q, theta, state.pair_index, state.log_psi)

        atom_terms = ln_Gamma_mix - state.ln_Gamma_pure  # [N, P]
        per_component = scatter_sum(atom_terms, state.node_component, B * n)
        ln_r = x.new_zeros(B, n, P)
        ln_r[state.component_mixture, state.component_slot] = per_component
        ln_r = ln_r.transpose(1, 2)
        out = ln_c + ln_r
        if squeeze:
            out, ln_c, ln_r = out[:, 0], ln_c[:, 0], ln_r[:, 0]
        if return_parts:
            return out, {"combinatorial": ln_c, "residual": ln_r, "atom_residual": atom_terms, "theta": theta}
        return out

    def forward(self, batch: Batch, temperature: torch.Tensor, x: torch.Tensor):
        state = self.prepare(batch, temperature)
        return self.ln_gamma(state, x), state
