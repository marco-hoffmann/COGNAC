"""SMILES -> molecular graph, and the model's applicability domain.

A molecule is a graph whose nodes are its heavy atoms plus the hydrogens bonded to
O, N, S or P ("polar" hydrogens, the H-bond donors). Carbon-bound hydrogens stay
implicit and enter only through the ``num_h`` feature of their carbon.

Node order: heavy atoms in RDKit order, then the polar hydrogens grouped by parent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from rdkit import Chem

#: Elements COGNAC covers: organic molecules and ionic liquids. Salts and
#: electrolytes are out of scope (there is no long-range electrostatic term).
DOMAIN_ELEMENTS = ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I")
DOMAIN_ATOMIC_NUMBERS = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)

HYBRIDIZATION = {
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
}
BOND_TYPE = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}


class OutOfDomainError(ValueError):
    """The molecule contains an element or an ionic species COGNAC does not cover."""


def out_of_domain_reason(smiles: str) -> str | None:
    """Why *smiles* is outside the model's domain, or ``None`` if it is inside.

    Inside means: every atom is one of `DOMAIN_ELEMENTS`, and every positively
    charged fragment contains carbon (the organic cations of ionic liquids are in,
    inorganic cations such as Na+ or NH4+ are out).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"invalid SMILES {smiles!r}"
    foreign = sorted({a.GetSymbol() for a in mol.GetAtoms()} - set(DOMAIN_ELEMENTS))
    if foreign:
        return f"element(s) {', '.join(foreign)} not covered"
    for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
        charge = sum(a.GetFormalCharge() for a in frag.GetAtoms())
        if charge > 0 and not any(a.GetSymbol() == "C" for a in frag.GetAtoms()):
            return "inorganic cation (salt) not covered"
    return None


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol)


def build_mol(smiles: str):
    """RDKit mol with the polar hydrogens (on O, N, S, P) as explicit atoms, or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    parents = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6) and a.GetTotalNumHs() > 0]
    return Chem.AddHs(mol, onlyOnAtoms=parents) if parents else mol


@dataclass
class Graph:
    """One molecule. ``x``: [n, 6] atom features (atomic number, degree, formal charge
    + 4, hybridization, aromatic, number of H); ``edge_index``: [2, e] directed bonds;
    ``edge_attr``: [e, 3] bond features (type, conjugated, in ring)."""

    smiles: str
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    symbols: list[str]

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]


def smiles_to_graph(smiles: str) -> Graph:
    """Molecular graph of *smiles*; raises `OutOfDomainError` outside the domain."""
    reason = out_of_domain_reason(smiles)
    if reason is not None:
        raise OutOfDomainError(f"{smiles}: {reason}. COGNAC covers organic molecules and ionic liquids only.")
    mol = build_mol(smiles)
    clamp = lambda v, lo, hi: max(lo, min(hi, v))  # noqa: E731
    x = [
        [
            a.GetAtomicNum(),
            clamp(a.GetTotalDegree(), 0, 7),
            clamp(a.GetFormalCharge(), -4, 4) + 4,
            HYBRIDIZATION.get(a.GetHybridization(), 0),
            int(a.GetIsAromatic()),
            clamp(a.GetTotalNumHs(), 0, 5),
        ]
        for a in mol.GetAtoms()
    ]
    src, dst, attr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        feat = [BOND_TYPE.get(b.GetBondType(), 0), int(b.GetIsConjugated()), int(b.IsInRing())]
        src += [i, j]
        dst += [j, i]
        attr += [feat, feat]
    return Graph(
        smiles=smiles,
        x=torch.tensor(x, dtype=torch.long),
        edge_index=torch.tensor([src, dst], dtype=torch.long).view(2, -1),
        edge_attr=torch.tensor(attr, dtype=torch.long).view(-1, 3),
        symbols=[a.GetSymbol() for a in mol.GetAtoms()],
    )


def _complete_edges(starts: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """All ordered node pairs (self-loops included) within each contiguous node span."""
    per_group = sizes * sizes
    total = int(per_group.sum())
    local = torch.arange(total) - torch.repeat_interleave(torch.cumsum(per_group, 0) - per_group, per_group)
    n = torch.repeat_interleave(sizes, per_group)
    offset = torch.repeat_interleave(starts, per_group)
    return torch.stack([local // n + offset, local % n + offset])


@dataclass
class Batch:
    """B mixtures of the same number of components n, flattened into one graph.

    ``node_component`` maps each atom to its molecule, ``component_mixture`` and
    ``component_slot`` map each molecule to its mixture and its position in it.
    ``pair_index`` holds every ordered atom pair within a mixture (self-pairs
    included): the complete interaction graph on which UNIFAC's group sums run.
    ``pair_same_component`` marks the pairs inside one molecule, which form the
    pure-component reference.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_component: torch.Tensor
    component_mixture: torch.Tensor
    component_slot: torch.Tensor
    pair_index: torch.Tensor
    pair_same_component: torch.Tensor
    n_mixtures: int
    n_components: int

    def to(self, device) -> Batch:
        return Batch(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in self.__dict__.items()})


def collate(mixtures: list[list[Graph]]) -> Batch:
    """Batch a list of mixtures, each a list of `Graph` of the same length."""
    n = len(mixtures[0])
    if any(len(m) != n for m in mixtures):
        raise ValueError("All mixtures in a batch need the same number of components.")
    graphs = [g for m in mixtures for g in m]
    sizes = torch.tensor([g.num_nodes for g in graphs])
    starts = torch.cumsum(sizes, 0) - sizes
    node_component = torch.repeat_interleave(torch.arange(len(graphs)), sizes)
    pair_index = _complete_edges(starts[::n], sizes.view(-1, n).sum(1))
    return Batch(
        x=torch.cat([g.x for g in graphs]),
        edge_index=torch.cat([g.edge_index + s for g, s in zip(graphs, starts.tolist())], dim=1),
        edge_attr=torch.cat([g.edge_attr for g in graphs]),
        node_component=node_component,
        component_mixture=torch.arange(len(mixtures)).repeat_interleave(n),
        component_slot=torch.arange(n).repeat(len(mixtures)),
        pair_index=pair_index,
        pair_same_component=node_component[pair_index[0]] == node_component[pair_index[1]],
        n_mixtures=len(mixtures),
        n_components=n,
    )
