"""Binary mixture data: one CSV of binary mixture data, one row per data point.

Every row has ``Datatype``, ``SMILES1``, ``SMILES2`` and ``T_K``. The other columns
depend on the datatype (see docs/data_format.md):

=========  ======================================================  ==========================
Datatype   columns                                                 compared quantity
=========  ======================================================  ==========================
TPXY       x1, ln_gamma1, ln_gamma2                                ln gamma_1, ln gamma_2
TPX        x1, p_bar, ps1_bar, ps2_bar                             ln p, p = sum x_i gamma_i ps_i
ACT        ln_gamma2  (component 2 infinitely diluted in 1)        ln gamma_2^inf
HE         x1, hE_kJ_mol                                           h^E = -R T^2 d(g^E/RT)/dT
LLE        x1_phase1 and/or x1_phase2                              coexisting phase compositions
SLE        x1, ln_gamma1 or ln_gamma2  (of the solid component)    ln gamma of the solid
=========  ======================================================  ==========================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .chem import Batch, canonical_smiles, collate, out_of_domain_reason, smiles_to_graph

DATATYPES = ("TPXY", "TPX", "ACT", "HE", "LLE", "SLE")
REQUIRED = {
    "TPXY": ["x1", "ln_gamma1", "ln_gamma2"],
    "TPX": ["x1", "p_bar", "ps1_bar", "ps2_bar"],
    "ACT": ["ln_gamma2"],
    "HE": ["x1", "hE_kJ_mol"],
    "LLE": [],
    "SLE": ["x1"],
}
NUMERIC = ["T_K", "x1", "ln_gamma1", "ln_gamma2", "p_bar", "ps1_bar", "ps2_bar", "hE_kJ_mol", "x1_phase1", "x1_phase2"]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read and validate a data CSV; drops (and reports) rows outside the model domain."""
    df = pd.read_csv(path)
    missing = [c for c in ("Datatype", "SMILES1", "SMILES2", "T_K") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    df["Datatype"] = df["Datatype"].astype(str).str.upper()
    unknown = sorted(set(df["Datatype"]) - set(DATATYPES))
    if unknown:
        raise ValueError(f"{path}: unknown Datatype(s) {unknown}; expected {DATATYPES}")

    def fail(mask, message):
        if mask.any():
            rows = (df.index[mask] + 2).tolist()[:10]  # +2: header line and 1-based numbering
            raise ValueError(f"{path}: {message} (CSV lines {rows}{' ...' if mask.sum() > 10 else ''})")

    fail(~(df["T_K"] > 0), "T_K must be a positive number")
    df.loc[df["Datatype"] == "ACT", "x1"] = 1.0
    for dtype, cols in REQUIRED.items():
        rows = (df["Datatype"] == dtype).to_numpy()
        for col in cols:
            fail(rows & ~np.isfinite(df[col].to_numpy()), f"{dtype} rows need a value in {col}")
    fail(~df["x1"].between(0, 1) & df["x1"].notna(), "x1 must lie in [0, 1]")
    lle = (df["Datatype"] == "LLE").to_numpy()
    fail(lle & df[["x1_phase1", "x1_phase2"]].isna().all(axis=1).to_numpy(), "LLE rows need x1_phase1 or x1_phase2")
    swap = (df["x1_phase1"] > df["x1_phase2"]).to_numpy()  # convention: phase 1 is the one lean in component 1
    df.loc[swap, ["x1_phase1", "x1_phase2"]] = df.loc[swap, ["x1_phase2", "x1_phase1"]].to_numpy()
    sle = (df["Datatype"] == "SLE").to_numpy()
    fail(sle & df[["ln_gamma1", "ln_gamma2"]].isna().all(axis=1).to_numpy(), "SLE rows need ln_gamma1 or ln_gamma2")

    reasons = {s: out_of_domain_reason(s) for s in pd.unique(df[["SMILES1", "SMILES2"]].values.ravel())}
    bad = df["SMILES1"].map(reasons).notna() | df["SMILES2"].map(reasons).notna()
    if bad.any():
        examples = sorted({f"{s} ({r})" for s, r in reasons.items() if r})[:5]
        print(f"[data] {path}: dropped {int(bad.sum())} of {len(df)} rows outside the model domain, e.g. {examples}")
        df = df.loc[~bad]
    df = df.reset_index(drop=True)
    for col in ("SMILES1", "SMILES2"):
        df[col] = df[col].map({s: canonical_smiles(s) for s in pd.unique(df[col])})
    return df


def batch_of(smiles_rows: list[list[str]]) -> Batch:
    """Batch of mixtures given as SMILES lists (graphs built once per molecule)."""
    graphs = {s: smiles_to_graph(s) for s in {s for row in smiles_rows for s in row}}
    return collate([[graphs[s] for s in row] for row in smiles_rows])
