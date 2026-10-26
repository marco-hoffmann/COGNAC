"""Predict every row of a data CSV (layout: docs/data_format.md) and score it.

    python -m cognac.predict --data test.csv --out predictions.csv [--model COGNAC-IMP]

Adds ``*_pred`` columns: ln_gamma1/2 (TPXY, ACT, SLE), p_bar (TPX), hE_kJ_mol (HE) and
x1_phase1/2 (LLE), plus ``abs_err``, the error measure of each datatype. Where
targets are present it prints the mean absolute error per datatype, over rows and
over systems (mean of the per-system means).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from .api import Predictor, load_model
from .data import batch_of, load_csv

CHUNK = 256


def predict_frame(model: Predictor, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    net = model.model
    for col in ("ln_gamma1_pred", "ln_gamma2_pred", "p_bar_pred", "hE_kJ_mol_pred", "x1_phase1_pred", "x1_phase2_pred"):
        df[col] = np.nan

    rows = np.flatnonzero(df["Datatype"].isin(["TPXY", "TPX", "ACT", "SLE", "HE"]).to_numpy())
    for start in range(0, len(rows), CHUNK):
        idx = rows[start : start + CHUNK]
        sub = df.iloc[idx]
        T = torch.tensor(sub["T_K"].to_numpy(), dtype=torch.float32, device=net.device, requires_grad=True)
        x1 = torch.tensor(sub["x1"].to_numpy(), dtype=torch.float32, device=net.device)
        x = torch.stack([x1, 1 - x1], dim=1)
        ln_g = net.ln_gamma(net.prepare(batch_of(sub[["SMILES1", "SMILES2"]].values.tolist()), T), x)
        (dgdT,) = torch.autograd.grad((x * ln_g).sum(1).sum(), T)
        ln_g = ln_g.detach().cpu().numpy()
        df.iloc[idx, df.columns.get_loc("ln_gamma1_pred")] = ln_g[:, 0]
        df.iloc[idx, df.columns.get_loc("ln_gamma2_pred")] = ln_g[:, 1]
        df.iloc[idx, df.columns.get_loc("hE_kJ_mol_pred")] = (-0.008314 * T.pow(2) * dgdT).detach().cpu().numpy()
        x1n = sub["x1"].to_numpy()
        df.iloc[idx, df.columns.get_loc("p_bar_pred")] = (
            x1n * np.exp(ln_g[:, 0]) * sub["ps1_bar"].to_numpy()
            + (1 - x1n) * np.exp(ln_g[:, 1]) * sub["ps2_bar"].to_numpy()
        )

    lle = df["Datatype"] == "LLE"
    for (s1, s2, T), group in df[lle].groupby(["SMILES1", "SMILES2", "T_K"]):
        phases = model.lle([s1, s2], T)
        if phases is not None:
            df.loc[group.index, ["x1_phase1_pred", "x1_phase2_pred"]] = phases[:, 0]
    return add_errors(df)


def add_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Error per row: mean |d ln gamma| over the given components (TPXY, SLE), |d ln gamma_2|
    (ACT), |d ln p| (TPX), |d h^E| in kJ/mol (HE), mean |d x1| over the given phases (LLE)."""
    d = lambda c: (df[f"{c}_pred"] - df[c]).abs()  # noqa: E731
    err = pd.Series(np.nan, index=df.index)
    dt = df["Datatype"]
    err[dt.isin(["TPXY", "SLE"])] = pd.concat([d("ln_gamma1"), d("ln_gamma2")], axis=1).mean(axis=1)
    err[dt == "ACT"] = d("ln_gamma2")
    err[dt == "TPX"] = (np.log(df["p_bar_pred"]) - np.log(df["p_bar"])).abs()
    err[dt == "HE"] = d("hE_kJ_mol")
    err[dt == "LLE"] = pd.concat([d("x1_phase1"), d("x1_phase2")], axis=1).mean(axis=1)
    df["abs_err"] = err[dt.isin(["TPXY", "SLE", "ACT", "TPX", "HE", "LLE"])]
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """MAE per datatype over rows and over systems (unordered molecule pairs).
    LLE rows without a predicted split are counted separately, not in the MAE."""
    df = df.assign(system=[" + ".join(sorted(p)) for p in df[["SMILES1", "SMILES2"]].values])
    out = []
    for dtype, g in df.groupby("Datatype"):
        scored = g.dropna(subset=["abs_err"])
        out.append(
            {
                "Datatype": dtype,
                "rows": len(g),
                "scored": len(scored),
                "MAE_rows": scored["abs_err"].mean(),
                "MAE_systems": scored.groupby("system")["abs_err"].mean().mean(),
            }
        )
    return pd.DataFrame(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="COGNAC", help="COGNAC, COGNAC-IMP or a checkpoint path")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    df = predict_frame(load_model(args.model, args.device), load_csv(args.data))
    df.to_csv(args.out, index=False)
    print(summary(df).to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
