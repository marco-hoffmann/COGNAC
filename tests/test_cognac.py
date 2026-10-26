"""Tests: python -m pytest tests (CPU, a few seconds)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from cognac import OutOfDomainError, load_model, out_of_domain_reason, smiles_to_graph, unifac
from cognac.data import load_csv

@pytest.fixture(scope="module", params=["COGNAC", "COGNAC-IMP"])
def model(request):
    return load_model(request.param)


def test_reference_values(model):
    """The shipped weights reproduce the values of the paper models."""
    ref = {"COGNAC": ([0.54459, 0.13888], -0.50647), "COGNAC-IMP": ([0.54825, 0.13979], -0.48502)}[model.name]
    np.testing.assert_allclose(model.ln_gamma(["CCO", "O"], [0.3, 0.7], 298.15), ref[0], atol=2e-5)
    assert model.excess_enthalpy(["CCO", "O"], [0.5, 0.5], 298.15) == pytest.approx(ref[1], abs=2e-4)


def test_thermodynamic_consistency(model):
    smiles = ["CC(C)=O", "O", "CCCCCC"]
    # ln gamma_i -> 0 for the pure component i
    for i in range(3):
        x = np.eye(3)[i]
        assert abs(model.ln_gamma(smiles, x, 320.0)[i]) < 1e-4
    # Gibbs-Duhem: sum_i x_i d ln gamma_i = 0 along any composition direction
    x0, d = np.array([0.3, 0.5, 0.2]), np.array([1.0, -0.5, -0.5]) * 1e-3
    dlng = model.ln_gamma(smiles, x0 + d, 320.0) - model.ln_gamma(smiles, x0 - d, 320.0)
    assert abs(x0 @ dlng) < 1e-4
    # symmetric pair parameters with a_kk = 0
    s = model.state(smiles, 320.0)
    a = torch.zeros(s.Q.numel(), s.Q.numel())
    a[s.pair_index[0], s.pair_index[1]] = s.a.detach()
    assert torch.allclose(a, a.T, atol=1e-3) and a.diagonal().abs().max() == 0


def test_component_order_does_not_matter(model):
    a = model.ln_gamma(["CCO", "O", "CC#N"], [0.2, 0.5, 0.3], 300.0)
    b = model.ln_gamma(["CC#N", "CCO", "O"], [0.3, 0.2, 0.5], 300.0)
    np.testing.assert_allclose(a, b[[1, 2, 0]], atol=1e-5)


def test_decomposition_adds_up(model):
    components, atoms = model.decompose(["CCO", "O"], [0.3, 0.7], 298.15)
    np.testing.assert_allclose(components.combinatorial + components.residual, components.ln_gamma, atol=1e-6)
    np.testing.assert_allclose(atoms.groupby("molecule").residual_contribution.sum(), components.residual, atol=1e-5)
    assert list(model.atoms("CC(=O)O").atom) == ["C1", "C2", "O3", "O4", "H5(O4)"]


def test_lle(model):
    split = model.lle(["CCCCO", "O"], 298.15)  # n-butanol + water
    assert split is not None and 0.01 < split[0, 0] < 0.05 and 0.4 < split[1, 0] < 0.6
    assert model.lle(["CCO", "O"], 298.15) is None
    phases = model.lle(["CCCCCC", "O", "CC(C)=O"], 298.15, feed=[0.45, 0.45, 0.1])
    assert phases.shape == (2, 3) and np.allclose(phases.sum(1), 1)


def test_domain():
    assert out_of_domain_reason("CCCCn1cc[n+](C)c1.F[B-](F)(F)F") is None  # ionic liquid
    assert out_of_domain_reason("[Na+].[Cl-]") is not None
    assert out_of_domain_reason("[NH4+].[Cl-]") is not None
    with pytest.raises(OutOfDomainError):
        smiles_to_graph("CC[Sn](CC)(CC)CC")
    assert smiles_to_graph("O").num_nodes == 3  # polar hydrogens are atoms


def test_data_validation(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"Datatype": ["TPXY"], "SMILES1": ["CCO"], "SMILES2": ["O"], "T_K": [300.0], "x1": [0.5]}).to_csv(
        bad, index=False
    )
    with pytest.raises(ValueError, match="ln_gamma1"):
        load_csv(bad)
    good = tmp_path / "good.csv"
    pd.DataFrame({"Datatype": ["lle"], "SMILES1": ["CCCCO"], "SMILES2": ["O"], "T_K": [298.15], "x1_phase1": [0.48],
                  "x1_phase2": [0.019]}).to_csv(good, index=False)
    df = load_csv(good)
    assert df.Datatype[0] == "LLE" and df.x1_phase1[0] < df.x1_phase2[0]  # phase 1 is the one lean in component 1


def test_unifac_table_reproduces_cognac(tmp_path):
    """The exported table in the plain mod. UNIFAC equations (cognac.unifac) gives COGNAC."""
    model = load_model("COGNAC")
    smiles = ["CC(C)=O", "O", "CCCCO"]
    model.unifac_table(smiles, tmp_path / "table.json")
    a = np.array(json.loads((tmp_path / "table.json").read_text())["a"])
    assert np.allclose(a, a.T) and not a.diagonal().any()
    x = np.array([[0.2, 0.5, 0.3], [0.05, 0.9, 0.05], [1.0, 0.0, 0.0]])
    for T in (280.0, 330.0):
        np.testing.assert_allclose(
            unifac.ln_gamma(tmp_path / "table.json", x, T), model.ln_gamma(smiles, x, T), atol=1e-4
        )
        h = unifac.excess_enthalpy(tmp_path / "table.json", x[:2], T)
        np.testing.assert_allclose(h, model.excess_enthalpy(smiles, x[:2], T), atol=1e-4)
        g = lambda t: (x[:2] * unifac.ln_gamma(tmp_path / "table.json", x[:2], t)).sum(1)  # noqa: E731
        np.testing.assert_allclose(h, -unifac.R_GAS * T**2 * (g(T + 1e-3) - g(T - 1e-3)) / 2e-3, atol=1e-6)
