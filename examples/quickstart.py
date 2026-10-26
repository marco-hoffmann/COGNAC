"""A tour of COGNAC's predictions and model insights: python examples/quickstart.py"""

import numpy as np

from cognac import load_model

model = load_model("COGNAC")  # try "COGNAC-IMP" too
mixture = ["CCO", "O"]  # ethanol + water

x = np.array([[x1, 1 - x1] for x1 in np.linspace(0, 1, 6)])
print("ln gamma of ethanol + water at 298.15 K")
for xi, (g1, g2) in zip(x, model.ln_gamma(mixture, x, 298.15)):
    print(f"  x1 = {xi[0]:.1f}: ln gamma1 = {g1:7.4f}  ln gamma2 = {g2:7.4f}")

print("\nh^E (kJ/mol) at x1 = 0.5:", round(float(model.excess_enthalpy(mixture, [0.5, 0.5], 298.15)), 4))
p, y = model.vle(mixture, [0.3, 0.7], 343.15, ps_bar=[0.72, 0.31])
print(f"VLE at 343.15 K, x1 = 0.3: p = {p:.3f} bar, y1 = {y[0]:.3f}")

print("\nn-butanol + water at 298.15 K splits into x1 =", model.lle(["CCCCO", "O"], 298.15)[:, 0].round(4))
phases = model.lle(["CCCCCC", "O", "CC(C)=O"], 298.15, feed=[0.45, 0.45, 0.1])
print("hexane + water + acetone, feed (0.45, 0.45, 0.1):\n", phases.round(4))
print(
    "solubility of naphthalene in benzene at 298.15 K: x =",
    round(model.sle(["c1ccc2ccccc2c1", "c1ccccc1"], 298.15, Tm_K=353.4, hm_kJ_mol=19.0), 4),
)

print("\nAtom geometry of acetic acid:\n", model.atoms("CC(=O)O").round(3).to_string(index=False))
pairs = model.interactions(["CO", "O"], T=298.15)
cross = pairs[pairs.molecule_k != pairs.molecule_m]
print("\nMost attractive atom pairs between methanol and water (lowest a at 298.15 K):")
print(cross.nsmallest(4, "a_T")[["atom_k", "atom_m", "a", "b", "c", "a_T"]].round(3).to_string(index=False))

components, atoms = model.decompose(mixture, [0.3, 0.7], 298.15)
print("\nln gamma = combinatorial + residual at x1 = 0.3:\n", components.round(4).to_string(index=False))
print("\nResidual contribution of each atom:\n", atoms.round(4).to_string(index=False))
