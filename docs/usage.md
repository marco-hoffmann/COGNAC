# Predictions and model insights

To try COGNAC without installing anything, use it in the browser on
[MLPROP](https://ml-prop.mv.rptu.de).

```python
from cognac import load_model

model = load_model("COGNAC")          # "COGNAC-IMP" for the implicit closure
model = load_model("COGNAC-IMP", device="cuda")           # on a GPU
model = load_model("path/to/checkpoint.pt")              # a checkpoint file
```

Every method takes the mixture as a list of SMILES. Compositions are mole fractions:
one point `[x1, x2, ...]` or an array of points `[[x1, x2, ...], ...]`. Temperatures are
in K, pressures in bar, enthalpies in kJ/mol. Results are NumPy arrays or pandas
DataFrames. A molecule outside the model domain raises `cognac.OutOfDomainError`.

## Thermodynamic properties

```python
water_ethanol = ["O", "CCO"]

model.ln_gamma(water_ethanol, [0.7, 0.3], T=298.15)        # array([0.1389, 0.5446])
model.ln_gamma(water_ethanol, [[0.9, 0.1], [0.5, 0.5]], T=298.15)   # one row per point
model.ln_gamma_inf("CCO", "O", T=298.15)                  # ln gamma of ethanol at infinite dilution in water
model.gE(water_ethanol, [0.5, 0.5], T=298.15)              # g^E / RT
model.excess_enthalpy(water_ethanol, [0.5, 0.5], T=298.15) # h^E in kJ/mol, -0.506
```

Mixtures can have any number of components, e.g.
`model.ln_gamma(["O", "CCO", "CCCCCC"], [0.4, 0.3, 0.3], 298.15)`.

**Phase equilibria**

```python
# vapor-liquid: bubble pressure and vapor composition for given pure-component vapor pressures
p, y = model.vle(water_ethanol, [0.7, 0.3], T=343.15, ps_bar=[0.31, 0.72])

# liquid-liquid: coexisting phases, or None if the mixture is stable
model.lle(["CCCCO", "O"], T=298.15)                              # binary: the miscibility gap
model.lle(["CCCCCC", "O", "CC(C)=O"], T=298.15, feed=[0.45, 0.45, 0.1])   # ternary: phases of this feed

# solid-liquid: solubility of a pure solid (naphthalene) in benzene
model.sle(["c1ccc2ccccc2c1", "c1ccccc1"], T=298.15, Tm_K=353.4, hm_kJ_mol=19.0)   # mole fraction
```

`lle` finds the split with the convex envelope of Δg_mix on a composition grid (step
1/128 for binary and ternary mixtures, 1/32 for quaternary ones, as in the paper) and
refines it with a Newton solve of the isoactivity equations.

**Prediction of entire data sets**

To predict a whole data set in the [data layout](data_format.md) and
score it against the values in it:

```bash
python -m cognac.predict --data my_data.csv --out predictions.csv --model COGNAC
```

## Model insights

COGNAC is mod. UNIFAC with every atom as its own group, so everything the equations
use can be inspected.

**Atom geometry** 

Cavity volume and surface area per atom, and the UNIFAC size
parameters R_k and Q_k (relative to a CH₂ group):

```python
model.atoms("CC(=O)O")
#     atom  volume_A3  area_A2      R      Q
# 0     C1      30.13    36.72  1.493  1.735
# 1     C2      10.94    10.76  0.542  0.508
# 2     O3      15.02    20.10  0.744  0.949
# 3     O4      11.56    15.00  0.573  0.709
# 4  H5(O4)      4.77     8.14  0.236  0.384
```

Atoms are named by element and position in the SMILES (heavy atoms first); the hydrogens
on O, N, S and P are atoms of their own, named with their parent (`H5(O4)`). Hydrogens on
carbon are part of their carbon.

**Interaction parameters** 

Every atom pair of a mixture with the coefficients of the
mod. UNIFAC temperature function, in the model's basis a(T) = A + B τ + C τ²
(τ = T/298.15 K − 1) and in the published form a + b T + c T², plus a(T) and
Ψ = exp(−a/T) at the requested temperature:

```python
pairs = model.interactions(["CO", "O"], T=298.15)
pairs[(pairs.molecule_k != pairs.molecule_m)].sort_values("a_T").head()
```

Pairs within one molecule belong to the pure-component reference. The parameters are
symmetric, a_km = a_mk, and a_kk = 0 at every temperature by construction: the network's
raw pair output u_km enters only as the exchange combination
a_km = u_km − (u_kk + u_mm)/2.

**mod. UNIFAC parameter table** 

For a process simulator, or any mod. UNIFAC (Dortmund)
implementation, export the complete parameters of a mixture in one consistent set:

```python
table = model.unifac_table(["CCO", "O"], "ethanol_water.json")   # also returned as a dict
```

```json
{
 "model": "COGNAC",
 "residual": "explicit",
 "form": "mod. UNIFAC (Dortmund): Psi_mn = exp(-(a_mn + b_mn*T + c_mn*T^2)/T), T in K",
 "molecules": [{"smiles": "CCO", "groups": [0, 1, 2, 3]}, {"smiles": "O", "groups": [4, 5, 6]}],
 "groups": [{"id": 0, "molecule": 0, "atom": "C1", "R": 1.4899, "Q": 1.7687}, ...],
 "a": [[0.0, ...], ...],
 "b": [[0.0, ...], ...],
 "c": [[0.0, ...], ...]
}
```

Every atom is one group that occurs once in its molecule; `a` (K), `b` (–) and `c` (1/K)
are N × N matrices over all groups of the mixture, symmetric with a zero diagonal. The
standard mod. UNIFAC equations with these parameters (Dortmund combinatorial term,
z = 10) reproduce COGNAC to float precision (≤ 4e-6 in ln γ). COGNAC-IMP uses the same
parameters in a self-consistent residual term, which standard implementations lack.

`cognac/unifac.py` is such an implementation: the textbook equations in about 40 lines of
NumPy, with no network involved. It needs nothing but NumPy, so the single file can be
copied into another project:

```python
from cognac import unifac

unifac.ln_gamma("ethanol_water.json", x=[0.3, 0.7], T=298.15)            # array([0.5446, 0.1389])
unifac.ln_gamma("ethanol_water.json", x=[[0.1, 0.9], [0.5, 0.5]], T=320.0)   # one row per point
unifac.excess_enthalpy("ethanol_water.json", x=[0.5, 0.5], T=298.15)       # -0.506 kJ/mol
```

**Decomposing ln γ**

```python
components, atoms = model.decompose(["CCO", "O"], [0.3, 0.7], T=298.15)
```

`components` gives, per component, ln γ = combinatorial + residual. `atoms` gives each
atom's area fraction θ in the mixture and its residual contribution
Q_k [ln Γ_k(mixture) − ln Γ_k(pure)]; the contributions of a molecule's atoms sum to its
residual ln γ.

**Lower level** 

`model.state(smiles, T)` runs the network and returns a `cognac.model.State`
with all per-atom and per-pair quantities as tensors; `model.model.ln_gamma(state, x)`
evaluates any compositions without re-running the network, with `return_parts=True` for
the parts above. This is what the phase equilibrium solvers use.

## Available models

| | COGNAC | COGNAC-IMP |
|---|---|---|
| residual term | closed-form mod. UNIFAC | self-consistent (COSMOSPACE) fixed point |
| in a process simulator | yes: `unifac_table()` is a mod. UNIFAC parameter set | needs the fixed-point solver |
| accuracy | the paper's main model | slightly better for binary mixtures at infinite dilution  |

