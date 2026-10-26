# Data format

`python -m cognac.predict` reads **binary** mixture data from one CSV file, one row per
data point, and scores the predictions against the values in it.

## Columns every row needs

| column | meaning |
|---|---|
| `Datatype` | `TPXY`, `TPX`, `ACT`, `HE`, `LLE` or `SLE` |
| `SMILES1`, `SMILES2` | the two components (any valid SMILES; canonicalized on reading) |
| `T_K` | temperature in K |

Other columns are ignored, so you can keep other identifiers in the file.

## Columns per data type

| `Datatype` | required | compared quantity |
|---|---|---|
| `TPXY` (VLE with vapor composition) | `x1`, `ln_gamma1`, `ln_gamma2` | both activity coefficients |
| `TPX` (VLE without vapor composition) | `x1`, `p_bar`, `ps1_bar`, `ps2_bar` | ln p, with p = x1 γ1 ps1 + x2 γ2 ps2 |
| `ACT` (activity coefficient at infinite dilution) | `ln_gamma2` | ln γ2∞, component 2 infinitely diluted in component 1 (`x1` is set to 1) |
| `HE` (excess enthalpy) | `x1`, `hE_kJ_mol` | hᴱ = −R T² ∂(gᴱ/RT)/∂T in kJ/mol |
| `LLE` (liquid-liquid equilibrium) | `x1_phase1` and/or `x1_phase2` | the coexisting phase compositions |
| `SLE` (solid-liquid equilibrium) | `x1`, and `ln_gamma1` **or** `ln_gamma2` | ln γ of the crystallizing component |

Units: mole fractions, K, bar, kJ/mol.

### Deriving the targets from measured data

The activity coefficients are not measured directly; compute them before scoring:

* **TPXY**: modified Raoult's law, `ln_gamma_i = ln(p y_i / (ps_i x_i))`, with the
  pure-component vapor pressures `ps_i` at `T`.
* **SLE**: for a pure solid i that crystallizes from the liquid,
  `ln_gamma_i = -dh_fus,i / R (1/T - 1/T_fus,i) - ln x_i`, with the enthalpy of fusion
  `dh_fus,i` and the melting temperature `T_fus,i`. Enter only the crystallizing component;
  leave the other `ln_gamma` empty.
* **LLE**: `x1_phase1` is the phase lean in component 1, `x1_phase2` the phase rich in
  it (rows with both are sorted on reading). Mutual solubility data with only one
  measured phase are fine: leave the other column empty.

## Model domain

COGNAC covers organic molecules and ionic liquids made of the elements H, B, C, N, O, F, Si, P, S, Cl, Br and I, where every cation contains carbon. Rows with a molecule outside this domain are dropped on reading, with a message saying how many.

## Example

```csv
Datatype,SMILES1,SMILES2,T_K,x1,ln_gamma1,ln_gamma2,p_bar,ps1_bar,ps2_bar,hE_kJ_mol,x1_phase1,x1_phase2
TPXY,CCO,O,343.15,0.3,0.55,0.12,,,,,,
TPX,CC(C)=O,CO,328.15,0.4,,,1.02,1.03,0.69,,,
ACT,O,CCCCO,298.15,,,3.9,,,,,,
HE,CCO,O,298.15,0.5,,,,,,-0.51,,
LLE,CCCCO,O,298.15,,,,,,,,0.019,0.48
SLE,c1ccc2ccccc2c1,c1ccccc1,298.15,0.29,0.04,,,,,,,
```
