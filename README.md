<div align="center">

# COGNAC

**Learning atomic interactions for the thermodynamics of liquid mixtures**

[![arXiv](https://img.shields.io/badge/arXiv-2610.36440-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.36440)
[![MLPROP](https://img.shields.io/badge/MLPROP-use%20COGNAC%20online-0b7a75)](https://ml-prop.mv.rptu.de)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776ab?logo=python&logoColor=white)](pyproject.toml)

</div>

COGNAC (local-**CO**mposition **G**raph **N**eural network for the prediction of
**A**ctivity **C**oefficients) predicts activity coefficients of liquid mixtures from
molecular structure alone. A graph neural network gives every atom its own size and
pair-interaction parameters, shaped by the atom's molecular neighborhood, and evaluates
them in the unchanged equations of **modified UNIFAC (Dortmund)**.

- **Any mixture**: SMILES in, ln γ out, for any number of components and any temperature.
- **Thermodynamically consistent**: VLE, LLE, SLE and excess enthalpies all follow from one gᴱ model.
- **Interpretable**: atom volumes, surface areas and pair interaction parameters can be read off directly.
- **Process-ready**: COGNAC *is* mod. UNIFAC: export the parameter table of any mixture as JSON.

<div align="center">

**Use without installation on [MLPROP](https://ml-prop.mv.rptu.de)**

<a href="https://ml-prop.mv.rptu.de"><img src="docs/images/mlprop_logo.png" alt="MLPROP logo" height="120"></a>

</div>

## Installation

Clone the repository and create a virtual environment for it (Python ≥ 3.10):

```bash
git clone https://github.com/marco-hoffmann/COGNAC.git
cd COGNAC
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Then install PyTorch for your hardware, followed by COGNAC and its other dependencies
(RDKit, NumPy, SciPy, pandas):

**CPU only** (enough for predictions)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

**NVIDIA GPU** (for large data sets)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA version on pytorch.org
pip install -e .
python -c "import torch; print(torch.cuda.is_available())"             # should print True
```

Typical install time on a normal desktop computer: about 2.6 minutes on Windows (CPU build of PyTorch) and 3.5 minutes on Linux (CUDA build).
To check the installation, run `pip install -e ".[test]"` and `python -m pytest tests` (13 tests, about 6 s).

## Demo

Score a small demo data set with one row per data type
([examples/demo.csv](examples/demo.csv); the values are illustrative, not measured).
Your own binary data go in the same CSV layout, described in [docs/data_format.md](docs/data_format.md).

```bash
python -m cognac.predict --data examples/demo.csv --out demo_predictions.csv
```

Expected output, a table of mean absolute errors per data type (run time about 5 s on a desktop computer):

```
Datatype  rows  scored  MAE_rows  MAE_systems
     ACT     1       1    0.0473       0.0473
      HE     1       1    0.0035       0.0035
     LLE     1       1    0.0075       0.0075
     SLE     1       1    0.0173       0.0173
     TPX     1       1    0.0488       0.0488
    TPXY     1       1    0.0317       0.0317
```

The row-wise predictions are written to `demo_predictions.csv`.


## Quickstart

```python
from cognac import load_model

# Model type: "COGNAC" or "COGNAC-IMP"
model = load_model("COGNAC")                

# Activity coefficients
model.ln_gamma(["CCO", "O"], x=[0.3, 0.7], T=298.15)

# Activity coefficients at infinite dilution
model.ln_gamma_inf("CCO", "O", T=298.15)                

# Excess enthalpy     
model.excess_enthalpy(["CCO", "O"], [0.5, 0.5], T=298.15) 

# LLE phase split
model.lle(["CCCCO", "O"], T=298.15)                         
model.lle(["CCCCCC", "O", "CC(C)=O"], T=298.15, feed=[0.45, 0.45, 0.1])

# Total pressure and gas phase composition of VLE
p, y = model.vle(["CCO", "O"], [0.3, 0.7], T=343.15, ps_bar=[0.72, 0.31])

# Volume, surface area, R and Q of every atom
model.atoms("CC(=O)O")                     

# Mod. UNIFAC-type interaction parameters of every atom pair
model.interactions(["CO", "O"])             

# combinatorial and residual contributions
model.decompose(["CCO", "O"], [0.3, 0.7], T=298.15)

# export of a mod. UNIFAC parameter table for a process simulator
model.unifac_table(["CCO", "O"], "ethanol_water.json") 


# plain mod. UNIFAC in NumPy, no network
from cognac import unifac                 
unifac.ln_gamma("ethanol_water.json", x=[0.3, 0.7], T=298.15)
unifac.excess_enthalpy("ethanol_water.json", x=[0.5, 0.5], T=298.15)
```

More in [docs/usage.md](docs/usage.md) and [examples/quickstart.py](examples/quickstart.py).

## Models

| | residual term | use |
|---|---|---|
| **`COGNAC`** | closed-form mod. UNIFAC | the main model; `unifac_table()` exports it as a mod. UNIFAC parameter table for a process simulator |
| **`COGNAC-IMP`** | self-consistent (COSMOSPACE) fixed point | same network, implicit closure |

Both are the paper's full-data models, trained on about 820,000 experimental data points
for binary mixtures from the Dortmund Data Bank. They cover organic molecules and ionic
liquids of the elements H, B, C, N, O, F, Si, P, S, Cl, Br and I. Details: [docs/models.md](docs/models.md).

## Citation

If you use COGNAC, please cite

> M. Hoffmann, S. Schmitt, H. Hasse, F. Jirasek. *Learning atomic interactions for the
> thermodynamics of liquid mixtures* (2026).
> [arXiv:2610.36440](https://arxiv.org/abs/2610.36440)

## System requirements

| | |
|---|---|
| Operating system | Linux (tested: Rocky Linux 9.8, and Ubuntu via GitHub Actions) and Windows. macOS is untested. |
| Python | ≥ 3.10 (tested: 3.11 on Linux) |
| Dependencies | PyTorch, RDKit, NumPy, SciPy, pandas (declared in [pyproject.toml](pyproject.toml)) |
| Tested versions | Linux: torch 2.9.0, rdkit 2025.09.5, numpy 2.4.2, scipy 1.17.0, pandas 3.0.1. Windows: torch 2.15.0 (CPU), rdkit 2026.9.1, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0 |
| Hardware | No special hardware. Predictions run on a CPU; a GPU only helps for large data sets. |

## License

Code and trained models: [MIT License](LICENSE).
