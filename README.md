<div align="center">

# COGNAC

**Learning atomic interactions for the thermodynamics of liquid mixtures**

<!-- TODO: replace "#citation" with the arXiv URL once the preprint is online -->
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?logo=arxiv&logoColor=white)](#citation)
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

The two trained models (17 MB) come with the repository. `-e` installs COGNAC in
editable mode: the code stays in this folder, so changes and `git pull` take effect
without reinstalling.

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

## Documentation

| | |
|---|---|
| [docs/usage.md](docs/usage.md) | predictions, phase equilibria and model insights |
| [docs/data_format.md](docs/data_format.md) | the CSV layout for scoring a data set |
| [docs/models.md](docs/models.md) | the shipped models, training data and verification |

## Citation

If you use COGNAC, please cite

> M. Hoffmann, S. Schmitt, H. Hasse, F. Jirasek. *Learning atomic interactions for the
> thermodynamics of liquid mixtures* (2026). arXiv: coming soon.

## License

Code and trained models: [MIT License](LICENSE).
