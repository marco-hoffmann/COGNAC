# The trained models

`cognac/checkpoints/` holds the two full-data models of the paper, weights only (8.8 MB each):

```
cognac/checkpoints/
├── COGNAC/full_data.pt
├── COGNAC-IMP/full_data.pt
└── SHA256SUMS.json
```

| | COGNAC | COGNAC-IMP |
|---|---|---|
| residual term | closed-form mod. UNIFAC | self-consistent COSMOSPACE fixed point |
| `load_model(...)` | `"COGNAC"` | `"COGNAC-IMP"` |

Both share every other setting: the
same network, the mod. UNIFAC (Dortmund) combinatorial term and temperature function,
polar hydrogens as atoms, COSMO and H-bond regularization, 100 epochs, seed 42. The
shipped weights are those after the last epoch.

## Training data

About 820,000 experimental data points for 48,000 binary systems from the Dortmund Data
Bank (VLE, LLE, SLE, activity coefficients at infinite dilution and excess enthalpies),
split by system into ten folds. The data are licensed and cannot be shared here.

The shipped full-data models were trained on folds 0–8 with fold 9 for validation (no
test set) and are used in the paper for everything except the binary accuracy. That
accuracy comes from a ten-fold cross-validation of the same configuration, whose ten
models per variant are not distributed here.

The checkpoints were trained before the model domain was fixed to organic molecules and
ionic liquids, and saw about 500 rows (0.08 %) of simple salts. Such molecules are now
rejected on input (`OutOfDomainError`) and are not part of any reported number.

## Checkpoint format

```python
payload = torch.load("cognac/checkpoints/COGNAC/full_data.pt", weights_only=True)
payload["config"]       # arguments of cognac.model.COGNAC
payload["state_dict"]   # weights
payload["training"]     # the training settings of the paper run
payload["provenance"]   # model, split and source run
```
