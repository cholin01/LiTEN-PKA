# 🧪 LiTEN-pKa

**LiTEN-pKa** is a 3D molecular pKa prediction workflow built on the LiTEN machine-learned potential backbone. This release exposes three public workflows:

| Workflow | Script | Purpose |
| --- | --- | --- |
| 🔁 Reproduce training | `scripts/reproduce_train.py` | Reproduce the entry2/no-carbon training protocol using train/validation data only |
| 📊 Test checkpoints | `scripts/test_checkpoints.py` | Evaluate the two released checkpoints on explicitly named benchmark tasks |
| 🚀 External prediction | `scripts/predict_external.py` | Predict pKa for an external CSV |

✅ The released checkpoints are validation-selected. Training code does not monitor test or external benchmark sets.

## 📁 Repository Layout

```text
configs/                       Training configuration
scripts/
  reproduce_train.py           Reproduce our entry2/no-carbon training results
  test_checkpoints.py          Test the two released checkpoints on named benchmark tasks
  predict_external.py          Predict external CSV files
src/liten_pka/                 LiTEN-pKa package and model code
checkpoints/
  known_ic_entry2_noC/best.pt  Known-ionization-center checkpoint
  joint_entry2_noC/best.pt     Joint ionization-site and pKa checkpoint
data/                          Prepared train/validation/test and benchmark CSVs
pretrain_ckpt/                 LiTEN_SPICE.pt backbone checkpoint
vendor/                        Minimal LiTEN backbone source required at runtime
```

## ⚙️ Installation

```bash
conda create -n liten-pka python=3.10 -y
conda activate liten-pka
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build that matches your system if GPU inference/training is needed.

## 🧾 Input Columns

Training and labeled evaluation tables use:

| column | description |
| --- | --- |
| `mol_id` | unique molecule/site identifier |
| `smiles` | SMILES used for RDKit conformer generation |
| `pka` | experimental pKa label |
| `task` | `acid`, `base`, or `unknown` |
| `center_atom_idx` | zero-based heavy-atom ionization-center index |

External joint prediction only requires `smiles`. If `mol_id`, `pka`, `task`, or `center_atom_idx` are missing, the prediction loader creates temporary placeholders without modifying the input CSV. Known-IC prediction requires valid `center_atom_idx` values.

## 🔁 Reproduce Our Training Results

This script reproduces the pKaLearn article Table 1 entry2/no-carbon protocol used for the released checkpoints:

- 🧬 training file: `data/article_protocols/article_entry2_noC_train.csv`
- 🧪 validation file: `data/article_protocols/article_entry2_noC_val.csv`
- 🎲 training uses one randomly sampled conformer per molecule per epoch
- ✅ checkpoint selection uses validation MAE only

Run both known-IC and joint training:

```bash
PYTHONPATH=src python scripts/reproduce_train.py \
  --mode both \
  --seed 0 \
  --batch-size 32 \
  --lr 1e-4 \
  --workers 8
```

Outputs are written to `models_reproduce/` and `reports_reproduce/`.

## 📊 Test Released Checkpoints

The released checkpoints are:

- 🎯 known IC: `checkpoints/known_ic_entry2_noC/best.pt`
- 🔗 joint IC+pKa: `checkpoints/joint_entry2_noC/best.pt`

Default benchmark: `entry2_noC`, which tests the article entry2/no-carbon protocol on:

`data/article_protocols/article_entry2_noC_test.csv`

```bash
PYTHONPATH=src python scripts/test_checkpoints.py \
  --benchmark entry2_noC \
  --mode both \
  --workers 8 \
  --batch-size 8
```

Available named benchmark tasks:

| benchmark | CSV | purpose |
| --- | --- | --- |
| `entry2_noC` | `data/article_protocols/article_entry2_noC_test.csv` | pKaLearn Table 1 entry2/no-carbon test |
| `official_test` | `data/processed/test.csv` | included official held-out test split |
| `sampl8` | `data/downstream/SAMPL8.csv` | external SAMPL8 benchmark |
| `eurosampl` | `data/downstream/euroSAMPL.csv` | external euroSAMPL benchmark |

Outputs are written to `predictions/<benchmark>/` with one CSV and one `*.metrics.json` file per model.

## 🚀 Predict External CSV Files

For a new CSV with only SMILES, use the joint model:

```bash
PYTHONPATH=src python scripts/predict_external.py \
  --mode joint \
  --input path/to/external.csv \
  --output predictions/external_predictions.csv \
  --workers 8
```

Minimum external CSV:

```csv
smiles
CCN(CC)CC
O=C(O)c1ccccc1
```

If the CSV includes `pka`, metrics are printed and saved in the output dataframe attributes by the Python API. If `task` is unknown, the model still predicts, but providing `acid` or `base` is recommended for best performance.

Known-IC external prediction requires `center_atom_idx`:

```bash
PYTHONPATH=src python scripts/predict_external.py \
  --mode known_ic \
  --input path/to/external_with_ic.csv \
  --output predictions/external_known_ic_predictions.csv
```

## 🔒 Notes on Data Use

Training scripts read only the configured train and validation CSV files. Test and external benchmark CSVs are used only by `scripts/test_checkpoints.py` or by explicit external prediction/evaluation commands after training.
