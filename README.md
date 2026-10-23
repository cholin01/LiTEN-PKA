# 🧪 LiTEN-pKa

**LiTEN-pKa** is a 3D molecular pKa prediction workflow built on the LiTEN machine-learned potential backbone. This release exposes three public workflows:

| Workflow | Script | Purpose |
| --- | --- | --- |
| 🔁 Reproduce training | `scripts/reproduce_train.py` | Split a training pool into train/validation (90:10 by molecule) and train without loading test data |
| 📊 Test checkpoints | `scripts/test_checkpoints.py` | Evaluate checkpoints on explicitly named benchmark tasks after training |
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

The training script accepts only a training pool. It makes a deterministic 9:1 train/validation split grouped by canonical molecular structure, so records for the same molecule cannot fall on both sides. The split is reproducible with `--split-seed` and independent of the model-initialization `--seed`. Early stopping and checkpoint selection use validation MAE. The held-out test CSV is not an argument to, or loaded by, the training script.

- 🧬 default training pool: `data/article_protocols/article_entry2_noC_train_pool.csv`
- 🧪 alternative pools: `data/article_protocols/article_entry3_fromNS_train_pool.csv` and `data/article_protocols/article_entry4_noNovartis_train_pool.csv`
- 🎲 training uses one randomly sampled conformer per molecule per epoch
- ✅ validation is generated internally; no test data are read during training

Run both known-IC and joint training:

```bash
PYTHONPATH=src python scripts/reproduce_train.py \
  --mode both \
  --seed 0 \
  --batch-size 32 \
  --lr 1e-4 \
  --workers 8
```

The command finishes both requested model settings for this seed before returning. Outputs and the generated split are written to `models_reproduce/` and `reports_reproduce/`. To train Entry 3 or Entry 4, pass the corresponding training-pool CSV with `--train-file` and use distinct checkpoint/report names.

## 📊 Final Test Evaluation

The test script accepts either released checkpoints or checkpoints produced by `reproduce_train.py`. For a final Entry 2 test of the newly trained models, run this only after training has completed:

```bash
PYTHONPATH=src python scripts/test_checkpoints.py \
  --benchmark entry2_noC \
  --known-ic-checkpoint models_reproduce/known_ic_entry2_noC/seed_0/best.pt \
  --joint-checkpoint models_reproduce/joint_entry2_noC/seed_0/best.pt \
  --mode both \
  --workers 8 \
  --batch-size 8
```

The default benchmark is `entry2_noC`, using:

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
| `entry1` | `data/article_protocols/article_entry1_test.csv` | pKaLearn Table 1 entry 1 testing set |
| `entry2_noC` | `data/article_protocols/article_entry2_noC_test.csv` | pKaLearn Table 1 entry2/no-carbon test |
| `entry3_fromNS` | `data/article_protocols/article_entry3_fromNS_test.csv` | pKaLearn Table 1 entry 3 held-out Novartis test set |
| `entry4_noNovartis` | `data/article_protocols/article_entry4_noNovartis_test.csv` | pKaLearn Table 1 entry 4 held-out Novartis test set |
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

Training and testing are separate entry points. `scripts/reproduce_train.py` reads only the specified training-pool CSV and creates its own 9:1 train/validation split; it has no test-file option. `scripts/test_checkpoints.py` is run separately after training and loads held-out benchmark data only for final evaluation. Validation MAE is the only quantity used for early stopping and checkpoint selection.
