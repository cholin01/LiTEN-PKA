# 🧪 LiTEN-pKa

**LiTEN-pKa** is a 3D molecular pKa prediction workflow built on the LiTEN machine-learned potential backbone. This release exposes three public workflows:

| Workflow | Script | Purpose |
| --- | --- | --- |
| 🔁 Reproduce training | `scripts/reproduce_train.py` | Split a training pool into train/validation (90:10 by molecule)  |
| 📊 Test checkpoints | `scripts/test_checkpoints.py` | Evaluate released or newly trained checkpoints on named benchmarks |
| 🚀 External prediction | `scripts/predict_external.py` | Predict pKa for an external CSV |

## 📁 Repository Layout

```text
configs/                       Training configuration
scripts/
  reproduce_train.py           Train from a selected Entry training pool
  test_checkpoints.py          Evaluate checkpoints on named benchmark tasks
  predict_external.py          Predict external CSV files
src/liten_pka/                 LiTEN-pKa package and model code
checkpoints/
  known_ic_entry2_noC/best.pt  Known-ionization-center checkpoint
  joint_entry2_noC/best.pt     Joint ionization-site and pKa checkpoint
data/                          Prepared training-pool/test and benchmark CSVs
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

Each protocol provides one training-pool CSV. Training and validation are split internally.

| Protocol | Training-pool records | Test CSV records | Test records with usable conformers |
| --- | ---: | ---: | ---: |
| Entry 1 | 9,759 | 2,187 | 2,182 |
| Entry 2 NoC | 9,460 | 2,096 | 2,091 |
| Entry 3 fromNS | 11,974 | 276 | 276 |

Entry 1–3 counts match Supplementary Table S1. All protocol filenames and counts are listed in [the protocol manifest](data/article_protocols/protocol_manifest.csv).

The training script creates an approximately 9:1 split grouped by standardized molecular structure. Keep `--split-seed` fixed when varying the training `--seed`. Early stopping and checkpoint selection use validation MAE.

- 🧬 default training pool: `data/article_protocols/article_entry2_noC_train_pool.csv`
- 🧪 alternative pools: Entry 1 and Entry 3, as listed in [the protocol manifest](data/article_protocols/protocol_manifest.csv)
- 🎲 training uses one randomly sampled conformer per molecule per epoch
- ✅ validation is generated internally

Run both known-IC and joint training:

```bash
PYTHONPATH=src python scripts/reproduce_train.py \
  --mode both \
  --seed 0 \
  --batch-size 32 \
  --lr 1e-4 \
  --workers 8
```

The command finishes both requested model settings for this seed before returning. Outputs and the generated split are written to `models_reproduce/` and `reports_reproduce/`.

For Entry 1, pass `--train-file data/article_protocols/article_entry1_train_pool.csv`. For scratch training, add `--config configs/pkalearn_liten_scratch.yaml`. Use separate output directories for each protocol and model setting.

Entry 3 uses its own training pool and output directory:

```bash
PYTHONPATH=src python scripts/reproduce_train.py \
  --mode both --seed 0 \
  --train-file data/article_protocols/article_entry3_fromNS_train_pool.csv \
  --known-ic-name known_ic_entry3_fromNS \
  --joint-name joint_entry3_fromNS \
  --checkpoint-dir models_reproduce/entry3_fromNS \
  --report-dir reports_reproduce/entry3_fromNS
```

## 📊 Final Test Evaluation

The test script accepts either released checkpoints or checkpoints produced by `reproduce_train.py`. For a final Entry 2 test of the newly trained models:

```bash
PYTHONPATH=src python scripts/test_checkpoints.py \
  --benchmark entry2_noC \
  --known-ic-checkpoint models_reproduce/known_ic_entry2_noC/seed_0/best.pt \
  --joint-checkpoint models_reproduce/joint_entry2_noC/seed_0/best.pt \
  --mode both \
  --workers 8 \
  --batch-size 8
```

For Entry 3, select the corresponding benchmark and checkpoints:

```bash
PYTHONPATH=src python scripts/test_checkpoints.py \
  --benchmark entry3_fromNS \
  --known-ic-checkpoint models_reproduce/entry3_fromNS/known_ic_entry3_fromNS/seed_0/best.pt \
  --joint-checkpoint models_reproduce/entry3_fromNS/joint_entry3_fromNS/seed_0/best.pt \
  --mode both
```

To evaluate the repository's released Entry 2 checkpoints instead, omit the checkpoint paths; the default benchmark is `entry2_noC`, using:

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

To support a fair comparison with pKaLearn, the Entry datasets follow the corresponding pKaLearn training/test partitions listed in [the protocol manifest](data/article_protocols/protocol_manifest.csv): retained records are not reassigned between training and test sets. The released CSVs reflect preprocessing and may contain fewer records than the original source files.

## Acknowledgments

We thank the pKaLearn authors for making their datasets and benchmark partitions available, and the organizers and contributors of SAMPL8 and euroSAMPL for providing the experimental pKa benchmark data.
