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

## Reproducibility Workflows

The reproduction workflows use five training seeds (`0, 1, 2, 3, 4`) with a fixed molecule-grouped training/validation split (`split_seed=0`). Early stopping and checkpoint selection use validation MAE. The five best validation checkpoints are retained (`keep_top_k=5`). All five runs contribute to the reported mean and sample standard deviation.

| Analysis | Entry point | Inputs |
|---|---|---|
| Five-run training and evaluation | `scripts/reproduce_five_seeds.py` | Protocol-specific training pool and test CSVs |
| Five-run summaries | `scripts/reproduce_results.py` | Training histories, checkpoint metadata and predictions |
| Bootstrap confidence intervals | `scripts/reproduce_bootstrap.py` | Aligned predictions exported by the summary script |
| SchNet comparison | `scripts/reproduce_five_seeds.py --model schnet` | The same protocol-specific data partitions |
| Figure 5 representation analyses | `scripts/extract_figure5.py` | Fine-tuned and scratch checkpoints, training/test CSVs and conformer manifest |
| Figure 5 plotting | `scripts/reproduce_figure5.py` | Exported representation-analysis tables |

Protocol selection determines the corresponding training pool and evaluation tasks:

| `--entry` | Training pool in `data/article_protocols/` | Evaluation tasks |
|---|---|---|
| `entry1` | `article_entry1_train_pool.csv` | Entry 1 Testing |
| `entry2` | `article_entry2_noC_train_pool.csv` | NoC, SAMPL8 and euroSAMPL |
| `entry3` | `article_entry3_fromNS_train_pool.csv` | Held-out Novartis |

Run the following commands from the repository root. This example runs the Entry 2 comparison protocol:

```bash
export PYTHONPATH=src
for model in ft scratch frozen schnet; do
  python scripts/reproduce_five_seeds.py --phase train --entry entry2 --model "$model" --mode known_ic
  python scripts/reproduce_five_seeds.py --phase test --entry entry2 --model "$model" --mode known_ic
done
python scripts/reproduce_results.py --entry entry2 --models ft scratch frozen schnet
python scripts/reproduce_bootstrap.py --input-dir results/reproduced
```

Here, `ft` denotes SPICE-initialized fine-tuning, `scratch` denotes LiTEN trained from random initialization, `frozen` denotes a frozen SPICE backbone with a trainable downstream head, and `schnet` denotes the site-aware SchNet baseline trained from scratch. The examples use provided ionization centers. Joint training is available through `--mode joint` for the LiTEN variants.

The default configuration is `configs/pkalearn_liten.yaml`, with batch size 32, learning rate `1e-4` and a maximum of 100 epochs. Runs are identified by protocol, model, prediction mode and seed, for example `entry2_noC_ft_known_ic/seed_0`. Outputs under `results/fresh_five_runs/` include the generated splits, training histories, checkpoint configurations and evaluation metadata. Summaries include checkpoint hashes to link predictions to the evaluated weights.

Bootstrap analysis uses 10,000 resamples and reports 95% percentile intervals for the scratch-minus-fine-tuned error difference. Molecules are resampled as canonical-structure clusters; SAMPL8 records are grouped by parent compound. Separate intervals describe sample uncertainty, training-run uncertainty and their combination.

For the Figure 5 fine-tuned-versus-scratch analysis, use fixed checkpoints and the training partition generated above:

```bash
BASE=results/fresh_five_runs
FT="$BASE/entry2_noC_ft_known_ic/seed_0"
SC="$BASE/entry2_noC_scratch_known_ic/seed_0"
python scripts/extract_figure5.py \
  --pretrained-ckpt "$FT/models/entry2_noC_ft_known_ic/seed_0/best.pt" \
  --scratch-ckpt "$SC/models/entry2_noC_scratch_known_ic/seed_0/best.pt" \
  --train-file "$FT/reports/splits/article_entry2_noC_train_pool_split_seed_0/train.csv" \
  --test-file data/article_protocols/article_entry2_noC_test.csv \
  --manifest "$BASE/conformers/entry2_noC/manifest.csv" \
  --out-dir results/figure5/data/interpretability
python scripts/reproduce_figure5.py --data-dir results/figure5/data --out-dir results/figure5/plots
```

Linear probes and nearest-neighbor readouts are fitted using training-set representations only. Test representations are used for evaluation and visualization. The original SPICE checkpoint supplies the reference weights for the parameter-space comparisons.

## Conformer Cache

Conformers are generated from the input SMILES with explicit hydrogen atoms using RDKit ETKDGv3, followed by MMFF minimization, or UFF when MMFF is unavailable. Default settings are five conformers, base random seed `2026`, embedding limit `1000`, RMS pruning threshold `-1.0` and a maximum of 500 force-field minimization iterations.

Each cache records the exact input SMILES, RDKit version, generation settings, derived molecular seed and cache-schema version. Molecular seeds are derived deterministically from the base seed and input SMILES, independently of row order and worker scheduling. Cache reuse checks the metadata, atom ordering, coordinate dimensions and file checksum. Incompatible or corrupted caches are regenerated. Legacy caches without generation metadata are regenerated by default; explicit reuse through `legacy_cache_policy: allow` is marked as unverified.

## Ionization-Site Candidates

Hydrogen atoms remain explicit and participate in molecular encoding and graph aggregation. During joint-model inference and evaluation, candidate ionization centers are restricted to heavy atoms (`atomic number > 1`) before selecting the highest-scoring site. The selected atom embedding is then used for pKa prediction. This inference-time restriction does not change the training site-classification loss.

Known-IC prediction instead uses the supplied zero-based `center_atom_idx`, which must identify the intended heavy atom in the RDKit-parsed input SMILES before hydrogen addition.

## 🔒 Notes on Data Use

To support a fair comparison with pKaLearn, the Entry datasets follow the corresponding pKaLearn training/test partitions listed in [the protocol manifest](data/article_protocols/protocol_manifest.csv): retained records are not reassigned between training and test sets. The released CSVs reflect preprocessing and may contain fewer records than the original source files.

## Acknowledgments

We thank the pKaLearn authors for making their datasets and benchmark partitions available, and the organizers and contributors of SAMPL8 and euroSAMPL for providing the experimental pKa benchmark data.
