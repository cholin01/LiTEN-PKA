"""Bootstrap paired prediction errors from five fixed model runs."""
import csv
import json
import re
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/bootstrap'
BOOTSTRAPS = 10000


def interval(x):
    return [float(v) for v in np.quantile(x, [.025, .975])]


def analyze(dataset, payload, family, rng_seed):
    records = payload['records']
    y = np.asarray(payload['y_true'], float)
    models = {g: sorted([r for r in payload['models'] if r['group'] == g], key=lambda x: x['seed'])
              for g in ('ft', 'scratch')}
    errors = {g: np.asarray([r['predictions'] for r in rows]) - y for g, rows in models.items()}
    seeds = {g: [r['seed'] for r in rows] for g, rows in models.items()}
    if not models['ft'] or not models['scratch']:
        raise ValueError("Bootstrap requires both ft and scratch predictions.")
    paired_seeds = seeds['ft'] == seeds['scratch']
    structure_count = len(set(r['canonical_smiles'] for r in records))
    # Different ionizations of the same SAMPL8 compound are not independent molecules.
    cluster_keys = ([re.sub(r'-[AB]$', '', r['source_name']) for r in records]
                    if dataset == 'SAMPL8' else [r['canonical_smiles'] for r in records])
    clusters, inverse = np.unique(cluster_keys, return_inverse=True)
    n_clusters = len(clusters)
    cluster_counts = np.bincount(inverse, minlength=n_clusters)
    sums = {g: {metric: np.asarray([np.bincount(inverse, weights=(np.abs(e) if metric == 'mae' else e**2),
                                                minlength=n_clusters) for e in error])
                for metric in ('mae', 'rmse')} for g, error in errors.items()}
    observed = {g: {'mae': np.abs(e).mean(axis=1), 'rmse': np.sqrt((e**2).mean(axis=1))}
                for g, e in errors.items()}
    draws = {metric: {source: [] for source in ('test_samples_only', 'training_runs_only', 'joint')}
             for metric in ('mae', 'rmse')}
    rng = np.random.default_rng(rng_seed)
    for start in range(0, BOOTSTRAPS, 100):
        n = min(100, BOOTSTRAPS-start)
        weights = rng.multinomial(n_clusters, np.full(n_clusters, 1/n_clusters), size=n)
        denominators = weights @ cluster_counts
        indexes = {'ft': rng.integers(0, len(seeds['ft']), size=(n, len(seeds['ft'])))}
        indexes['scratch'] = indexes['ft'] if paired_seeds else rng.integers(0, len(seeds['scratch']), size=(n, len(seeds['scratch'])))
        for metric in ('mae', 'rmse'):
            group_samples = {}
            for group in ('ft', 'scratch'):
                value = (weights @ sums[group][metric].T) / denominators[:, None]
                group_samples[group] = np.sqrt(value) if metric == 'rmse' else value
            sample_diff = group_samples['scratch'].mean(axis=1) - group_samples['ft'].mean(axis=1)
            run_diff = observed['scratch'][metric][indexes['scratch']].mean(axis=1) - observed['ft'][metric][indexes['ft']].mean(axis=1)
            joint_diff = np.take_along_axis(group_samples['scratch'], indexes['scratch'], axis=1).mean(axis=1) - np.take_along_axis(group_samples['ft'], indexes['ft'], axis=1).mean(axis=1)
            for key, values in [('test_samples_only', sample_diff), ('training_runs_only', run_diff), ('joint', joint_diff)]:
                draws[metric][key].extend(values.tolist())
    metrics = {}
    for metric in ('mae', 'rmse'):
        metrics[metric] = {'ft_mean': float(observed['ft'][metric].mean()),
                           'ft_sample_sd': float(observed['ft'][metric].std(ddof=1)),
                           'scratch_mean': float(observed['scratch'][metric].mean()),
                           'scratch_sample_sd': float(observed['scratch'][metric].std(ddof=1)),
                           'scratch_minus_ft': float(observed['scratch'][metric].mean() - observed['ft'][metric].mean()),
                           'ci95': {source: interval(values) for source, values in draws[metric].items()}}
    ft_error = np.abs(errors['ft']).mean(axis=0)
    scratch_error = np.abs(errors['scratch']).mean(axis=0)
    gain = scratch_error-ft_error
    ordered = np.argsort(gain)[::-1]
    cases = []
    for i, r in enumerate(records):
        cases.append({**r, 'ft_mean_absolute_error': float(ft_error[i]), 'scratch_mean_absolute_error': float(scratch_error[i]),
                      'scratch_minus_ft_absolute_error': float(gain[i]),
                      'ft_predictions_by_seed': {str(m['seed']): m['predictions'][i] for m in models['ft']},
                      'scratch_predictions_by_seed': {str(m['seed']): m['predictions'][i] for m in models['scratch']}})
    distribution = {'n_records_with_lower_ft_error': int((gain>0).sum()), 'n_records_with_higher_ft_error': int((gain<0).sum()),
                    'median_record_gain': float(np.median(gain)),
                    'leave_one_record_out_delta_range': [float(v) for v in [(gain.sum()-gain).min()/(len(gain)-1), (gain.sum()-gain).max()/(len(gain)-1)]],
                    'largest_positive_gains': [cases[i] for i in ordered[:3]],
                    'largest_negative_gains': [cases[i] for i in ordered[-3:]],
                    'scratch_mean_error_above_1': int((scratch_error>1).sum()),
                    'above_1_corrected_to_at_most_1': int(((scratch_error>1)&(ft_error<=1)).sum()),
                    'at_most_1_regressed_above_1': int(((scratch_error<=1)&(ft_error>1)).sum())}
    for k in (1,3,5):
        remaining = np.delete(gain, ordered[:k])
        positive_total = float(np.maximum(gain, 0).sum())
        distribution[f'after_removing_top_{k}_positive_gain_records'] = float(remaining.mean()) if len(remaining) else None
        distribution[f'top_{k}_share_of_all_positive_gains'] = float(np.maximum(gain[ordered[:k]],0).sum()/positive_total) if positive_total else None
    (OUT / f'{dataset}_molecule_level_results.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2)+'\n')
    with (OUT / f'{dataset}_molecule_level_results.csv').open('w', newline='') as f:
        fields = ['row_idx', 'mol_id', 'source_name', 'source_dataset', 'smiles', 'canonical_smiles', 'pka', 'task', 'center_symbol',
                  'ft_mean_absolute_error', 'scratch_mean_absolute_error', 'scratch_minus_ft_absolute_error']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader();writer.writerows(cases)
    return {'family': family, 'dataset': dataset, 'n_records': len(y), 'n_structures': structure_count,
            'n_bootstrap_clusters': n_clusters,
            'bootstrap_cluster_definition': 'Parent challenge compound ID, pooling A/B ionization records' if dataset == 'SAMPL8' else 'Canonical isomeric SMILES after removing atom maps and explicit hydrogens',
            'seeds': seeds, 'seed_resampling': 'matched pairs' if paired_seeds else 'separately within each model group',
            'metrics': metrics, 'gain_distribution': distribution}


def main():
    global OUT, BOOTSTRAPS
    from reproduce_results import load_predictions
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.draws < 2:
        parser.error("--draws must be >= 2")
    OUT, BOOTSTRAPS = args.out_dir.resolve(), args.draws
    if OUT == args.input_dir.resolve() or args.input_dir.resolve() in OUT.parents:
        parser.error("Outputs must be separate from source data")
    OUT.mkdir(parents=True, exist_ok=True)
    results = [analyze(name, payload, family, args.seed + i)
               for i, (family, name, payload) in enumerate(load_predictions(args.input_dir))]
    report = {
        "bootstrap_draws": BOOTSTRAPS, "seed": args.seed,
        "interval_method": "Two-sided 95% percentile bootstrap. SAMPL8 parent compounds or canonical structure clusters are sampled jointly across models; metrics remain record-weighted. Compute each run's RMSE before averaging runs.",
        "important_scope": "Conditional on the supplied fixed runs; not a correction for checkpoint/seed selection.",
        "uncertainty_components": {
            "test_samples_only": "Resample structure clusters with models fixed.",
            "training_runs_only": "Resample model runs with test records fixed.",
            "joint": "Resample model runs and shared structure clusters."},
        "results": results}
    if args.reference:
        reference = json.loads(args.reference.read_text())
        if reference["bootstrap_draws"] != args.draws or reference["seed"] != args.seed:
            raise ValueError("Reference Bootstrap seed/draw count differs.")
        for got, expected in zip(results, reference["results"], strict=True):
            if got["dataset"] != expected["dataset"]:
                raise ValueError("Bootstrap dataset order differs from reference")
            for metric in ("mae", "rmse"):
                for component in ("test_samples_only", "training_runs_only", "joint"):
                    if not np.allclose(got["metrics"][metric]["ci95"][component],
                                       expected["metrics"][metric]["ci95"][component], atol=1e-10, rtol=0):
                        raise ValueError(f"Bootstrap reference mismatch: {got['dataset']}/{metric}/{component}")
        report["reference_check"] = "passed"
    (OUT / "paired_uncertainty_and_gain_distribution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Bootstrap: {len(results)} datasets, {BOOTSTRAPS} draws; saved to {OUT}")


if __name__ == "__main__":
    main()
