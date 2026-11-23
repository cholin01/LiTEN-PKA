"""Read actual LiTEN message attention during an otherwise unchanged forward pass."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from .dataset import collate_pka
from .model import move_batch_to_device


@contextmanager
def capture_attention(model, layer: int = 6):
    """Capture one 1-based layer in single-threaded, offline evaluation only.

    MBI.message returns the signed, cutoff-gated attention actually multiplied
    into messages. These are not softmax probabilities or causal attributions.
    """
    if model.training:
        raise ValueError("Attention extraction requires model.eval().")
    blocks = model.encoder.backbone.Interaction
    if not 1 <= layer <= len(blocks):
        raise ValueError(f"Attention layer must be in [1, {len(blocks)}].")
    block = blocks[layer - 1]
    original = block.message
    had_instance_method = "message" in block.__dict__
    previous = block.__dict__.get("message")
    records = []

    def capture(*args, **kwargs):
        output = original(*args, **kwargs)
        edge_index = kwargs.get("edge_index", args[2] if len(args) > 2 else None)
        dist = kwargs.get("dist", args[3] if len(args) > 3 else None)
        if edge_index is None or dist is None or len(output) != 3:
            raise ValueError("Unsupported LiTEN message interface; no attention approximation is used.")
        records.append({"layer": layer, "edge_index": edge_index.detach().cpu().clone(),
                        "distance": dist.detach().cpu().clone(),
                        "attention": output[2].detach().cpu().clone()})
        return output

    block.message = capture
    try:
        yield records
    finally:
        if had_instance_method:
            block.message = previous
        else:
            del block.message


def site_attention_tables(record, batch):
    """Return per-head raw gates and mean normalized |gate| for one molecule.

    Source j sends to target i (edge_index[0]); exclude self-edges only from
    summaries. Normalize mean absolute head weights over all incoming neighbors
    (including H) within each conformer, then average over all conformers.
    Missing edges contribute zero, without renormalizing the displayed top-k.
    """
    if batch["task_id"].numel() != 1 or bool((batch["mol_for_conf"] != 0).any()):
        raise ValueError("Extract one molecule at a time to preserve atom correspondence.")
    edges, weights = record["edge_index"], record["attention"]
    if weights.ndim != 2 or weights.shape[0] != edges.shape[1] or not torch.isfinite(weights).all():
        raise ValueError("Attention must be a finite [edge, head] array.")
    n_conf = batch["mol_for_conf"].numel()
    conf_batch = batch["conf_batch"].cpu()
    z = batch["z"].cpu()
    n_atoms = int((conf_batch == 0).sum())
    signed = np.zeros((n_conf, n_atoms))
    magnitude = np.zeros_like(signed)
    normalized = np.zeros_like(signed)
    presence = np.zeros_like(signed, dtype=bool)
    raw = []
    center = None
    for conf in range(n_conf):
        atoms = torch.nonzero(conf_batch == conf, as_tuple=False).flatten()
        if len(atoms) != n_atoms or not torch.equal(z[atoms], z[conf_batch == 0]):
            raise ValueError("Conformers must have identical atom numbers and ordering.")
        local = {int(atom): i for i, atom in enumerate(atoms)}
        target = int(batch["center_indices"][conf])
        if target not in local or (center is not None and center != local[target]):
            raise ValueError("Conformer centers do not correspond to the same atom.")
        center = local[target]
        selected = torch.nonzero(edges[0] == target, as_tuple=False).flatten()
        seen = set()
        for edge in selected.tolist():
            source = int(edges[1, edge])
            if source not in local or source in seen:
                raise ValueError("Cross-conformer or duplicate attention edge detected.")
            seen.add(source)
            atom = local[source]
            values = weights[edge].numpy()
            for head, value in enumerate(values):
                raw.append({"conformer": conf, "layer": record["layer"], "head": head,
                            "source_atom_idx": atom, "target_atom_idx": center,
                            "distance_angstrom": float(record["distance"][edge]),
                            "attention_signed": float(value), "self_edge": atom == center})
            if atom != center:
                signed[conf, atom] = float(values.mean())
                magnitude[conf, atom] = float(np.abs(values).mean())
                presence[conf, atom] = True
        total = magnitude[conf].sum()
        if total <= 0:
            raise ValueError(f"No nonzero, non-self incoming attention in conformer {conf}.")
        normalized[conf] = magnitude[conf] / total
    first_z = z[conf_batch == 0].tolist()
    summary = pd.DataFrame([
        {"atom_idx": atom, "atomic_number": first_z[atom], "center_atom_idx": center,
         "layer": record["layer"], "n_conformers": n_conf, "n_heads": weights.shape[1],
         "edge_conformer_count": int(presence[:, atom].sum()),
         "attention_signed_mean": float(signed[:, atom].mean()),
         "attention_abs_mean": float(magnitude[:, atom].mean()),
         "attention_strength": float(normalized[:, atom].mean())}
        for atom in range(n_atoms) if presence[:, atom].any()
    ])
    return pd.DataFrame(raw), summary


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_cases(model, dataset, case_rows, mode, checkpoint, layer=6):
    device = next(model.parameters()).device
    raw_tables, summaries = [], []
    checkpoint_hash = sha256(checkpoint)
    for index in case_rows:
        batch = move_batch_to_device(collate_pka([dataset[index]]), device)
        with torch.no_grad():
            expected = model(batch)
            with capture_attention(model, layer) as records:
                predicted = model(batch)
        # GPU scatter reductions may differ at floating-point roundoff between calls.
        tolerance = 0.0 if device.type == "cpu" else 1e-5
        torch.testing.assert_close(predicted, expected, rtol=tolerance, atol=tolerance)
        if len(records) != 1:
            raise ValueError("Expected exactly one attention capture per molecule.")
        raw, summary = site_attention_tables(records[0], batch)
        row = dataset.table.iloc[index]
        smiles = str(row[dataset.smiles_column])
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        summary["symbol"] = [mol.GetAtomWithIdx(int(i)).GetSymbol() for i in summary.atom_idx]
        metadata = {"mode": mode, "input_row_index": index,
                    "mol_id": str(row[dataset.id_column]), "smiles": smiles,
                    "task": str(row[dataset.task_column]), "prediction": float(predicted.item()),
                    "checkpoint_sha256": checkpoint_hash, "conformer_sha256": sha256(row.conformer_npz)}
        for frame in (raw, summary):
            for key, value in metadata.items():
                frame[key] = value
        raw_tables.append(raw)
        summaries.append(summary)
    return pd.concat(raw_tables, ignore_index=True), pd.concat(summaries, ignore_index=True)


def write_attention_outputs(pairs, checkpoints, table, manifest, out_dir, layer=6):
    summary = pd.concat([pair[1] for pair in pairs], ignore_index=True)
    for _, rows in summary.groupby("input_row_index"):
        if set(rows["mode"]) != {"pretrained_ft", "scratch"}:
            raise ValueError("Each case requires both models.")
        for key in ("mol_id", "smiles", "task", "center_atom_idx", "conformer_sha256", "n_conformers"):
            if rows[key].nunique() != 1:
                raise ValueError(f"Models used different {key} values for the same case.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([pair[0] for pair in pairs], ignore_index=True).to_csv(out_dir / "attention_raw.csv", index=False)
    summary.to_csv(out_dir / "atom_attention.csv", index=False)
    protocol = {"schema": 1,
                "checkpoints": {mode: {"checkpoint": str(Path(path).resolve()), "sha256": sha256(path)}
                                for mode, path in checkpoints.items()},
                "layer": layer, "table_sha256": sha256(table), "manifest_sha256": sha256(manifest),
                "source": "MBI.message return value attn; edge_index[0] target, edge_index[1] source",
                "raw": "Signed, cutoff-gated message attention; no softmax applied.",
                "strength": "Mean over conformers of mean-head absolute attention divided by its sum over all non-self incoming atoms, including H. Absent edges contribute zero.",
                "interpretation": "Attention strength is not a probability or causal atom attribution.",
                "prediction_unchanged": True,
                "prediction_check": "Exact on CPU; rtol=atol=1e-5 on GPU for scatter roundoff."}
    (out_dir / "attention_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
