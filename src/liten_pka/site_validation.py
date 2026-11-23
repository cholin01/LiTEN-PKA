"""Validate zero-based ionization sites without silently changing their indices."""
from __future__ import annotations

import math

import torch


def checked_center(value, atomic_numbers, *, allow_missing=False, context="input") -> int:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{context}: center_atom_idx must be a finite integer.") from None
    if isinstance(value, bool) or not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{context}: center_atom_idx must be a finite integer, got {value!r}.")
    index = int(number)
    if index == -1 and allow_missing:
        return index
    if not 0 <= index < len(atomic_numbers):
        raise ValueError(f"{context}: center_atom_idx {index} is outside [0, {len(atomic_numbers) - 1}].")
    if int(atomic_numbers[index]) <= 1:
        raise ValueError(f"{context}: center_atom_idx {index} must identify a heavy atom, not hydrogen.")
    return index


def validate_batch_centers(batch, *, allow_missing=False) -> None:
    """Protect direct model calls as well as CSV/dataset entry points."""
    centers = batch["center_indices"]
    raw = batch.get("center_atom_idx")
    n_conf = batch["mol_for_conf"].numel()
    if centers.numel() != n_conf:
        raise ValueError("center_indices must contain one index per conformer.")
    if raw is not None and raw.numel() != batch["task_id"].numel():
        raise ValueError("center_atom_idx must contain one index per molecule.")
    for conf_id in range(n_conf):
        atoms = torch.nonzero(batch["conf_batch"] == conf_id, as_tuple=False).flatten()
        if not atoms.numel():
            raise ValueError(f"No atoms found for conformer {conf_id}.")
        value = centers[conf_id].item()
        if not math.isfinite(float(value)) or float(value) != int(value):
            raise ValueError("center_indices must contain finite integers.")
        matches = torch.nonzero(atoms == int(value), as_tuple=False).flatten()
        if not matches.numel():
            raise ValueError(f"Center index {value} is outside conformer {conf_id}.")
        local = int(matches[0])
        if raw is not None:
            mol_id = int(batch["mol_for_conf"][conf_id])
            index = checked_center(raw[mol_id].item(), batch["z"][atoms],
                                   allow_missing=allow_missing, context=f"molecule {mol_id}")
            if index == -1:
                continue
            if index != local:
                raise ValueError(f"Inconsistent raw and batched center indices in molecule {mol_id}.")
        checked_center(local, batch["z"][atoms], context=f"conformer {conf_id}")
