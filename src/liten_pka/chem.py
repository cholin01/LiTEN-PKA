from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("RDKit is required for LiTEN-PKA preprocessing.") from exc


ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5, 6],
    "formal_charge": [-2, -1, 0, 1, 2],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

ELECTRONEGATIVITY = {
    "C": 6.27,
    "N": 7.30,
    "O": 7.54,
    "F": 10.41,
    "H": 7.18,
    "Cl": 8.30,
    "S": 6.22,
    "Br": 7.59,
    "I": 6.76,
    "P": 5.62,
    "B": 4.29,
    "Si": 4.77,
    "Se": 5.89,
    "As": 5.30,
    "Min": 4.29,
    "Range": 6.12,
}

HARDNESS = {
    "C": 5.00,
    "N": 7.23,
    "O": 6.08,
    "F": 7.01,
    "H": 6.43,
    "S": 4.14,
    "Cl": 4.68,
    "Br": 4.22,
    "I": 3.69,
    "P": 4.88,
    "B": 4.01,
    "Si": 3.38,
    "Se": 3.87,
    "As": 4.50,
    "Min": 3.38,
    "Range": 3.85,
}

ATOM_DIAMETER = {
    "C": 75.0,
    "N": 71.0,
    "O": 63.0,
    "F": 64.0,
    "H": 32.0,
    "S": 103.0,
    "Cl": 99.0,
    "Br": 114.0,
    "I": 133.0,
    "P": 111.0,
    "B": 85.0,
    "Si": 116.0,
    "Se": 116.0,
    "As": 121.0,
    "Min": 32.0,
    "Range": 101.0,
}

ELEMENTS_PKALEARN = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
PKALEARN_FEATURE_SETS = {
    "element",
    "pkalearn_element",
    "default",
    "pkalearn_default",
    "all",
    "pkalearn_all",
    "phys",
    "pkalearn_phys",
    "chem",
    "pkalearn_chem",
}


@dataclass(frozen=True)
class MoleculeRecord:
    mol_id: str
    smiles: str
    canonical_smiles: str
    pka: float | None = None
    center_atom_idx: int | None = None
    task: str = "acid"


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def canonicalize_smiles(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def stable_mol_id(smiles: str) -> str:
    canonical = canonicalize_smiles(smiles)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def atom_numbers(mol: Chem.Mol) -> np.ndarray:
    return np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)


def one_hot(value: object, choices: list[object]) -> list[float]:
    if value not in choices:
        value = choices[-1]
    return [float(value == choice) for choice in choices]


def explicit_h_mol(mol: Chem.Mol) -> Chem.Mol:
    return Chem.AddHs(Chem.Mol(mol))


def clear_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def pkalearn_atom_features_from_mol(
    mol: Chem.Mol,
    feature_set: str = "none",
    center_atom_idx: int = -1,
    include_center_feature: bool = False,
) -> np.ndarray:
    feature_set = str(feature_set or "none").lower()
    mol = explicit_h_mol(mol)
    if feature_set in {"", "none", "off", "false"}:
        dim = 1 if include_center_feature else 0
        rows = np.zeros((mol.GetNumAtoms(), dim), dtype=np.float32)
        if include_center_feature and 0 <= int(center_atom_idx) < mol.GetNumAtoms():
            rows[int(center_atom_idx), 0] = 1.0
        return rows

    no_element = feature_set.endswith("_noelement")
    base_feature_set = feature_set.removesuffix("_noelement")
    if base_feature_set not in PKALEARN_FEATURE_SETS:
        raise ValueError(f"Unknown atom feature set: {feature_set}")
    use_element = not no_element and base_feature_set in {
        "element",
        "pkalearn_element",
        "default",
        "pkalearn_default",
        "all",
        "pkalearn_all",
        "phys",
        "pkalearn_phys",
        "chem",
        "pkalearn_chem",
    }
    use_number_of_rings = base_feature_set in {"all", "pkalearn_all"}
    use_phys = base_feature_set in {"default", "pkalearn_default", "all", "pkalearn_all", "phys", "pkalearn_phys"}
    use_chem = base_feature_set in {"default", "pkalearn_default", "all", "pkalearn_all", "chem", "pkalearn_chem"}

    ring = mol.GetRingInfo()
    rows: list[list[float]] = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        feats: list[float] = []
        if use_element:
            feats += one_hot(symbol, ELEMENTS_PKALEARN)
        if use_phys:
            feats.append((ELECTRONEGATIVITY.get(symbol, ELECTRONEGATIVITY["Min"]) - ELECTRONEGATIVITY["Min"]) / ELECTRONEGATIVITY["Range"])
            feats.append((HARDNESS.get(symbol, HARDNESS["Min"]) - HARDNESS["Min"]) / HARDNESS["Range"])
            feats.append((ATOM_DIAMETER.get(symbol, ATOM_DIAMETER["Min"]) - ATOM_DIAMETER["Min"]) / ATOM_DIAMETER["Range"])
        if use_chem:
            feats += one_hot(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                ],
            )
            feats.append(float(atom.GetIsAromatic()))
        if use_number_of_rings:
            num_rings = ring.NumAtomRings(atom.GetIdx())
            feats += one_hot(num_rings if num_rings < 2 else 2, [0, 1, 2])
        if use_chem:
            feats += [
                float(ring.IsAtomInRingOfSize(atom.GetIdx(), 3)),
                float(ring.IsAtomInRingOfSize(atom.GetIdx(), 4)),
                float(ring.IsAtomInRingOfSize(atom.GetIdx(), 5)),
                float(any(ring.IsAtomInRingOfSize(atom.GetIdx(), size) for size in range(6, 11))),
            ]
            h_count = atom.GetTotalNumHs()
            feats += one_hot(h_count if h_count < 3 else 3, [0, 1, 2, 3])
            charge = atom.GetFormalCharge()
            feats += one_hot(charge if charge in {-1, 0, 1} else 1, [-1, 0, 1])
        if include_center_feature:
            feats.append(float(atom.GetIdx() == int(center_atom_idx)))
        rows.append(feats)

    return np.asarray(rows, dtype=np.float32)


def pkalearn_atom_features(
    smiles: str,
    feature_set: str = "none",
    center_atom_idx: int = -1,
    include_center_feature: bool = False,
) -> np.ndarray:
    return pkalearn_atom_features_from_mol(
        mol_from_smiles(smiles),
        feature_set=feature_set,
        center_atom_idx=center_atom_idx,
        include_center_feature=include_center_feature,
    )


def pkalearn_atom_feature_dim(feature_set: str = "none", include_center_feature: bool = False) -> int:
    feature_set = str(feature_set or "none").lower()
    if feature_set in {"", "none", "off", "false"}:
        return 1 if include_center_feature else 0
    no_element = feature_set.endswith("_noelement")
    base_feature_set = feature_set.removesuffix("_noelement")
    if base_feature_set not in PKALEARN_FEATURE_SETS:
        raise ValueError(f"Unknown atom feature set: {feature_set}")
    dim = 0
    if not no_element and base_feature_set in {
        "element",
        "pkalearn_element",
        "default",
        "pkalearn_default",
        "all",
        "pkalearn_all",
        "phys",
        "pkalearn_phys",
        "chem",
        "pkalearn_chem",
    }:
        dim += len(ELEMENTS_PKALEARN)
    if base_feature_set in {"default", "pkalearn_default", "all", "pkalearn_all", "phys", "pkalearn_phys"}:
        dim += 3
    if base_feature_set in {"default", "pkalearn_default", "all", "pkalearn_all", "chem", "pkalearn_chem"}:
        dim += 3 + 1 + 4 + 4 + 3
    if base_feature_set in {"all", "pkalearn_all"}:
        dim += 3
    if include_center_feature:
        dim += 1
    return dim


def embed_rdkit_conformers(
    smiles: str,
    n_conformers: int = 5,
    seed: int = 2026,
    max_attempts: int = 1000,
    prune_rms_thresh: float = 0.25,
) -> tuple[Chem.Mol, np.ndarray, np.ndarray]:
    mol = Chem.AddHs(mol_from_smiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = int(max_attempts)
    params.pruneRmsThresh = float(prune_rms_thresh)
    params.useRandomCoords = True
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not ids:
        raise ValueError(f"RDKit could not embed any conformer for {smiles}")

    attempts = 0
    while len(ids) < n_conformers and attempts < max(20, n_conformers * 10):
        attempts += 1
        params.randomSeed += 17
        if hasattr(params, "clearConfs"):
            params.clearConfs = False
        params.pruneRmsThresh = -1.0
        extra = list(AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params))
        if extra:
            ids.extend(extra)

    energies: list[float] = []
    positions: list[np.ndarray] = []
    for conf_id in ids[:n_conformers]:
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id) if props else None
            if ff is not None:
                ff.Minimize(maxIts=500)
                energies.append(float(ff.CalcEnergy()))
            else:
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                energies.append(float(ff.CalcEnergy()) if ff is not None else np.nan)
        except Exception:
            energies.append(np.nan)
        conf = mol.GetConformer(conf_id)
        positions.append(np.asarray(conf.GetPositions(), dtype=np.float32))

    if len(positions) != n_conformers:
        raise ValueError(f"Only generated {len(positions)} conformers for {smiles}; expected {n_conformers}.")

    return mol, np.stack(positions, axis=0), np.asarray(energies, dtype=np.float32)
