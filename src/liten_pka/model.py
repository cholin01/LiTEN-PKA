from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chem import pkalearn_atom_feature_dim


def mean_pool(x: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = x.new_zeros((dim_size, x.size(-1)))
    out.index_add_(0, index, x)
    count = x.new_zeros((dim_size, 1))
    count.index_add_(0, index, torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype))
    return out / count.clamp_min(1.0)


class LiTENNodeEncoder(nn.Module):
    def __init__(self, config: dict[str, Any], model_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        model_config = model_config or {}
        root = Path(config.get("root", "vendor/LiTEN")).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from model.LiTEN import LiTEN  # type: ignore

        self.backbone = LiTEN(
            num_heads=int(config.get("num_heads", 8)),
            num_layers=int(config.get("num_layers", 6)),
            hidden_channels=int(config.get("hidden_channels", 256)),
            num_rbf=int(config.get("num_rbf", 32)),
            cutoff=float(config.get("cutoff", 5.0)),
            max_neighbors=int(config.get("max_neighbors", 32)),
            max_z=int(config.get("max_z", 100)),
        )
        self.atom_feature_set = str(model_config.get("atom_feature_set", "none") or "none")
        self.atom_feature_include_center = bool(model_config.get("atom_feature_include_center", False))
        self.atom_feature_dim = pkalearn_atom_feature_dim(
            self.atom_feature_set,
            include_center_feature=self.atom_feature_include_center,
        )
        self.atom_feature_projection = (
            nn.Linear(self.atom_feature_dim, self.hidden_channels) if self.atom_feature_dim > 0 else None
        )
        checkpoint = str(config.get("checkpoint", "") or "")
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            state_dict = state.state_dict() if hasattr(state, "state_dict") else state.get("model", state)
            state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            current_state = self.backbone.state_dict()
            compatible_state = {
                key: value
                for key, value in state_dict.items()
                if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
            }
            skipped = sorted(set(state_dict) - set(compatible_state))
            missing, _ = self.backbone.load_state_dict(compatible_state, strict=False)
            if skipped:
                warnings.warn(
                    f"Skipped {len(skipped)} LiTEN checkpoint tensors that are absent or shape-mismatched "
                    f"for this backbone, examples: {skipped[:5]}."
                )
            if len(compatible_state) < len(current_state) // 2:
                raise ValueError("Checkpoint did not match most LiTEN backbone parameters.")
        if bool(config.get("freeze_backbone", False)):
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    @property
    def hidden_channels(self) -> int:
        return int(self.backbone.hidden_channels)

    def forward_nodes(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        atom_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = self.backbone
        if self.atom_feature_projection is None:
            node_scalar = b.embedding(z)
        else:
            if atom_features is None or atom_features.size(-1) != self.atom_feature_dim:
                got = None if atom_features is None else atom_features.size(-1)
                raise ValueError(f"Expected atom_features with dim={self.atom_feature_dim}, got {got}.")
            node_scalar = self.atom_feature_projection(atom_features.to(dtype=torch.float32))
        edge_index, dist, edge_vector = b.connect(pos, batch)
        edge_feats = b.edge_embedding(b.radial_fn(dist))
        node_vector = torch.zeros(node_scalar.size(0), 3, node_scalar.size(1), device=node_scalar.device)

        for idx, interaction in enumerate(b.Interaction):
            node_scalar, node_vector, edge_feats = interaction(
                dist=dist,
                node_scalar=node_scalar,
                node_vector=node_vector,
                edge_index=edge_index,
                edge_feats=edge_feats,
                edge_vector=edge_vector,
                update_edge=idx < len(b.Interaction) - 1,
            )
        return b.out_norm(node_scalar)


class LiTENPKA(nn.Module):
    def __init__(self, liten_config: dict[str, Any], model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = LiTENNodeEncoder(liten_config, model_config)
        hidden = self.encoder.hidden_channels
        task_dim = int(model_config.get("task_embedding_dim", 16))
        self.task_embedding = nn.Embedding(4, task_dim)
        self.conf_pool = str(model_config.get("conformer_pool", "node_mean"))
        self.use_site_readout = bool(model_config.get("use_site_readout", True))
        dropout = float(model_config.get("dropout", 0.1))

        head_in = hidden * 2 + task_dim
        self.conformer_head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.attention = nn.Sequential(
            nn.Linear(head_in, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        z = batch["z"]
        pos = batch["pos"]
        atom_features = batch.get("atom_features")
        conf_batch = batch["conf_batch"]
        mol_for_conf = batch["mol_for_conf"]
        center_indices = batch["center_indices"]
        raw_center_idx = batch.get("center_atom_idx")
        task_id = batch["task_id"]

        node_h = self.encoder.forward_nodes(z, pos, conf_batch, atom_features=atom_features)
        n_conf = int(mol_for_conf.numel())
        n_mol = int(task_id.numel())
        if self.conf_pool == "node_mean":
            mol_reprs = []
            for mol_idx in range(n_mol):
                conf_ids = torch.nonzero(mol_for_conf == mol_idx, as_tuple=False).flatten()
                if conf_ids.numel() == 0:
                    raise ValueError(f"No conformers found for molecule index {mol_idx}.")
                conf_nodes = []
                local_center = 0
                for j, conf_id in enumerate(conf_ids.tolist()):
                    atom_indices = torch.nonzero(conf_batch == conf_id, as_tuple=False).flatten()
                    if atom_indices.numel() == 0:
                        raise ValueError(f"No atoms found for conformer index {conf_id}.")
                    if self.use_site_readout and j == 0:
                        local_center = int((center_indices[conf_id] - atom_indices[0]).item())
                    conf_nodes.append(node_h[atom_indices])
                n_atoms = conf_nodes[0].size(0)
                if any(nodes.size(0) != n_atoms for nodes in conf_nodes):
                    raise ValueError("node_mean pooling requires matching atom counts across conformers.")
                atom_h = torch.stack(conf_nodes, dim=0).mean(dim=0)
                graph_h = atom_h.mean(dim=0)
                if self.use_site_readout:
                    center = max(0, min(local_center, n_atoms - 1))
                    site_h = atom_h[center]
                else:
                    site_h = torch.zeros_like(graph_h)
                task_h = self.task_embedding(task_id[mol_idx])
                mol_reprs.append(torch.cat([graph_h, site_h, task_h], dim=-1))
            return self.conformer_head(torch.stack(mol_reprs, dim=0)).squeeze(-1)

        graph_h = mean_pool(node_h, conf_batch, n_conf)
        site_h = node_h[center_indices] if self.use_site_readout else torch.zeros_like(graph_h)
        task_h = self.task_embedding(task_id)[mol_for_conf]
        conf_h = torch.cat([graph_h, site_h, task_h], dim=-1)
        conf_pred = self.conformer_head(conf_h).squeeze(-1)

        if self.conf_pool == "mean":
            out = conf_pred.new_zeros(n_mol)
            out.index_add_(0, mol_for_conf, conf_pred)
            count = conf_pred.new_zeros(n_mol)
            count.index_add_(0, mol_for_conf, torch.ones_like(conf_pred))
            return out / count.clamp_min(1.0)

        if self.conf_pool != "attention":
            raise ValueError(f"Unknown conformer_pool: {self.conf_pool}")
        scores = self.attention(conf_h).squeeze(-1)
        weights = torch.zeros_like(scores)
        for mol_idx in range(n_mol):
            mask = mol_for_conf == mol_idx
            weights[mask] = torch.softmax(scores[mask], dim=0)
        out = conf_pred.new_zeros(n_mol)
        out.index_add_(0, mol_for_conf, conf_pred * weights)
        return out


class LiTENJointSitePKA(nn.Module):
    def __init__(self, liten_config: dict[str, Any], model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = LiTENNodeEncoder(liten_config, model_config)
        hidden = self.encoder.hidden_channels
        task_dim = int(model_config.get("task_embedding_dim", 16))
        self.task_embedding = nn.Embedding(4, task_dim)
        dropout = float(model_config.get("dropout", 0.1))

        self.site_head = nn.Sequential(
            nn.Linear(hidden + task_dim, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.pka_head = nn.Sequential(
            nn.Linear(hidden * 2 + task_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z = batch["z"]
        pos = batch["pos"]
        atom_features = batch.get("atom_features")
        conf_batch = batch["conf_batch"]
        mol_for_conf = batch["mol_for_conf"]
        center_indices = batch["center_indices"]
        raw_center_idx = batch.get("center_atom_idx")
        task_id = batch["task_id"]

        node_h = self.encoder.forward_nodes(z, pos, conf_batch, atom_features=atom_features)
        n_mol = int(task_id.numel())
        pka_preds = []
        site_losses = []
        site_correct = []
        pred_centers = []

        for mol_idx in range(n_mol):
            conf_ids = torch.nonzero(mol_for_conf == mol_idx, as_tuple=False).flatten()
            if conf_ids.numel() == 0:
                raise ValueError(f"No conformers found for molecule index {mol_idx}.")

            conf_nodes = []
            fallback_center = 0
            for j, conf_id in enumerate(conf_ids.tolist()):
                atom_indices = torch.nonzero(conf_batch == conf_id, as_tuple=False).flatten()
                if atom_indices.numel() == 0:
                    raise ValueError(f"No atoms found for conformer index {conf_id}.")
                if j == 0:
                    fallback_center = int((center_indices[conf_id] - atom_indices[0]).item())
                conf_nodes.append(node_h[atom_indices])

            n_atoms = conf_nodes[0].size(0)
            if any(nodes.size(0) != n_atoms for nodes in conf_nodes):
                raise ValueError("Joint site-pKa model requires matching atom counts across conformers.")
            target_center = None
            if raw_center_idx is not None:
                raw_center = int(raw_center_idx[mol_idx].item())
                if raw_center >= 0:
                    target_center = max(0, min(raw_center, n_atoms - 1))
            else:
                target_center = max(0, min(fallback_center, n_atoms - 1))

            atom_h = torch.stack(conf_nodes, dim=0).mean(dim=0)
            graph_h = atom_h.mean(dim=0)
            task_h = self.task_embedding(task_id[mol_idx])
            task_atoms = task_h.unsqueeze(0).expand(n_atoms, -1)
            site_logits = self.site_head(torch.cat([atom_h, task_atoms], dim=-1)).squeeze(-1)
            pred_center = int(torch.argmax(site_logits.detach()).item())
            pred_centers.append(pred_center)
            if target_center is not None:
                target = torch.tensor([target_center], dtype=torch.long, device=site_logits.device)
                site_losses.append(F.cross_entropy(site_logits.unsqueeze(0), target))
                site_correct.append(float(pred_center == target_center))
            site_h = atom_h[pred_center]
            pka_preds.append(self.pka_head(torch.cat([graph_h, site_h, task_h], dim=-1)).squeeze(-1))

        return {
            "pka_pred": torch.stack(pka_preds, dim=0),
            "site_loss": torch.stack(site_losses).mean() if site_losses else node_h.new_tensor(0.0),
            "site_acc": torch.tensor(site_correct, dtype=node_h.dtype, device=node_h.device).mean() if site_correct else node_h.new_tensor(float("nan")),
            "site_correct_count": node_h.new_tensor(int(sum(site_correct)), dtype=torch.long),
            "site_count": node_h.new_tensor(len(site_correct), dtype=torch.long),
            "pred_center_idx": torch.tensor(pred_centers, dtype=torch.long, device=node_h.device),
        }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved
