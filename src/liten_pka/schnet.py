"""Archived site-aware SchNet baseline architecture, without training side effects."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import SchNet


class SiteSchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SchNet(
            hidden_channels=256, num_filters=256, num_interactions=6,
            num_gaussians=32, cutoff=5.0, max_num_neighbors=32, readout="mean",
        )
        self.task_embedding = nn.Embedding(4, 16)
        self.head = nn.Sequential(
            nn.Linear(528, 256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1),
        )

    def forward(self, batch):
        z, pos, conf_batch = batch["z"], batch["pos"], batch["conf_batch"]
        h = self.backbone.embedding(z)
        edge_index, distances = self.backbone.interaction_graph(pos, conf_batch)
        edge_attr = self.backbone.distance_expansion(distances)
        for interaction in self.backbone.interactions:
            h = h + interaction(h, edge_index, distances, edge_attr)
        graph_h = global_mean_pool(h, conf_batch)
        site_h = h[batch["center_indices"]]
        mol_for_conf = batch["mol_for_conf"]
        task_h = self.task_embedding(batch["task_id"])[mol_for_conf]
        pred = self.head(torch.cat([graph_h, site_h, task_h], dim=-1)).squeeze(-1)
        # Preserve the baseline's original prediction-level conformer averaging.
        out = pred.new_zeros(batch["task_id"].numel())
        count = torch.zeros_like(out)
        out.index_add_(0, mol_for_conf, pred)
        count.index_add_(0, mol_for_conf, torch.ones_like(pred))
        return out / count.clamp_min(1)
