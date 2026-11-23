from __future__ import annotations

import os
import argparse
import tempfile
from io import BytesIO
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "liten-pka-matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D


PACKAGE = Path(__file__).resolve().parents[1] / "results/figure5"
BASE = PACKAGE / "data/interpretability"
PRED = PACKAGE / "data/predictions"
OUT = PACKAGE / "figures"


MORANDI = {
    "pretrained": "#789A9A",
    "scratch": "#C48778",
    "spice": "#9A9788",
    "sand": "#D6C7A8",
    "sage": "#A8B59B",
    "slate": "#5E6D73",
    "ink": "#303436",
    "grid": "#D8D2C7",
    "light": "#F5F2EC",
}

PKA_CMAP = LinearSegmentedColormap.from_list(
    "morandi_pka",
    ["#4F6D7A", "#8FA79B", "#D8C59B", "#BF7E70"],
)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "font.weight": "regular",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "axes.edgecolor": MORANDI["ink"],
            "axes.labelcolor": MORANDI["ink"],
            "xtick.color": MORANDI["ink"],
            "ytick.color": MORANDI["ink"],
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def clean_group(group: str) -> str:
    if group == "input":
        return "Input"
    if group == "output_norm":
        return "Out"
    if group.startswith("layer_"):
        return "L" + group.split("_")[-1]
    return group


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.10,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=MORANDI["ink"],
    )


def fig_panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.055, dy: float = 0.014) -> None:
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + dx,
        bbox.y1 + dy,
        label,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=MORANDI["ink"],
    )


def ordered_param_frame() -> pd.DataFrame:
    df = pd.read_csv(BASE / "parameter_layer_metrics.csv")
    order = ["input"] + [f"layer_{idx}" for idx in range(1, 7)] + ["output_norm"]
    df = df[df["group"].isin(order)].copy()
    df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)
    return df.sort_values("group")


def plot_drift(ax: plt.Axes, param: pd.DataFrame) -> None:
    groups = list(param["group"].cat.categories)
    x = np.arange(len(groups))
    for comp, color, label in [
        ("pretrained_ft_vs_spice", MORANDI["pretrained"], "Pretrained fine-tuned"),
        ("scratch_final_vs_spice", MORANDI["scratch"], "Scratch trained"),
    ]:
        sub = param[param["comparison"] == comp].set_index("group").loc[groups]
        ax.plot(
            x,
            sub["relative_l2_drift"],
            marker="o",
            ms=5.6,
            lw=2.0,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=label,
        )
        ax.fill_between(x, sub["relative_l2_drift"], alpha=0.055, color=color, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([clean_group(g) for g in groups])
    ax.set_ylabel("Relative L2 drift to SPICE")
    ax.set_title("Parameter displacement", loc="left", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.28)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 1.055),
        ncol=2,
        handlelength=1.3,
        columnspacing=1.1,
        fontsize=6.4,
    )


def plot_cosine_heatmap(ax: plt.Axes, param: pd.DataFrame) -> None:
    groups = list(param["group"].cat.categories)
    rows = ["pretrained_ft_vs_spice", "scratch_final_vs_spice"]
    data = []
    for row in rows:
        sub = param[param["comparison"] == row].set_index("group").loc[groups]
        data.append(sub["cosine_similarity"].to_numpy())
    arr = np.vstack(data)
    cmap = LinearSegmentedColormap.from_list("morandi_cos", ["#EEE7DA", MORANDI["sage"], MORANDI["pretrained"]])
    im = ax.imshow(arr, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([clean_group(g) for g in groups])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Pretrained\nfine-tuned", "Scratch\ntrained"])
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.set_title("Directional alignment to SPICE", loc="left", fontsize=8, fontweight="bold")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            color = "white" if arr[i, j] > 0.75 else MORANDI["ink"]
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=6.2, fontweight="bold", color=color)
    cax = ax.inset_axes([1.025, 0.06, 0.030, 0.86])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Cosine", rotation=270, labelpad=7)
    cbar.ax.yaxis.label.set_fontweight("bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    cbar.outline.set_linewidth(0.4)


def plot_erank(ax: plt.Axes, param: pd.DataFrame) -> None:
    groups = list(param["group"].cat.categories)
    groups = [g for g in groups if g != "output_norm"]
    x = np.arange(len(groups))
    for comp, color, label in [
        ("spice_init_effective_rank", MORANDI["spice"], "SPICE init"),
        ("pretrained_ft_effective_rank", MORANDI["pretrained"], "Pretrained fine-tuned"),
        ("scratch_final_effective_rank", MORANDI["scratch"], "Scratch trained"),
    ]:
        sub = param[param["comparison"] == comp].set_index("group").loc[groups]
        ax.plot(
            x,
            sub["effective_rank"],
            marker="o",
            ms=5.0,
            lw=1.8,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.85,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([clean_group(g) for g in groups])
    ax.set_ylabel("Weighted effective rank")
    ax.set_title("Spectral complexity", loc="left", fontsize=8, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.09),
        ncol=3,
        handlelength=1.1,
        fontsize=6.15,
        columnspacing=0.75,
    )


def grouped_bar(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    embeds = ["graph_h", "site_h", "mol_repr"]
    labels = ["Graph", "Site", "Molecule"]
    x = np.arange(len(embeds))
    width = 0.34
    for offset, mode, color, label in [
        (-width / 2, "pretrained_ft", MORANDI["pretrained"], "Pretrained"),
        (width / 2, "scratch", MORANDI["scratch"], "Scratch"),
    ]:
        vals = [float(df[(df["mode"] == mode) & (df["embedding"] == e)][metric].iloc[0]) for e in embeds]
        bars = ax.bar(x + offset, vals, width=width, color=color, edgecolor="white", linewidth=0.75, label=label)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.018,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.7,
                fontweight="bold",
                color=MORANDI["ink"],
                clip_on=False,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.065),
        ncol=2,
        handlelength=1.0,
        fontsize=6.6,
        columnspacing=1.2,
    )


def plot_embedding_metrics(ax: plt.Axes) -> None:
    probe = pd.read_csv(BASE / "embedding_probe_metrics.csv")
    knn = pd.read_csv(BASE / "embedding_knn_metrics.csv")
    embeds = ["graph_h", "site_h", "mol_repr"]
    labels = ["Graph", "Site", "Molecule"]
    x = np.arange(len(embeds))
    width = 0.17
    specs = [
        (probe, "pretrained_ft", -1.5 * width, MORANDI["pretrained"], "Linear, pre"),
        (probe, "scratch", -0.5 * width, "#D7AAA0", "Linear, scratch"),
        (knn, "pretrained_ft", 0.5 * width, MORANDI["sage"], "kNN, pre"),
        (knn, "scratch", 1.5 * width, MORANDI["sand"], "kNN, scratch"),
    ]
    for frame, mode, offset, color, label in specs:
        vals = [float(frame[(frame["mode"] == mode) & (frame["embedding"] == e)]["mae"].iloc[0]) for e in embeds]
        ax.bar(x + offset, vals, width=width, color=color, edgecolor="white", linewidth=0.7, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("pKa MAE")
    ax.set_title("Frozen representation readout", loc="left", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.52)
    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.09),
        columnspacing=0.65,
        handlelength=0.9,
        fontsize=6.05,
    )


def plot_pairwise(ax: plt.Axes) -> None:
    pair = pd.read_csv(BASE / "embedding_pairwise_metrics.csv")
    grouped_bar(
        ax,
        pair.rename(columns={"spearman_rho": "rho"}),
        "rho",
        "pKa geometry in embedding space",
        "Spearman rho\ncosine distance vs |Delta pKa|",
    )
    ax.set_ylim(0, 0.66)


def prediction_cases(case_rows: list[int] | None = None, prediction_dir: Path | None = None) -> pd.DataFrame:
    prediction_dir = PRED if prediction_dir is None else prediction_dir
    pre = pd.read_csv(prediction_dir / "pretrained_predictions.csv")
    scratch = pd.read_csv(prediction_dir / "scratch_predictions.csv")
    cols = ["mol_id", "smiles", "pka", "task", "center_symbol", "center_atom_idx", "y_pred", "abs_error"]
    pd.testing.assert_frame_equal(pre[cols[:-2]], scratch[cols[:-2]], check_dtype=False)
    frame = pre[cols].rename(columns={"y_pred": "pred_pre", "abs_error": "err_pre"}).copy()
    frame["pred_scratch"] = scratch.y_pred.to_numpy()
    frame["err_scratch"] = scratch.abs_error.to_numpy()
    frame["gain"] = frame["err_scratch"] - frame["err_pre"]
    selected = ["42e16ecb4999446d", "20e7fc5be8b9ba30"]
    if case_rows is None:
        case_rows = [int(np.flatnonzero(frame.mol_id == key)[0]) for key in selected
                     if (frame.mol_id == key).sum() == 1]
        if len(case_rows) != 2:
            case_rows = [0, 1]
    if len(set(case_rows)) != 2 or any(i < 0 or i >= len(frame) for i in case_rows):
        raise ValueError("Specify two distinct, valid case-row indices.")
    return frame.iloc[case_rows].assign(input_row_index=case_rows).reset_index(drop=True)


def molecule_image(smiles: str, center_atom_idx: int, size: tuple[int, int] = (760, 420)) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2.2
    opts.minFontSize = 16
    opts.maxFontSize = 24
    opts.padding = 0.08
    highlight_atoms = [int(center_atom_idx)] if 0 <= int(center_atom_idx) < mol.GetNumAtoms() else []
    highlight_colors = {int(center_atom_idx): tuple(int(MORANDI["scratch"].lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))}
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii={int(center_atom_idx): 0.36} if highlight_atoms else {},
    )
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")


def case_attention(attention: pd.DataFrame, row: pd.Series) -> tuple[pd.DataFrame, list[int]]:
    required = {"mode", "mol_id", "smiles", "task", "center_atom_idx", "atom_idx", "symbol",
                "attention_strength", "prediction", "layer", "n_conformers", "n_heads",
                "checkpoint_sha256", "conformer_sha256"}
    if not required.issubset(attention.columns):
        raise ValueError("Use extract_atom_attention.py to generate actual model attention first.")
    scores = attention[(attention.mol_id.astype(str) == str(row.mol_id)) &
                       (attention.smiles == row.smiles) & (attention.task == row.task) &
                       (attention.center_atom_idx == row.center_atom_idx)].copy()
    if set(scores["mode"]) != {"pretrained_ft", "scratch"}:
        raise ValueError(f"Missing pretrained/scratch attention for {row.mol_id}.")
    if scores.duplicated(["mode", "atom_idx"]).any():
        raise ValueError("Ambiguous duplicate attention records for this molecule and site.")
    for key in ("layer", "conformer_sha256", "n_conformers", "n_heads"):
        if scores[key].nunique() != 1:
            raise ValueError(f"Attention comparisons require matching {key}.")
    mol = Chem.AddHs(Chem.MolFromSmiles(row.smiles))
    for atom in scores.itertuples():
        index = int(atom.atom_idx)
        if (index != atom.atom_idx or not 0 <= index < mol.GetNumAtoms() or
                index == int(row.center_atom_idx) or mol.GetAtomWithIdx(index).GetSymbol() != atom.symbol):
            raise ValueError("Attention atom indices do not match the case SMILES.")
    for mode, predicted in (("pretrained_ft", row.pred_pre), ("scratch", row.pred_scratch)):
        sub = scores[scores["mode"] == mode]
        if sub.checkpoint_sha256.nunique() != 1 or not np.allclose(sub.prediction, predicted, rtol=0, atol=5e-5):
            raise ValueError("Attention checkpoint predictions do not match Figure 5 source predictions.")
        values = sub.attention_strength.to_numpy()
        if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1.0):
            raise ValueError("Invalid normalized attention strengths.")
    # Both models show the same atom set; do not renormalize after selecting it.
    ranked = scores.groupby("atom_idx").attention_strength.sum().sort_values(ascending=False, kind="stable")
    return scores, ranked.head(6).index.astype(int).tolist()


def plot_atom_attention(ax: plt.Axes, row: pd.Series, scores: pd.DataFrame,
                        atom_indices: list[int], title: str, vmax: float) -> None:
    scores = scores.set_index("atom_idx").reindex(atom_indices)
    mol = Chem.AddHs(Chem.MolFromSmiles(row.smiles))
    ax.set_facecolor("#ECECEA")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.50,
        1.10,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.2,
        fontweight="bold",
        color=MORANDI["ink"],
        clip_on=False,
    )
    ax.scatter([0], [0], s=46, color="#C11FA8", edgecolor=MORANDI["ink"], lw=0.7, zorder=4)
    ax.text(0, -0.19, f"{row['center_symbol']}{int(row['center_atom_idx'])}", ha="center", va="top", fontsize=4.9, fontweight="bold")
    theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(scores), endpoint=False)
    radius = 0.92
    xs = radius * np.cos(theta)
    ys = radius * np.sin(theta)
    cmap = LinearSegmentedColormap.from_list("atom_attention", ["#D9D8CF", MORANDI["pretrained"], "#283A5A"])
    for i, (atom_idx, atom) in enumerate(scores.iterrows()):
        strength = 0.0 if pd.isna(atom["attention_strength"]) else float(atom["attention_strength"])
        scaled = strength / max(vmax, 1e-12)
        color = cmap(scaled)
        ax.plot([0, xs[i]], [0, ys[i]], color=color, lw=0.5 + 1.8 * scaled, alpha=0.95, zorder=1)
        ax.scatter([xs[i]], [ys[i]], s=26, color=color, edgecolor="white", lw=0.45, zorder=3)
        ha = "left" if xs[i] > 0.05 else "right" if xs[i] < -0.05 else "center"
        va = "bottom" if ys[i] > 0 else "top"
        ax.text(
            xs[i] * 1.15,
            ys[i] * 1.15,
            f"{mol.GetAtomWithIdx(int(atom_idx)).GetSymbol()}{int(atom_idx)}",
            ha=ha,
            va=va,
            fontsize=4.25,
            fontweight="bold",
            color=MORANDI["ink"],
        )
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.45, 1.55)


def plot_molecule_case(ax: plt.Axes, row: pd.Series, label: str, attention: pd.DataFrame) -> None:
    scores, atom_indices = case_attention(attention, row)
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    card = mpl.patches.FancyBboxPatch(
        (0.01, 0.03),
        0.98,
        0.92,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor="#FBFAF6",
        edgecolor="#E4DDD0",
        linewidth=0.8,
        zorder=0,
    )
    ax.add_patch(card)
    img = molecule_image(row["smiles"], int(row["center_atom_idx"]))
    img_ax = ax.inset_axes([0.025, 0.31, 0.42, 0.50])
    img_ax.imshow(img)
    img_ax.set_axis_off()
    img_ax.set_aspect("equal")
    vmax = float(scores.attention_strength.max())
    for x, mode, title in ((0.48, "pretrained_ft", "Pretrained"), (0.75, "scratch", "Scratch")):
        ctx_ax = ax.inset_axes([x, 0.40, 0.23, 0.34])
        plot_atom_attention(ctx_ax, row, scores[scores["mode"] == mode], atom_indices, title, vmax)
    ax.text(0.72, 0.835, f"Attention strength (L{int(scores.layer.iloc[0])})",
            transform=ax.transAxes, ha="center", fontsize=5.3, fontweight="bold")
    cmap = LinearSegmentedColormap.from_list("case_attention_scale", ["#D9D8CF", MORANDI["pretrained"], "#283A5A"])
    cax = ax.inset_axes([0.55, 0.29, 0.35, 0.025])
    cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, vmax), cmap=cmap),
                             cax=cax, orientation="horizontal", ticks=[0, vmax])
    cbar.ax.set_xticklabels(["0", f"{vmax:.2f}"])
    cbar.ax.tick_params(labelsize=4, length=1, pad=1)
    ax.text(
        0.040,
        0.94,
        f"Representative {row['task']} site: {row['center_symbol']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=MORANDI["ink"],
    )
    y_label = 0.190
    y_value = 0.082
    xs = [0.08, 0.36, 0.61]
    texts = [
        ("Exp.", f"{row['pka']:.2f}", MORANDI["spice"]),
        ("Pretrained", f"{row['pred_pre']:.2f}", MORANDI["pretrained"]),
        ("Scratch", f"{row['pred_scratch']:.2f}", MORANDI["scratch"]),
    ]
    for x, (name, value, color) in zip(xs, texts):
        ax.text(
            x,
            y_label,
            name,
            transform=ax.transAxes,
            fontsize=5.7,
            fontweight="bold",
            color=MORANDI["slate"],
            ha="left",
            va="bottom",
        )
        ax.text(
            x,
            y_value,
            value,
            transform=ax.transAxes,
            fontsize=10.2,
            fontweight="bold",
            color=color,
            ha="left",
            va="bottom",
        )
    ax.text(
        0.08,
        0.045,
        f"Error gain vs scratch: {row['gain']:.2f} pKa units",
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
        color=MORANDI["pretrained"],
        ha="left",
    )


def plot_pca_panel(spec: GridSpecFromSubplotSpec) -> list[plt.Axes]:
    pca = pd.read_csv(BASE / "test_site_h_pca.csv")
    modes = [("pretrained_ft", "Pretrained fine-tuned"), ("scratch", "Scratch trained")]
    axes = []
    vmin, vmax = np.percentile(pca["y"], [2, 98])
    for idx, (mode, title) in enumerate(modes):
        ax = plt.subplot(spec[idx])
        axes.append(ax)
        sub = pca[pca["mode"] == mode]
        sc = ax.scatter(
            sub["pca1"],
            sub["pca2"],
            c=sub["y"],
            cmap=PKA_CMAP,
            vmin=vmin,
            vmax=vmax,
            s=5,
            alpha=0.58,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(title, fontsize=7.2, loc="left", pad=2)
        ax.set_xlabel("PC1")
        if idx == 0:
            ax.set_ylabel("PC2")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.grid(False)
    axes[0].text(
        0,
        1.18,
        "Site embedding PCA",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=MORANDI["ink"],
    )
    cax = axes[1].inset_axes([1.030, 0.18, 0.050, 0.66])
    cbar = plt.colorbar(sc, cax=cax)
    cbar.ax.set_title("pKa", fontsize=6.5, pad=3, fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    cbar.outline.set_linewidth(0.4)
    return axes


def main() -> None:
    global BASE, PRED, OUT
    parser = argparse.ArgumentParser(description="Replot Figure 5 from its frozen source tables (no model training).")
    parser.add_argument("--data-dir", type=Path, default=PACKAGE / "data")
    parser.add_argument("--out-dir", type=Path, default=Path("results/reproduced/figure5"))
    parser.add_argument("--case-rows", type=int, nargs=2)
    parser.add_argument("--attention-csv", type=Path,
                        help="atom_attention.csv from extract_atom_attention.py using matching checkpoints/caches.")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg", "tiff"], default=["png", "pdf", "svg"])
    args = parser.parse_args()
    BASE, PRED, OUT = args.data_dir / "interpretability", args.data_dir / "predictions", args.out_dir
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)

    param = ordered_param_frame()
    fig = plt.figure(figsize=(7.20, 9.15), constrained_layout=False)
    gs = GridSpec(
        4,
        2,
        figure=fig,
        height_ratios=[0.88, 0.92, 1.05, 1.08],
        width_ratios=[1.0, 1.0],
        hspace=0.54,
        wspace=0.42,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    gs_f = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 1], wspace=0.16)
    ax_g = fig.add_subplot(gs[3, 0])
    ax_h = fig.add_subplot(gs[3, 1])

    plot_drift(ax_a, param)
    panel_label(ax_a, "A")
    plot_cosine_heatmap(ax_b, param)
    plot_erank(ax_c, param)
    panel_label(ax_c, "C")
    plot_embedding_metrics(ax_d)
    plot_pairwise(ax_e)
    panel_label(ax_e, "E")
    axes_f = plot_pca_panel(gs_f)
    cases = prediction_cases(args.case_rows)
    attention_path = args.attention_csv or BASE / "attention/atom_attention.csv"
    if not attention_path.is_file():
        raise FileNotFoundError("Real model attention is required. Run extract_figure5.py or extract_atom_attention.py first.")
    attention = pd.read_csv(attention_path)
    plot_molecule_case(ax_g, cases.iloc[0], "G", attention)
    panel_label(ax_g, "G")
    plot_molecule_case(ax_h, cases.iloc[1], "H", attention)

    fig.suptitle(
        "Machine-learned potential pretraining yields pKa-aware LiTEN representations",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=9.2,
        fontweight="bold",
        color=MORANDI["ink"],
    )
    fig.subplots_adjust(top=0.945, bottom=0.045, left=0.085, right=0.935)
    fig_panel_label(fig, ax_b, "B", dx=-0.055, dy=0.014)
    fig_panel_label(fig, ax_d, "D", dx=-0.055, dy=0.014)
    fig_panel_label(fig, axes_f[0], "F", dx=-0.055, dy=0.030)
    fig_panel_label(fig, ax_h, "H", dx=-0.055, dy=0.014)

    stem = OUT / "figure5"
    for extension in args.formats:
        fig.savefig(f"{stem}.{extension}", dpi=600 if extension == "tiff" else 450, bbox_inches="tight")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
