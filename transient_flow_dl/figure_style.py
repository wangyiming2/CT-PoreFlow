"""Shared Nature-style visual contract for paper-facing matplotlib figures.

The numerical analysis stays in the calling scripts.  This module owns only
typography, semantic method identity, panel labels and publication exports so
the same model never changes colour or line grammar between figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


METHOD_STYLES: dict[str, dict[str, object]] = {
    # NPG/Wong-inspired colourblind-safe identities.  The redundant marker and
    # line-style encodings keep the figures interpretable in grayscale.
    "Time-U-Net": {"color": "#4DBBD5", "marker": "o", "linestyle": "-"},
    "Time-FNO": {"color": "#3C5488", "marker": "s", "linestyle": "--"},
    "DeepONet": {"color": "#8491B4", "marker": "^", "linestyle": "-."},
    "CT-PoreFlow": {"color": "#E64B35", "marker": "D", "linestyle": "-"},
    "Topology-Time-U-Net": {"color": "#3F4B73", "marker": "o", "linestyle": "-"},
    "No spectral bottleneck": {"color": "#60759D", "marker": "s", "linestyle": "--"},
    "No directed geodesics": {"color": "#7E91B8", "marker": "^", "linestyle": "-."},
    "No steady anchor": {"color": "#9EACCA", "marker": "v", "linestyle": ":"},
    "Linear time": {"color": "#6F7189", "marker": "P", "linestyle": "--"},
    "Data loss only": {"color": "#9C86A5", "marker": "X", "linestyle": "-."},
}

FALLBACK_COLORS = ("#4DBBD5", "#3C5488", "#8491B4", "#00A087", "#F39B7F", "#E64B35")
FALLBACK_MARKERS = ("o", "s", "^", "v", "P", "D")
FALLBACK_LINESTYLES = ("-", "--", "-.", ":")


def apply_nature_style(font_size: float = 8.5) -> None:
    """Apply the paper-wide Python backend contract before creating a figure."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": font_size,
            "mathtext.fontset": "stix",
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 0.5,
            "axes.titleweight": "bold",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#30343B",
            "axes.labelcolor": "#20242B",
            "xtick.labelsize": max(font_size - 0.25, 7.5),
            "ytick.labelsize": max(font_size - 0.25, 7.5),
            "xtick.color": "#30343B",
            "ytick.color": "#30343B",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": True,
            "legend.framealpha": 0.88,
            "legend.edgecolor": "#B7BCC5",
            "legend.fancybox": False,
            "legend.fontsize": max(font_size - 0.25, 7.5),
            "lines.linewidth": 1.55,
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
        }
    )


def method_style(label: str, index: int = 0) -> dict[str, object]:
    """Return a redundant colour/marker/line identity for one method."""

    if label in METHOD_STYLES:
        return dict(METHOD_STYLES[label])
    return {
        "color": FALLBACK_COLORS[index % len(FALLBACK_COLORS)],
        "marker": FALLBACK_MARKERS[index % len(FALLBACK_MARKERS)],
        "linestyle": FALLBACK_LINESTYLES[index % len(FALLBACK_LINESTYLES)],
    }


def add_panel_label(axis, label: str, x: float = -0.12, y: float = 1.04) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color="#111827",
    )


def soften_axis(axis) -> None:
    """Nature-style axis cleanup without a decorative dashboard grid."""

    axis.grid(False)
    axis.tick_params(direction="in", which="both")
    axis.spines["left"].set_color("#4B5563")
    axis.spines["bottom"].set_color("#4B5563")


def save_publication_figure(
    fig,
    path: Path,
    *,
    dpi: int = 600,
    include_tiff: bool = False,
) -> list[Path]:
    """Save editable SVG/PDF plus the high-resolution raster used by Word."""

    base = Path(path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [base.with_suffix(".svg"), base.with_suffix(".pdf"), base.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outputs[2], dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    if include_tiff:
        tiff = base.with_suffix(".tiff")
        fig.savefig(
            tiff,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        outputs.append(tiff)
    return outputs


def close_figure(fig) -> None:
    plt.close(fig)
