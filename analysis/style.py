"""Publication figure style, in one place.

Every constraint here comes from the paper this lab feeds, and each one is a
real constraint rather than a preference:

  * **Vector PDF at exact column widths.** 3.3in single-column, 7in
    double-column. A figure sized in the plotting script and then scaled in
    LaTeX has text at the wrong size, which is the single most common way a
    good figure reads as amateur.
  * **Serif type at the body's own size.** The figure's tick labels should
    not be visibly a different typeface from the caption beneath them.
  * **Colourblind-safe, and legible in greyscale.** That is why FRAMEWORK is
    encoded by MARKER SHAPE and REPRESENTATION by COLOUR: a reader with
    deuteranopia, or a reader holding a photocopy, can still separate the
    three backends, because shape survives both. Colour then carries the
    axis where a collision is less costly.
  * **Direct labels over legends.** A legend forces a saccade per data
    point; a label at the end of a line does not. Legends appear only where
    a direct label would collide.

The palette is Okabe-Ito, which is designed for exactly this and is
distinguishable under all three common forms of colour vision deficiency.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SINGLE_COL = 3.3
DOUBLE_COL = 7.0

#: Okabe-Ito. Black is reserved for the FP16 reference line, so it is not in
#: the rotation.
OKABE_ITO = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000",
}

#: Colour carries REPRESENTATION - what the weights actually are.
#: `fp16_baseline` is the reference every other class is normalised against,
#: so it takes the neutral black; the quantized classes take saturated hues.
REPRESENTATION_COLOR = {
    "fp16_baseline": OKABE_ITO["black"],
    "fp8_e4m3": OKABE_ITO["sky"],
    "w8a16_int": OKABE_ITO["green"],
    "w8a8_int": OKABE_ITO["blue"],
    "w4_grouped": OKABE_ITO["vermillion"],
    "w4_grouped_mixed": OKABE_ITO["orange"],
    "unclassified": OKABE_ITO["purple"],
}

#: Shape carries FRAMEWORK, so the three backends stay separable in
#: greyscale and under any colour vision deficiency.
FRAMEWORK_MARKER = {
    "vllm": "o",
    "llama-cpp": "s",
    "edge-llm": "^",
}
FRAMEWORK_LABEL = {
    "vllm": "vLLM",
    "llama-cpp": "llama.cpp",
    "edge-llm": "Edge-LLM",
}


def apply() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # TrueType, not Type 3 - many venues reject Type 3
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "errorbar.capsize": 2,
    })


def color_for(row: dict) -> str:
    return REPRESENTATION_COLOR.get(
        (row.get("representation") or {}).get("comparability_class"), OKABE_ITO["purple"])


def marker_for(row: dict) -> str:
    return FRAMEWORK_MARKER.get(row.get("framework"), "D")


def label_for(row: dict) -> str:
    """One short label per config, used for direct labelling."""
    rep = row.get("representation") or {}
    fw = FRAMEWORK_LABEL.get(row.get("framework"), row.get("framework"))
    bits = rep.get("effective_bits_per_weight")
    return f"{fw} {bits:.1f}b" if bits else f"{fw}"


def direct_label(ax, x, y, text, color, dx=4, dy=0, **kw):
    """A label placed next to the last point of a series, in the series'
    own colour, instead of a legend entry."""
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                color=color, fontsize=7, va="center", **kw)
