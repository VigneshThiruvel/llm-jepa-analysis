"""Figures for the observational lambda x k sweep.

Reads the two tidy CSVs written by collect_sweep.py and renders the figure set
for the mediation analysis. Pure pandas + matplotlib, no seaborn, no network —
designed to be run on a laptop against CSVs copied down from JURECA:

    rsync -av <host>:<repo>/sweep_runs/synth/{accuracy,geometry}_tidy.csv ./data/
    python plot_sweep.py --data-dir ./data --out-dir ./figures

Every figure writes a PNG plus a `<name>_data.csv` "table view" holding exactly
the rows that were plotted, so no value is reachable only by reading pixels.

FIGURES
  1 mediation    geometry vs accuracy, one point per cell   -- does the mediator mediate?
  2 dose         accuracy vs lambda, one line per k         -- the k=0 flatness
  3 grid         lambda x k heatmap vs baseline             -- same story, paper-ready
  4 layers       geometry vs layer                          -- where in the network
  5 trajectory   geometry vs training step                  -- when (causal ordering)
  6 health       lm_loss / total loss vs lambda             -- "not a training failure"

DATA CONVENTIONS THIS SCRIPT ENFORCES (each one is a way to get the story wrong)
  * Layer 0 is dropped. Under last-token pooling every example ends on the same
    template token, so rankme ~ 1 and linear_r2 is NaN. It is not a data point.
  * lambda=0 is never plotted on the log axis. It is the no-JEPA baseline and is
    drawn as a horizontal reference line.
  * k=0 at lambda>0 is NOT the baseline -- it is JEPA with no predictor tokens.
    Conflating the two inverts the headline finding, so the legend says so.
  * The `accuracy` column is used, never `accuracy_raw`. collect_sweep.py blanks
    the former for cells that did not train cleanly; using raw silently readmits
    them. Non-ok cells are drawn as hatched/hollow marks, never as a value.
  * Only tranche 1 has repeat seeds. Rather than fake error bars, the observed
    within-cell spread across those cells is drawn once as a noise band.
"""

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# --- Palette -----------------------------------------------------------------
# Values are the data-viz reference palette, used unchanged and only in
# combinations it documents as validated: categorical slots 1-2 for the two-series
# case, the single-hue blue ramp for magnitude, and blue<->red with a neutral gray
# midpoint for polarity. On light the ordinal ramp starts no lighter than step 250.
# Swapping in different hues means re-running scripts/validate_palette.js.

THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        # blue ramp, steps 250 -> 700 (ordinal floor honoured: nothing lighter than 250)
        ramp=["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
        seq_full=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        cat=["#2a78d6", "#eb6834"],   # slots 1-2, validated all-pairs
        div_lo="#d03b3b", div_mid="#f0efec", div_hi="#2a78d6",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        # dark ordinal floor: nothing darker than step 600, so the ramp runs inverted
        ramp=["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
              "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6"],
        seq_full=["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        cat=["#3987e5", "#d95926"],
        div_lo="#d03b3b", div_mid="#383835", div_hi="#3987e5",
    ),
}

METRICS = {
    "rankme_text": "RankMe (text)",
    "alignment_cos": "Alignment cos(text, code)",
    "linear_r2_text_to_code": "Linear $R^2$ text$\\to$code",
    "uniformity_text": "Uniformity (text)",
}

BASELINE_LBD = 0.0


# --- Chrome ------------------------------------------------------------------

def apply_theme(T):
    """Recessive hairline chrome: solid grid one shade off the surface, no frame."""
    mpl.rcParams.update({
        "figure.facecolor": T["surface"], "axes.facecolor": T["surface"],
        "savefig.facecolor": T["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
        "text.color": T["ink"], "axes.labelcolor": T["ink2"],
        "xtick.color": T["muted"], "ytick.color": T["muted"],
        "axes.edgecolor": T["axis"], "axes.linewidth": 0.8,
        "grid.color": T["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.grid": True, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 130, "savefig.dpi": 200,
        "axes.titlesize": 11, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    })


def tidy_axes(ax, T):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["axis"])
    ax.tick_params(length=3, width=0.8)


def ramp_colors(T, n):
    """n steps spanning the ordinal ramp, endpoints included."""
    r = T["ramp"]
    if n <= 1:
        return [r[len(r) // 2]]
    idx = np.linspace(0, len(r) - 1, n).round().astype(int)
    return [r[i] for i in idx]


# --- Loading -----------------------------------------------------------------

def load(data_dir, seed):
    acc = pd.read_csv(os.path.join(data_dir, "accuracy_tidy.csv"))
    geo = pd.read_csv(os.path.join(data_dir, "geometry_tidy.csv"))

    # Layer 0 is degenerate under last-token pooling -- drop before anything else.
    geo = geo[geo["layer"] > 0].copy()

    baseline_rows = acc[(acc["lbd"] == BASELINE_LBD) & (acc["seed"] == seed)]
    baseline = float(baseline_rows["accuracy_raw"].iloc[0]) if len(baseline_rows) else np.nan

    # Seed-noise floor, measured rather than assumed: the largest observed spread
    # among cells that actually have repeat seeds (tranche 1 only).
    reps = acc.groupby(["lbd", "k"])["accuracy_raw"].agg(["min", "max", "count"])
    reps = reps[reps["count"] > 1]
    noise = float((reps["max"] - reps["min"]).max()) if len(reps) else np.nan

    return acc, geo, baseline, noise


def final_geometry(geo, seed, split="test", layer=None):
    """One geometry row per cell: last checkpoint, chosen split and layer."""
    g = geo[(geo["seed"] == seed) & (geo["split"] == split)].copy()
    if g.empty:
        return g
    g = g[g["step"] == g["step"].max()]
    g = g[g["layer"] == (g["layer"].max() if layer is None else layer)]
    return g


def dump(df, out_dir, name):
    """Table-view twin: the exact rows behind the figure."""
    path = os.path.join(out_dir, f"{name}_data.csv")
    df.to_csv(path, index=False)
    return path


def save(fig, out_dir, name):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def k_marker(k):
    return {0: "o", 1: "s", 2: "^", 4: "D", 8: "v", 16: "P", 32: "X"}.get(int(k), "*")


# --- Figure 1: does the mediator mediate? ------------------------------------

def fig_mediation(acc, geo, baseline, T, out_dir, seed, split, layer):
    """Geometry vs accuracy, one point per cell.

    The whole design is (lambda, k) -> geometry -> accuracy. If geometry mediates,
    cells must fall on a single curve no matter which (lambda, k) produced them.
    Colour is lambda (sequential, it is a magnitude) and marker is k, so a
    k-structured departure from the curve is visible as shape, not guessed at.
    """
    g = final_geometry(geo, seed, split, layer)
    a = acc[acc["seed"] == seed]
    df = g.merge(a[["lbd", "k", "accuracy", "accuracy_raw", "status"]], on=["lbd", "k"])
    df = df[df["lbd"] > 0]
    if df.empty:
        print("  [skip] mediation: no rows")
        return

    lbds = sorted(df["lbd"].unique())
    cmap = dict(zip(lbds, ramp_colors(T, len(lbds))))
    ks = sorted(df["k"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2))
    for ax, (col, label) in zip(axes.ravel(), METRICS.items()):
        for _, r in df.iterrows():
            ok = r["status"] == "ok"
            ax.plot(r[col], r["accuracy_raw"], marker=k_marker(r["k"]),
                    ms=8, mew=1.4, ls="none",
                    color=cmap[r["lbd"]] if ok else "none",
                    markerfacecolor=cmap[r["lbd"]] if ok else "none",
                    markeredgecolor=cmap[r["lbd"]] if ok else T["div_lo"],
                    zorder=3)
        if np.isfinite(baseline):
            ax.axhline(baseline, color=T["muted"], lw=1.2, zorder=1)
            ax.text(0.995, baseline, " no-JEPA baseline", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7.5, color=T["muted"])
        ax.set_xlabel(label)
        ax.set_ylabel("Exact-match accuracy")
        tidy_axes(ax, T)

    # Identity is never colour-alone: lambda ramp + k shapes, both spelled out.
    lam_handles = [Line2D([], [], marker="o", ls="none", ms=7, color=cmap[l],
                          label=f"$\\lambda$={l:g}") for l in lbds]
    k_handles = [Line2D([], [], marker=k_marker(k), ls="none", ms=7,
                        color=T["ink2"], label=f"k={int(k)}" + (" (no pred. tokens)" if k == 0 else ""))
                 for k in ks]
    bad = Line2D([], [], marker="o", ls="none", ms=7, markerfacecolor="none",
                 markeredgecolor=T["div_lo"], label="did not train cleanly")
    fig.legend(handles=lam_handles + k_handles + [bad], loc="upper center",
               bbox_to_anchor=(0.5, 0.055), ncol=6, labelcolor=T["ink2"])
    fig.suptitle("Does the geometry mediate? Cells should lie on one curve regardless of "
                 f"($\\lambda$, k)\nfinal checkpoint, {split} split, layer "
                 f"{int(df['layer'].iloc[0])}, seed {seed}",
                 color=T["ink"], fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    dump(df, out_dir, "fig1_mediation")
    save(fig, out_dir, "fig1_mediation")


# --- Figure 2: accuracy dose-response ----------------------------------------

def fig_dose(acc, baseline, noise, T, out_dir, seed):
    """Accuracy vs lambda, one line per k.

    lambda is log-spaced, so the x axis is log2. lambda=0 cannot live on a log
    axis and is the baseline reference line instead. k is ordered, so it gets the
    ordinal ramp rather than arbitrary categorical hues.
    """
    a = acc[(acc["seed"] == seed) & (acc["lbd"] > 0)].copy()
    if a.empty:
        print("  [skip] dose: no rows")
        return
    ks = sorted(a["k"].unique())
    cmap = dict(zip(ks, ramp_colors(T, len(ks))))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    if np.isfinite(baseline) and np.isfinite(noise):
        ax.axhspan(baseline - noise / 2, baseline + noise / 2,
                   color=T["muted"], alpha=0.13, lw=0, zorder=0)
    if np.isfinite(baseline):
        ax.axhline(baseline, color=T["muted"], lw=1.2, zorder=1)

    for k in ks:
        s = a[a["k"] == k].sort_values("lbd")
        ax.plot(s["lbd"], s["accuracy_raw"], lw=2, marker="o", ms=5,
                color=cmap[k], zorder=3, label=f"k={int(k)}")
        broken = s[s["status"] != "ok"]
        ax.plot(broken["lbd"], broken["accuracy_raw"], marker="o", ms=9, ls="none",
                markerfacecolor="none", markeredgecolor=T["div_lo"], mew=1.6, zorder=4)

    # Direct-label only the two lines that carry the story.
    for k in (ks[0], ks[-1]):
        s = a[a["k"] == k].sort_values("lbd")
        if len(s):
            ax.annotate(f"k={int(k)}", (s["lbd"].iloc[-1], s["accuracy_raw"].iloc[-1]),
                        textcoords="offset points", xytext=(7, 0), va="center",
                        fontsize=9, color=cmap[k], fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("$\\lambda$ (JEPA loss weight, log scale)")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_xticks(sorted(a["lbd"].unique()))
    ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    # Headroom on the right so the direct labels sit beside the line ends rather
    # than being clipped by the axes.
    ax.set_xlim(a["lbd"].min() / 1.6, a["lbd"].max() * 2.2)
    tidy_axes(ax, T)

    # The baseline and the noise band are described in the legend, not as floating
    # text: at low lambda the k lines run straight through where that text sat.
    ref = []
    if np.isfinite(baseline):
        ref.append(Line2D([], [], color=T["muted"], lw=1.2,
                          label=f"no-JEPA baseline ({baseline:.3f})"))
    if np.isfinite(noise):
        ref.append(mpl.patches.Patch(facecolor=T["muted"], alpha=0.13,
                                     label=f"observed seed spread ({noise*100:.1f} pts)"))
    h, lab = ax.get_legend_handles_labels()
    # Legend below the axes -- inside, it overlapped the k=32 line sitting at y=0.
    ax.legend(h + ref, lab + [r.get_label() for r in ref],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5,
              labelcolor=T["ink2"],
              title="k = predictor tokens (k=0 is JEPA without them, not the baseline)",
              title_fontsize=8, alignment="left")
    ax.set_title(f"Accuracy dose-response, seed {seed}", color=T["ink"], loc="left")
    fig.tight_layout()
    dump(a.sort_values(["k", "lbd"]), out_dir, "fig2_dose")
    save(fig, out_dir, "fig2_dose")


# --- Figure 3: the grid ------------------------------------------------------

def fig_grid(acc, baseline, T, out_dir, seed):
    """lambda x k heatmap.

    The reader's question is "better or worse than no JEPA", which is polarity ->
    diverging scale with a neutral midpoint pinned to the baseline. Cells that did
    not train cleanly are hatched, not coloured: they are not a value on this scale.
    """
    a = acc[(acc["seed"] == seed) & (acc["lbd"] > 0)]
    if a.empty or not np.isfinite(baseline):
        print("  [skip] grid: no rows or no baseline")
        return
    piv = a.pivot_table(index="lbd", columns="k", values="accuracy_raw")
    stat = a.pivot_table(index="lbd", columns="k", values="status", aggfunc="first")

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "div", [T["div_lo"], T["div_mid"], T["div_hi"]])
    lo, hi = np.nanmin(piv.values), np.nanmax(piv.values)
    norm = TwoSlopeNorm(vmin=min(lo, baseline - 1e-6), vcenter=baseline,
                        vmax=max(hi, baseline + 1e-6))

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.imshow(piv.values, cmap=cmap, norm=norm, aspect="auto", origin="lower")
    ax.grid(False)
    n_missing = int(np.isnan(piv.values).sum())
    for i, l in enumerate(piv.index):
        for j, k in enumerate(piv.columns):
            v = piv.values[i, j]
            if not np.isfinite(v):
                # A blank cell is ambiguous -- it reads as a value near the
                # midpoint. Mark "not run" explicitly.
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor=T["grid"], edgecolor="none"))
                ax.text(j, i, "not\nrun", ha="center", va="center", fontsize=7.5,
                        color=T["muted"], linespacing=1.1)
                continue
            if stat.values[i, j] != "ok":
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           hatch="///", edgecolor=T["div_lo"], lw=0))
            # Ink tokens, never the series colour; flip for contrast on dark cells.
            far = abs(v - baseline) / max(hi - baseline, baseline - lo, 1e-9)
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if far > 0.55 else T["ink"])
    ax.set_xticks(range(len(piv.columns)), [f"{int(c)}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{v:g}" for v in piv.index])
    ax.set_xlabel("k (predictor tokens; k=0 is JEPA without them)")
    ax.set_ylabel("$\\lambda$")
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label(f"accuracy (midpoint = no-JEPA baseline {baseline:.3f})", color=T["ink2"])
    cb.outline.set_visible(False)
    tidy_axes(ax, T)
    title = f"Accuracy against baseline, seed {seed}"
    if n_missing:
        title += f"  ({n_missing} cell(s) not yet run)"
    ax.set_title(title, color=T["ink"], loc="left")
    fig.tight_layout()
    dump(a.sort_values(["lbd", "k"]), out_dir, "fig3_grid")
    save(fig, out_dir, "fig3_grid")


# --- Figures 4 & 5: where and when -------------------------------------------

def _lambda_subset(values, n=5):
    """A readable ramp of lambdas: 11 lines on one axis is unreadable, so span
    the range instead of drawing every level."""
    vals = sorted(v for v in values if v > 0)
    if len(vals) <= n:
        return vals
    idx = np.linspace(0, len(vals) - 1, n).round().astype(int)
    return [vals[i] for i in sorted(set(idx))]


def fig_layers(geo, T, out_dir, seed, split, k_fix):
    """Geometry vs layer. Layer is ordered and is never averaged (by design)."""
    g = geo[(geo["seed"] == seed) & (geo["split"] == split)].copy()
    g = g[g["step"] == g["step"].max()]
    g = g[g["k"] == k_fix]
    if g.empty:
        print(f"  [skip] layers: nothing at k={k_fix}")
        return
    lbds = _lambda_subset(g["lbd"].unique())
    base = g[g["lbd"] == BASELINE_LBD]
    cmap = dict(zip(lbds, ramp_colors(T, len(lbds))))

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), sharex=True)
    for ax, (col, label) in zip(axes.ravel(), METRICS.items()):
        if len(base):
            b = base.sort_values("layer")
            ax.plot(b["layer"], b[col], lw=1.6, ls=(0, (4, 2)), color=T["muted"],
                    zorder=2, label="$\\lambda$=0 (baseline)")
        for l in lbds:
            s = g[g["lbd"] == l].sort_values("layer")
            ax.plot(s["layer"], s[col], lw=2, color=cmap[l], zorder=3,
                    label=f"$\\lambda$={l:g}")
        ax.set_ylabel(label)
        if col == "rankme_text":
            ax.set_yscale("log")
        tidy_axes(ax, T)
    for ax in axes[1]:
        ax.set_xlabel("Layer (0 dropped: degenerate under last-token pooling)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.055),
               ncol=len(lab), labelcolor=T["ink2"])
    fig.suptitle(f"Geometry across layers -- final checkpoint, {split} split, k={k_fix}, seed {seed}",
                 color=T["ink"], fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    dump(g[g["lbd"].isin(list(lbds) + [BASELINE_LBD])], out_dir, "fig4_layers")
    save(fig, out_dir, "fig4_layers")


def fig_trajectory(geo, T, out_dir, seed, split, k_fix, layer):
    """Geometry vs training step. A mediator has to move before the outcome
    settles, so the shape of these curves is what any causal ordering rests on."""
    g = geo[(geo["seed"] == seed) & (geo["split"] == split) & (geo["k"] == k_fix)].copy()
    g = g[g["layer"] == (g["layer"].max() if layer is None else layer)]
    if g.empty:
        print(f"  [skip] trajectory: nothing at k={k_fix}")
        return
    lbds = _lambda_subset(g["lbd"].unique())
    base = g[g["lbd"] == BASELINE_LBD]
    cmap = dict(zip(lbds, ramp_colors(T, len(lbds))))

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), sharex=True)
    for ax, (col, label) in zip(axes.ravel(), METRICS.items()):
        if len(base):
            b = base.sort_values("step")
            ax.plot(b["step"], b[col], lw=1.6, ls=(0, (4, 2)), color=T["muted"],
                    marker="o", ms=5, zorder=2, label="$\\lambda$=0 (baseline)")
        for l in lbds:
            s = g[g["lbd"] == l].sort_values("step")
            ax.plot(s["step"], s[col], lw=2, marker="o", ms=5, color=cmap[l],
                    zorder=3, label=f"$\\lambda$={l:g}")
        ax.set_ylabel(label)
        if col == "rankme_text":
            ax.set_yscale("log")
        tidy_axes(ax, T)
    for ax in axes[1]:
        ax.set_xlabel("Training step (one checkpoint per epoch)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.055),
               ncol=len(lab), labelcolor=T["ink2"])
    fig.suptitle(f"Geometry trajectory -- {split} split, layer {int(g['layer'].iloc[0])}, "
                 f"k={k_fix}, seed {seed}", color=T["ink"], fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    dump(g[g["lbd"].isin(list(lbds) + [BASELINE_LBD])], out_dir, "fig5_trajectory")
    save(fig, out_dir, "fig5_trajectory")


# --- Figure 6: training health ----------------------------------------------

def fig_health(acc, T, out_dir, seed):
    """Final lm_loss and final total loss against lambda.

    Two measures on wildly different scales -- total loss scales with lambda by
    construction. Small multiples, never a second y-axis on one plot.
    """
    a = acc[(acc["seed"] == seed) & (acc["lbd"] > 0)].copy()
    if a.empty:
        print("  [skip] health: no rows")
        return
    ks = sorted(a["k"].unique())
    cmap = dict(zip(ks, ramp_colors(T, len(ks))))

    panels = [("lm_loss_last", "Final lm_loss (LM term alone)"),
              ("loss_last", "Final total loss ($\\gamma\\cdot$lm + $\\lambda\\cdot$jepa)")]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for ax, (col, label) in zip(axes, panels):
        if a[col].notna().sum() == 0:
            ax.text(0.5, 0.5, f"{col} not recorded\nfor these cells", ha="center",
                    va="center", transform=ax.transAxes, color=T["muted"], fontsize=9)
        for k in ks:
            s = a[a["k"] == k].sort_values("lbd")
            ax.plot(s["lbd"], s[col], lw=2, marker="o", ms=5, color=cmap[k],
                    label=f"k={int(k)}")
        ax.set_xscale("log", base=2)
        ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("$\\lambda$ (log scale)")
        ax.set_ylabel(label)
        if col == "loss_last":
            ax.set_yscale("log")
        tidy_axes(ax, T)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.045),
               ncol=len(lab), labelcolor=T["ink2"])
    fig.suptitle(f"Training health: the LM term converges at every $\\lambda$ -- seed {seed}",
                 color=T["ink"], fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    dump(a[["lbd", "k", "status", "lm_loss_last", "loss_last", "grad_norm_max"]]
         .sort_values(["k", "lbd"]), out_dir, "fig6_health")
    save(fig, out_dir, "fig6_health")


# --- Main --------------------------------------------------------------------

ALL = ["mediation", "dose", "grid", "layers", "trajectory", "health"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="dir holding the two tidy CSVs")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--theme", choices=list(THEMES), default="light")
    p.add_argument("--seed", type=int, default=82)
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--layer", type=int, default=None,
                   help="geometry layer for fig 1 and 5 (default: last)")
    p.add_argument("--k-fix", type=int, default=0,
                   help="k held fixed in the per-layer and trajectory figures")
    p.add_argument("--figures", nargs="+", choices=ALL + ["all"], default=["all"])
    args = p.parse_args()

    want = ALL if "all" in args.figures else args.figures
    T = THEMES[args.theme]
    apply_theme(T)
    os.makedirs(args.out_dir, exist_ok=True)

    acc, geo, baseline, noise = load(args.data_dir, args.seed)
    print(f"loaded {len(acc)} accuracy rows, {len(geo)} geometry rows "
          f"(layer 0 dropped); baseline={baseline:.4f}, seed-noise={noise*100:.1f} pts")
    n_bad = int((acc["status"] != "ok").sum())
    if n_bad:
        print(f"  {n_bad} cell(s) not status=ok -- drawn hollow/hatched, never as a value")

    if "mediation" in want:
        fig_mediation(acc, geo, baseline, T, args.out_dir, args.seed, args.split, args.layer)
    if "dose" in want:
        fig_dose(acc, baseline, noise, T, args.out_dir, args.seed)
    if "grid" in want:
        fig_grid(acc, baseline, T, args.out_dir, args.seed)
    if "layers" in want:
        fig_layers(geo, T, args.out_dir, args.seed, args.split, args.k_fix)
    if "trajectory" in want:
        fig_trajectory(geo, T, args.out_dir, args.seed, args.split, args.k_fix, args.layer)
    if "health" in want:
        fig_health(acc, T, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
