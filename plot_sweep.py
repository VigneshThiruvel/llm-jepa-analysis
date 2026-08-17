"""Figures for the observational lambda x k sweep.

Reads the tidy CSVs written by collect_sweep.py and renders the figure set for
the mediation analysis. Pure pandas + matplotlib, no seaborn, no network —
designed to run on a laptop against CSVs copied down from JURECA:

    rsync -av <host>:<repo>/sweep_runs/*/{accuracy,geometry}_tidy_*.csv ./data/
    python plot_sweep.py --data-dir ./data --out-dir ./figures

SEEDS
-----
By default **every seed in the table is used**, and each figure shows the mean
across seeds with the observed min-max spread drawn per point (whiskers on the
dose-response, a band on the profile figures, +/- in the heatmap cells). Real
per-cell spread replaces the single global "noise band" the earlier single-seed
version had to fall back on. `--seeds 82` (or `82,23`) restricts the set; with
one seed the spread collapses and the figures degrade gracefully to points.

Aggregation rule: the **`accuracy`** column is averaged, never `accuracy_raw`.
collect_sweep.py blanks `accuracy` for cells that did not train cleanly, so those
cells drop out of the mean instead of dragging it down, and every figure reports
where that happened (`n_ok < n_seeds`) rather than hiding it. `accuracy_raw` is
used only to *position* the hollow marker that flags such a cell, never as a value.

DATASETS AND MODELS
-------------------
Every run dir used to write the same two basenames, so copying synth's and turk's
tables into one folder clobbered one of them. collect_sweep.py now writes
`accuracy_tidy_<dataset>_<model>.csv` (+ matching geometry file) and stamps
`dataset` / `model` onto every row, so a table is identifiable by name *and* by
content. This script discovers every such pair in --data-dir, renders the full
set once per (dataset, model) into `<out-dir>/<dataset>_<model>/`, and names the
dataset and model in every figure title. --dataset / --model render a subset.

FIGURES
  1 mediation    geometry vs accuracy, one panel per k   -- does the mediator mediate?
  2 dose         accuracy vs lambda, one line per k      -- the k=0 flatness
  3 grid         lambda x k heatmap vs baseline          -- same story, paper-ready
  4 layers       geometry vs layer                       -- where in the network
  5 trajectory   geometry vs training step               -- when (causal ordering)
  6 health       lm_loss / total loss vs lambda          -- "not a training failure"

Each figure also writes a `<name>_data.csv` with exactly the rows plotted, so no
value is reachable only by reading pixels.

OTHER CONVENTIONS ENFORCED HERE (each one is a way to get the story wrong)
  * Layer 0 is dropped. Under last-token pooling every example ends on the same
    template token, so rankme ~ 1 and linear_r2 is NaN. It is not a data point.
  * lambda=0 is never plotted on the log axis. It is the no-JEPA baseline and is
    drawn as a reference line / reference marker.
  * k=0 at lambda>0 is NOT the baseline -- it is JEPA with no predictor tokens.
"""

import argparse
import glob
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# --- Palette -----------------------------------------------------------------
# The data-viz reference palette, used unchanged and only in combinations it
# documents as validated: the single-hue blue ramp for magnitude, and blue<->red
# with a neutral gray midpoint for polarity. On light the ordinal ramp starts no
# lighter than step 250. Different hues means re-running validate_palette.js.

THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        ramp=["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
        div_lo="#d03b3b", div_mid="#f0efec", div_hi="#2a78d6",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        ramp=["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
              "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6"],
        div_lo="#d03b3b", div_mid="#383835", div_hi="#3987e5",
    ),
}

METRICS = {
    "rankme_text": "RankMe (text)",
    "alignment_cos": "Alignment cos(text, code)",
    "linear_r2_text_to_code": "Linear $R^2$ text$\\to$code",
    "uniformity_text": "Uniformity (text)",
}
LOG_METRICS = {"rankme_text"}

BASELINE_LBD = 0.0


# --- Chrome ------------------------------------------------------------------

def apply_theme(T):
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
        "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    })


def tidy_axes(ax, T):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["axis"])
    ax.tick_params(length=3, width=0.8)


def ramp_colors(T, n):
    r = T["ramp"]
    if n <= 1:
        return [r[len(r) // 2]]
    idx = np.linspace(0, len(r) - 1, n).round().astype(int)
    return [r[i] for i in idx]


def facet_grid(n, panel=(3.1, 2.7)):
    """Square-ish small-multiple grid with one spare slot for the legend."""
    total = n + 1
    ncols = min(4, total)
    nrows = int(np.ceil(total / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                             figsize=(panel[0] * ncols, panel[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes, ncols


def seed_note(seeds):
    if len(seeds) == 1:
        return f"seed {seeds[0]}"
    return f"mean of {len(seeds)} seeds ({', '.join(str(s) for s in seeds)})"


# --- Discovery ---------------------------------------------------------------

def discover(data_dir):
    candidates = sorted(glob.glob(os.path.join(data_dir, "accuracy_tidy_*.csv")))
    if not candidates:
        # Only fall back to the pre-2026-08-16 unsuffixed pair when there is no
        # suffixed table. Reading both renders the same runs twice, once with
        # real identity and once as "unknown", from a file that is stale.
        candidates = sorted(glob.glob(os.path.join(data_dir, "accuracy_tidy.csv")))
    elif os.path.exists(os.path.join(data_dir, "accuracy_tidy.csv")):
        print(f"[note] ignoring legacy {data_dir}/accuracy_tidy.csv "
              f"(superseded by accuracy_tidy_<dataset>_<model>.csv)")

    found = []
    for acc_path in candidates:
        slug = re.sub(r"^accuracy_tidy_?", "", os.path.basename(acc_path)[:-4])
        geo_path = os.path.join(
            data_dir, f"geometry_tidy_{slug}.csv" if slug else "geometry_tidy.csv")
        if not os.path.exists(geo_path):
            print(f"[skip] {os.path.basename(acc_path)}: no matching geometry CSV")
            continue
        found.append((acc_path, geo_path, slug))
    return found


def identity(acc, slug):
    def one(col, default):
        if col in acc.columns and acc[col].notna().any():
            vals = sorted(acc[col].dropna().unique())
            return vals[0] if len(vals) == 1 else "+".join(map(str, vals))
        return default
    dataset = one("dataset", slug.split("_")[0] if slug else "unknown")
    return dataset, one("model", "unknown").split("/")[-1]


# --- Loading and seed aggregation --------------------------------------------

def load(acc_path, geo_path, seeds_arg):
    acc = pd.read_csv(acc_path)
    geo = pd.read_csv(geo_path)
    geo = geo[geo["layer"] > 0].copy()   # layer 0 is degenerate

    available = sorted(int(s) for s in acc["seed"].unique())
    if seeds_arg and seeds_arg != "all":
        want = [int(s) for s in re.split(r"[,\s]+", seeds_arg) if s]
        missing = [s for s in want if s not in available]
        if missing:
            print(f"  [warn] seeds not in table, ignored: {missing}")
        seeds = [s for s in want if s in available]
    else:
        seeds = available
    if not seeds:
        raise SystemExit(f"no usable seeds (table has {available})")

    return acc[acc["seed"].isin(seeds)].copy(), geo[geo["seed"].isin(seeds)].copy(), seeds


def agg_acc(acc):
    """Per (lbd, k): mean accuracy over seeds, with spread and clean-run counts.

    `accuracy` is NaN for cells that did not train cleanly, and mean()/min()/max()
    skip NaN — so a broken seed drops out of the mean rather than dragging it
    down. n_ok vs n_seeds records that it happened; `raw_mean` keeps the
    contaminated average available for anyone who explicitly wants it.
    """
    g = acc.groupby(["lbd", "k"])
    out = pd.DataFrame({
        "accuracy": g["accuracy"].mean(),
        "acc_min": g["accuracy"].min(),
        "acc_max": g["accuracy"].max(),
        "n_ok": g["accuracy"].count(),
        "n_seeds": g.size(),
        "raw_mean": g["accuracy_raw"].mean(),
    }).reset_index()
    for col in ("lm_loss_last", "loss_last", "grad_norm_max"):
        if col in acc.columns:
            out[col] = g[col].mean().values
    return out


def agg_geo(geo):
    """Per (lbd, k, step, split, layer): mean of each metric over seeds."""
    keys = ["lbd", "k", "step", "split", "layer"]
    present = [c for c in METRICS if c in geo.columns]
    g = geo.groupby(keys)
    out = g[present].mean()
    # Deliberately not "n_seeds": this frame gets merged with agg_acc(), which
    # has its own n_seeds, and the collision silently becomes n_seeds_x/_y.
    out["n_seeds_geo"] = g.size()
    return out.reset_index()


def at_final(geo, split, layer=None):
    """Last checkpoint, chosen split and layer."""
    g = geo[geo["split"] == split].copy()
    if g.empty:
        return g
    g = g[g["step"] == g["step"].max()]
    return g[g["layer"] == (g["layer"].max() if layer is None else layer)]


def dump(df, out_dir, name):
    df.to_csv(os.path.join(out_dir, f"{name}_data.csv"), index=False)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    {path}")


# --- Figure 1: does the mediator mediate? ------------------------------------

def fig_mediation(A, G, acc, geo, baseline, T, out_dir, seeds, split, layer, tag):
    """Geometry vs accuracy, ONE PANEL PER k, one figure per metric.

    The design is (lambda, k) -> geometry -> accuracy. If geometry mediates, the
    same geometry implies the same accuracy however it was reached — i.e. every
    panel traces the same curve. Faceting by k is what makes that readable; the
    original single-panel version put every cell, 11 colours and 7 marker shapes
    on one axis.

    Large markers are the seed mean, joined in lambda order so each panel is a
    *path* through (geometry, accuracy) as the JEPA weight rises. Small faint dots
    are the individual seeds, so the reader can see the spread behind each mean.
    Axes are shared across panels — panels are only comparable on common limits.
    """
    gm = at_final(G, split, layer)
    df = gm.merge(A, on=["lbd", "k"])
    if df.empty:
        print("    [skip] mediation: no rows")
        return
    per_seed = at_final(geo, split, layer).merge(
        acc[["lbd", "k", "seed", "accuracy", "accuracy_raw", "status"]],
        on=["lbd", "k", "seed"])

    base = df[df["lbd"] == BASELINE_LBD]
    jepa = df[df["lbd"] > 0]
    if jepa.empty:
        print("    [skip] mediation: no lambda>0 rows")
        return
    lbds = sorted(jepa["lbd"].unique())
    cmap = dict(zip(lbds, ramp_colors(T, len(lbds))))
    ks = sorted(jepa["k"].unique())
    layer_no = int(df["layer"].iloc[0])

    for col, mlabel in METRICS.items():
        if col not in jepa.columns or jepa[col].notna().sum() == 0:
            continue
        fig, axes, ncols = facet_grid(len(ks))
        for ax, k in zip(axes, ks):
            s = jepa[jepa["k"] == k].sort_values("lbd")
            ps = per_seed[(per_seed["k"] == k) & (per_seed["lbd"] > 0)]
            # Individual seeds behind the mean.
            for _, r in ps.iterrows():
                if r["lbd"] in cmap:
                    ax.plot(r[col], r["accuracy_raw"], marker="o", ms=3.5, ls="none",
                            color=cmap[r["lbd"]], alpha=0.35, zorder=3)
            ax.plot(s[col], s["accuracy"], "-", lw=1.2, color=T["muted"],
                    alpha=0.55, zorder=2)
            for _, r in s.iterrows():
                clean = r["n_ok"] == r["n_seeds"]
                y = r["accuracy"] if np.isfinite(r["accuracy"]) else r["raw_mean"]
                ax.plot(r[col], y, marker="o", ms=8, mew=1.5, ls="none",
                        markerfacecolor=cmap[r["lbd"]] if clean else "none",
                        markeredgecolor=cmap[r["lbd"]] if clean else T["div_lo"],
                        zorder=4)
            if np.isfinite(baseline):
                ax.axhline(baseline, color=T["muted"], lw=1, zorder=1)
            if len(base):
                ax.plot(base[col].iloc[0], base["accuracy"].iloc[0], marker="*",
                        ms=13, ls="none", color=T["muted"], zorder=3)
            # annotation_clip=False: at the panel edge these were cut mid-number
            # ("0.12" for 0.125), which reads as a different lambda.
            for r, off, ha in ((s.iloc[0], (0, 9), "center"), (s.iloc[-1], (7, 4), "left")):
                if np.isfinite(r[col]):
                    ax.annotate(f"$\\lambda$={r['lbd']:g}", (r[col], r["accuracy"]),
                                textcoords="offset points", xytext=off, ha=ha,
                                fontsize=7.5, color=cmap[r["lbd"]],
                                annotation_clip=False, zorder=5)
            ax.set_title(f"k = {int(k)}" + ("  (no predictor tokens)" if k == 0 else ""),
                         color=T["ink"], loc="left")
            if col in LOG_METRICS:
                ax.set_xscale("log")
            tidy_axes(ax, T)

        # sharex hides tick labels on all but the last row, which strips them from
        # a panel whose column ends early (the legend slot sits under it).
        for i, ax in enumerate(axes[:len(ks)]):
            if i + ncols >= len(ks):
                ax.set_xlabel(mlabel)
                ax.tick_params(labelbottom=True)
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel("Exact-match accuracy")

        handles = [Line2D([], [], marker="o", ls="none", ms=7, color=cmap[l],
                          label=f"$\\lambda$={l:g}") for l in lbds]
        handles += [
            Line2D([], [], marker="*", ls="none", ms=11, color=T["muted"],
                   label="$\\lambda$=0 baseline"),
            Line2D([], [], marker="o", ls="none", ms=4, color=T["muted"], alpha=0.5,
                   label="individual seeds"),
            Line2D([], [], marker="o", ls="none", ms=7, markerfacecolor="none",
                   markeredgecolor=T["div_lo"], label="a seed not status=ok"),
        ]
        spare = axes[len(ks)] if len(axes) > len(ks) else None
        if spare is not None:
            spare.legend(handles=handles, loc="center", ncol=2, labelcolor=T["ink2"])
        else:
            fig.legend(handles=handles, loc="upper center",
                       bbox_to_anchor=(0.5, 0.03), ncol=7, labelcolor=T["ink2"])
        fig.suptitle(
            f"Does geometry mediate? Same curve in every panel = yes  —  {tag}\n"
            f"{mlabel} vs accuracy, final checkpoint, {split} split, layer {layer_no}, "
            f"{seed_note(seeds)}; joined in $\\lambda$ order",
            color=T["ink"], fontsize=11.5)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        dump(df, out_dir, f"fig1_mediation_{col}")
        save(fig, out_dir, f"fig1_mediation_{col}")


# --- Figure 2: accuracy dose-response ----------------------------------------

def fig_dose(A, baseline, T, out_dir, seeds, tag):
    """Accuracy vs lambda, one line per k, mean over seeds with min-max whiskers.

    k is ordered, so it gets the ordinal ramp rather than arbitrary categorical
    hues. The whiskers are the real observed spread per cell; with >1 seed they
    replace the single global noise band the single-seed version had to use.
    """
    a = A[A["lbd"] > 0].copy()
    if a.empty:
        print("    [skip] dose: no rows")
        return
    ks = sorted(a["k"].unique())
    cmap = dict(zip(ks, ramp_colors(T, len(ks))))

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    if np.isfinite(baseline):
        ax.axhline(baseline, color=T["muted"], lw=1.2, zorder=1)

    for k in ks:
        s = a[a["k"] == k].sort_values("lbd")
        lo = (s["accuracy"] - s["acc_min"]).clip(lower=0)
        hi = (s["acc_max"] - s["accuracy"]).clip(lower=0)
        ax.errorbar(s["lbd"], s["accuracy"], yerr=[lo, hi], fmt="-o", lw=2, ms=5,
                    color=cmap[k], ecolor=cmap[k], elinewidth=1.1, capsize=2.5,
                    capthick=1.1, zorder=3, label=f"k={int(k)}")
        bad = s[s["n_ok"] < s["n_seeds"]]
        ax.plot(bad["lbd"], bad["accuracy"].fillna(bad["raw_mean"]), marker="o",
                ms=10, ls="none", markerfacecolor="none",
                markeredgecolor=T["div_lo"], mew=1.6, zorder=4)
    for k in (ks[0], ks[-1]):
        s = a[a["k"] == k].sort_values("lbd")
        if len(s):
            ax.annotate(f"k={int(k)}", (s["lbd"].iloc[-1], s["accuracy"].iloc[-1]),
                        textcoords="offset points", xytext=(8, 0), va="center",
                        fontsize=9, color=cmap[k], fontweight="bold",
                        annotation_clip=False)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("$\\lambda$ (JEPA loss weight, log scale)")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_xticks(sorted(a["lbd"].unique()))
    ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(a["lbd"].min() / 1.6, a["lbd"].max() * 2.4)
    tidy_axes(ax, T)

    ref = []
    if np.isfinite(baseline):
        ref.append(Line2D([], [], color=T["muted"], lw=1.2,
                          label=f"no-JEPA baseline ({baseline:.3f})"))
    if len(seeds) > 1:
        ref.append(Line2D([], [], color=T["muted"], lw=1.1,
                          label=f"whiskers = min-max over {len(seeds)} seeds"))
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h + ref, lab + [r.get_label() for r in ref],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5,
              labelcolor=T["ink2"],
              title="k = predictor tokens (k=0 is JEPA without them, not the baseline)",
              title_fontsize=8, alignment="left")
    ax.set_title(f"Accuracy dose-response, {seed_note(seeds)}  —  {tag}",
                 color=T["ink"], loc="left")
    fig.tight_layout()
    dump(a.sort_values(["k", "lbd"]), out_dir, "fig2_dose")
    save(fig, out_dir, "fig2_dose")


# --- Figure 3: the grid ------------------------------------------------------

def fig_grid(A, baseline, T, out_dir, seeds, tag):
    """lambda x k heatmap of the seed mean. "Better or worse than no JEPA" is
    polarity, so the scale diverges around the baseline. Each cell also carries
    its min-max spread, so a mean is never read without its uncertainty."""
    a = A[A["lbd"] > 0]
    if a.empty or not np.isfinite(baseline):
        print("    [skip] grid: no rows or no baseline")
        return
    piv = a.pivot_table(index="lbd", columns="k", values="accuracy")
    spread = a.pivot_table(index="lbd", columns="k", values="acc_max") - \
        a.pivot_table(index="lbd", columns="k", values="acc_min")
    nok = a.pivot_table(index="lbd", columns="k", values="n_ok")
    nsd = a.pivot_table(index="lbd", columns="k", values="n_seeds")

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "div", [T["div_lo"], T["div_mid"], T["div_hi"]])
    lo, hi = np.nanmin(piv.values), np.nanmax(piv.values)
    norm = TwoSlopeNorm(vmin=min(lo, baseline - 1e-6), vcenter=baseline,
                        vmax=max(hi, baseline + 1e-6))

    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    ax.imshow(piv.values, cmap=cmap, norm=norm, aspect="auto", origin="lower")
    ax.grid(False)
    n_missing = 0
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            if not np.isfinite(v):
                # Blank reads as a value near the midpoint; say which it is.
                present = np.isfinite(nsd.values[i, j]) and nsd.values[i, j] > 0
                n_missing += 0 if present else 1
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor=T["grid"], edgecolor="none"))
                ax.text(j, i, "no ok\nseed" if present else "not\nrun",
                        ha="center", va="center", fontsize=7,
                        color=T["div_lo"] if present else T["muted"], linespacing=1.1)
                continue
            if nok.values[i, j] < nsd.values[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           hatch="///", edgecolor=T["div_lo"], lw=0))
            far = abs(v - baseline) / max(hi - baseline, baseline - lo, 1e-9)
            c = "#ffffff" if far > 0.55 else T["ink"]
            sp = spread.values[i, j]
            if len(seeds) > 1 and np.isfinite(sp):
                ax.text(j, i + 0.16, f"{v:.3f}", ha="center", va="center",
                        fontsize=8.5, color=c)
                ax.text(j, i - 0.20, f"$\\pm${sp/2:.3f}", ha="center", va="center",
                        fontsize=6.5, color=c, alpha=0.85)
            else:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5, color=c)
    ax.set_xticks(range(len(piv.columns)), [f"{int(c)}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{v:g}" for v in piv.index])
    ax.set_xlabel("k (predictor tokens; k=0 is JEPA without them)")
    ax.set_ylabel("$\\lambda$")
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label(f"mean accuracy (midpoint = no-JEPA baseline {baseline:.3f})",
                 color=T["ink2"])
    cb.outline.set_visible(False)
    tidy_axes(ax, T)
    title = f"Accuracy against baseline, {seed_note(seeds)}  —  {tag}"
    if len(seeds) > 1:
        title += "\n$\\pm$ is half the min-max range across seeds"
    if n_missing:
        title += f"  ({n_missing} not yet run)"
    ax.set_title(title, color=T["ink"], loc="left")
    fig.tight_layout()
    dump(a.sort_values(["lbd", "k"]), out_dir, "fig3_grid")
    save(fig, out_dir, "fig3_grid")


# --- Figures 4 & 5: where and when -------------------------------------------

def _lambda_subset(values, n=5):
    """A readable ramp of lambdas: 11 lines on one axis is unreadable."""
    vals = sorted(v for v in values if v > 0)
    if len(vals) <= n:
        return vals
    idx = np.linspace(0, len(vals) - 1, n).round().astype(int)
    return [vals[i] for i in sorted(set(idx))]


def _profile(G, geo, T, out_dir, seeds, split, k_fix, xcol, xlabel, name, subtitle, tag):
    """Shared body of the per-layer and per-step geometry figures.

    Lines are the seed mean; the band is the min-max across seeds, so spread is
    visible without inventing a standard error from 3 points.
    """
    g = G[(G["split"] == split) & (G["k"] == k_fix)].copy()
    raw = geo[(geo["split"] == split) & (geo["k"] == k_fix)]
    if g.empty:
        print(f"    [skip] {name}: nothing at k={k_fix}")
        return
    lbds = _lambda_subset(g["lbd"].unique())
    cmap = dict(zip(lbds, ramp_colors(T, len(lbds))))

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), sharex=True)
    for ax, (col, label) in zip(axes.ravel(), METRICS.items()):
        if col not in g.columns:
            continue
        b = g[g["lbd"] == BASELINE_LBD].sort_values(xcol)
        if len(b):
            ax.plot(b[xcol], b[col], lw=1.6, ls=(0, (4, 2)), color=T["muted"],
                    marker="o", ms=4, zorder=2, label="$\\lambda$=0 (baseline)")
        for l in lbds:
            s = g[g["lbd"] == l].sort_values(xcol)
            if len(seeds) > 1:
                r = raw[raw["lbd"] == l].groupby(xcol)[col].agg(["min", "max"])
                if len(r):
                    ax.fill_between(r.index, r["min"], r["max"], color=cmap[l],
                                    alpha=0.16, lw=0, zorder=2)
            ax.plot(s[xcol], s[col], lw=2, marker="o", ms=4, color=cmap[l],
                    zorder=3, label=f"$\\lambda$={l:g}")
        ax.set_ylabel(label)
        if col in LOG_METRICS:
            ax.set_yscale("log")
        tidy_axes(ax, T)
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.055),
               ncol=len(lab), labelcolor=T["ink2"])
    band = "; band = min-max over seeds" if len(seeds) > 1 else ""
    fig.suptitle(f"{subtitle}{band}  —  {tag}", color=T["ink"], fontsize=11.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    dump(g[g["lbd"].isin(list(lbds) + [BASELINE_LBD])], out_dir, name)
    save(fig, out_dir, name)


def fig_layers(G, geo, T, out_dir, seeds, split, k_fix, tag):
    g = G[G["step"] == G["step"].max()] if len(G) else G
    raw = geo[geo["step"] == geo["step"].max()] if len(geo) else geo
    _profile(g, raw, T, out_dir, seeds, split, k_fix, "layer",
             "Layer (0 dropped: degenerate under last-token pooling)", "fig4_layers",
             f"Geometry across layers — final checkpoint, {split} split, k={k_fix}, "
             f"{seed_note(seeds)}", tag)


def fig_trajectory(G, geo, T, out_dir, seeds, split, k_fix, layer, tag):
    g = G[G["layer"] == (G["layer"].max() if layer is None else layer)] if len(G) else G
    raw = geo[geo["layer"] == (geo["layer"].max() if layer is None else layer)]
    lay = int(g["layer"].iloc[0]) if len(g) else "?"
    _profile(g, raw, T, out_dir, seeds, split, k_fix, "step",
             "Training step (one checkpoint per epoch)", "fig5_trajectory",
             f"Geometry trajectory — {split} split, layer {lay}, k={k_fix}, "
             f"{seed_note(seeds)}", tag)


# --- Figure 6: training health ----------------------------------------------

def fig_health(A, T, out_dir, seeds, tag):
    """Final lm_loss and final total loss vs lambda. Two measures on wildly
    different scales -- small multiples, never a second y-axis on one plot."""
    a = A[A["lbd"] > 0].copy()
    if a.empty:
        print("    [skip] health: no rows")
        return
    ks = sorted(a["k"].unique())
    cmap = dict(zip(ks, ramp_colors(T, len(ks))))
    panels = [("lm_loss_last", "Final lm_loss (LM term alone)"),
              ("loss_last", "Final total loss ($\\gamma\\cdot$lm + $\\lambda\\cdot$jepa)")]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for ax, (col, label) in zip(axes, panels):
        if col not in a.columns or a[col].notna().sum() == 0:
            ax.text(0.5, 0.5, f"{col} not recorded\nfor these cells", ha="center",
                    va="center", transform=ax.transAxes, color=T["muted"], fontsize=9)
        else:
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
    if h:
        fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.045),
                   ncol=len(lab), labelcolor=T["ink2"])
    fig.suptitle(f"Training health: does the LM term converge at every $\\lambda$? "
                 f"{seed_note(seeds)}  —  {tag}", color=T["ink"], fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    dump(a[["lbd", "k", "n_ok", "n_seeds", "lm_loss_last", "loss_last", "grad_norm_max"]]
         .sort_values(["k", "lbd"]), out_dir, "fig6_health")
    save(fig, out_dir, "fig6_health")


# --- Main --------------------------------------------------------------------

ALL = ["mediation", "dose", "grid", "layers", "trajectory", "health"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="dir holding the tidy CSVs")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--theme", choices=list(THEMES), default="light")
    p.add_argument("--seeds", default="all",
                   help="'all' (default) or a comma list, e.g. 82,23. Figures show "
                        "the mean over these seeds with the min-max spread.")
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--layer", type=int, default=None,
                   help="geometry layer for figures 1 and 5 (default: last)")
    p.add_argument("--k-fix", type=int, default=0,
                   help="k held fixed in the per-layer and trajectory figures")
    p.add_argument("--dataset", default=None, help="only render this dataset")
    p.add_argument("--model", default=None, help="only render this model (substring ok)")
    p.add_argument("--figures", nargs="+", choices=ALL + ["all"], default=["all"])
    args = p.parse_args()

    want = ALL if "all" in args.figures else args.figures
    T = THEMES[args.theme]
    apply_theme(T)

    pairs = discover(args.data_dir)
    if not pairs:
        raise SystemExit(f"no accuracy_tidy*.csv found in {args.data_dir}")

    for acc_path, geo_path, slug in pairs:
        acc, geo, seeds = load(acc_path, geo_path, args.seeds)
        dataset, model = identity(acc, slug)
        if args.dataset and args.dataset != dataset:
            continue
        if args.model and args.model not in model:
            continue

        A, G = agg_acc(acc), agg_geo(geo)
        b = A[A["lbd"] == BASELINE_LBD]
        baseline = float(b["accuracy"].iloc[0]) if len(b) else np.nan

        tag = f"{dataset}_{model}"
        out_dir = os.path.join(args.out_dir, tag)
        os.makedirs(out_dir, exist_ok=True)
        title_tag = f"{dataset} / {model}"

        print(f"\n{title_tag}: seeds {seeds}, {len(acc)} accuracy rows -> "
              f"{len(A)} cells, {len(geo)} geometry rows, baseline={baseline:.4f}")
        partial = A[A["n_ok"] < A["n_seeds"]]
        if len(partial):
            print(f"  {len(partial)} cell(s) have a seed that did not train cleanly "
                  f"— excluded from the mean, marked on the figures")

        if "mediation" in want:
            fig_mediation(A, G, acc, geo, baseline, T, out_dir, seeds, args.split,
                          args.layer, title_tag)
        if "dose" in want:
            fig_dose(A, baseline, T, out_dir, seeds, title_tag)
        if "grid" in want:
            fig_grid(A, baseline, T, out_dir, seeds, title_tag)
        if "layers" in want:
            fig_layers(G, geo, T, out_dir, seeds, args.split, args.k_fix, title_tag)
        if "trajectory" in want:
            fig_trajectory(G, geo, T, out_dir, seeds, args.split, args.k_fix,
                           args.layer, title_tag)
        if "health" in want:
            fig_health(A, T, out_dir, seeds, title_tag)


if __name__ == "__main__":
    main()
