#!/usr/bin/env python3
"""Tables for Phase A (diagnostics/jepa_probe.py output). CPU only, read-only.

    python3 diagnostics/jepa_probe_report.py [--dir=diag_runs/jepa_null/phaseA]

Writes <dir>/report.txt plus tidy CSVs embed.csv / pairs.csv (all shards merged).
Cells are averaged over seeds; the spread is shown as min-max where it matters.
"""
import argparse
import collections
import csv
import glob
import json
import os
import statistics as st


def rows(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


def tag(r):
    if r["cell"] == "base":
        return "base"
    return r["cell"].rsplit("_", 1)[0]  # "<lbd>_<k>", as the on-disk tag


def sort_key(t):
    if t == "base":
        return (-2, 0)
    lbd, k = t.split("_")
    return (float(lbd), int(k))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def write_csv(path, rs):
    keys = sorted({k for r in rs for k in r if not isinstance(r[k], dict)})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="diag_runs/jepa_null/phaseA")
    a = ap.parse_args()
    E = rows(os.path.join(a.dir, "embed.*.jsonl"))
    P = rows(os.path.join(a.dir, "pairs.*.jsonl"))
    write_csv(os.path.join(a.dir, "embed.csv"), E)
    write_csv(os.path.join(a.dir, "pairs.csv"), P)
    L = []
    last = max(r["layer"] for r in E)

    # ---- A1/A2: last layer, test split ----
    for view in ("train", "raw"):
        L += ["", f"=== A1/A2 last layer ({last}), test split, view={view} — mean over seeds ===",
              f"{'ds':5} {'cell':8} {'n':>2} {'pos':>7} {'neg':>7} {'gap':>7} {'r1':>6} {'r1ctr':>6} "
              f"{'gapctr':>7} {'rankT':>6} {'rankC':>6} {'H_t':>6} {'peot_t':>6} {'H_c':>6} {'peot_c':>6}"]
        g = collections.defaultdict(list)
        for r in E:
            if r["layer"] == last and r["split"] == "test" and (r["view"] == view or (view == "raw" and r["k"] == 0)):
                g[(r["dataset"], tag(r))].append(r)
        for (ds, t) in sorted(g, key=lambda x: (x[0], sort_key(x[1]))):
            rs = g[(ds, t)]
            m = lambda c: mean(r.get(c) for r in rs)  # noqa: E731
            L.append(f"{ds:5} {t:8} {len(rs):2d} {m('pos_cos'):7.4f} {m('neg_cos'):7.4f} {m('gap'):7.4f} "
                     f"{m('r1_t2c'):6.3f} {m('r1_t2c_ctr'):6.3f} {m('gap_ctr'):7.4f} {m('rankme_t'):6.1f} "
                     f"{m('rankme_c'):6.1f} {m('next_entropy_t'):6.2f} {m('p_eot_t'):6.3f} "
                     f"{m('next_entropy_c'):6.2f} {m('p_eot_c'):6.3f}")
    n_embed = max((r["n"] for r in E), default=0)
    L.append(f"(retrieval chance = 1/{n_embed} = {1 / max(n_embed, 1):.4f})")

    # ---- A1 per layer: where does the collapse start? ----
    L += ["", "=== A1 by layer, test split, view=train — pos-neg gap (centred) | r1 (centred) | rankT/rankC ==="]
    g = collections.defaultdict(list)
    for r in E:
        if r["split"] == "test" and r["view"] == "train":
            g[(r["dataset"], tag(r), r["layer"])].append(r)
    for ds in sorted({k[0] for k in g}):
        L.append(f"-- {ds}")
        for t in sorted({k[1] for k in g if k[0] == ds}, key=sort_key):
            cells = []
            for layer in (2, 4, 8, 12, 14, 15, 16):
                rs = g.get((ds, t, layer))
                if rs:
                    cells.append(f"L{layer}:{mean(r['gap_ctr'] for r in rs):.3f}|{mean(r['r1_t2c_ctr'] for r in rs):.2f}|"
                                 f"{mean(r['rankme_t'] for r in rs):.0f}/{mean(r['rankme_c'] for r in rs):.0f}")
            L.append(f"   {t:8} " + "  ".join(cells))

    # ---- A3: function and weight distance ----
    L += ["", "=== A3 pair distances (test, teacher-forced gold+eot) — mean [min-max] over pairs ===",
          f"{'ds':5} {'kind':13} {'cell':8} {'n':>2} {'KLsym (min-max)':>22} {'argmax≠':>22} {'|dθ|/|θ|':>22} "
          f"{'match_agree':>12} {'resp_same':>10}"]
    g = collections.defaultdict(list)
    for r in P:
        cell = os.path.basename(r["b"]).rsplit("_", 1)[0] if r["kind"] in ("jepa_vs_base", "jepa_seed") else "-"
        g[(r["dataset"], r["kind"], cell)].append(r)
    order = {"ft_vs_init": 0, "base_seed": 1, "jepa_seed": 2, "jepa_vs_base": 3}
    for key in sorted(g, key=lambda x: (x[0], order[x[1]], sort_key(x[2]) if x[2] != "-" else (-3, 0))):
        rs = g[key]
        def rng(c, fmt="{:.4f}"):
            v = [r[c] for r in rs if c in r]
            if not v:
                return "-".rjust(22)
            return f"{fmt.format(st.mean(v))} ({fmt.format(min(v))}-{fmt.format(max(v))})".rjust(22)
        L.append(f"{key[0]:5} {key[1]:13} {key[2]:8} {len(rs):2d} {rng('kl_sym')} {rng('argmax_disagree')} "
                 f"{rng('dtheta_rel')} {mean(r.get('match_agree') for r in rs):12.4f} "
                 f"{mean(r.get('response_same') for r in rs):10.4f}")

    # ratio: same-seed JEPA-vs-baseline over seed-vs-seed baseline
    L += ["", "=== A3 ratio: jepa_vs_base / base_seed (1 = indistinguishable from seed noise) ==="]
    for ds in sorted({r["dataset"] for r in P}):
        bs = [r for r in P if r["dataset"] == ds and r["kind"] == "base_seed"]
        if not bs:
            continue
        ref = {c: mean(r[c] for r in bs) for c in ("kl_sym", "argmax_disagree", "dtheta_rel")}
        ref["mis"] = mean(1 - r["match_agree"] for r in bs)
        for cell in sorted({k[2] for k in g if k[0] == ds and k[1] == "jepa_vs_base"}, key=sort_key):
            rs = g[(ds, "jepa_vs_base", cell)]
            L.append(f"{ds:5} {cell:8} KL x{mean(r['kl_sym'] for r in rs) / ref['kl_sym']:5.2f}  "
                     f"argmax≠ x{mean(r['argmax_disagree'] for r in rs) / ref['argmax_disagree']:5.2f}  "
                     f"|dθ| x{mean(r['dtheta_rel'] for r in rs) / ref['dtheta_rel']:5.2f}  "
                     f"match-disagree x{mean(1 - r['match_agree'] for r in rs) / ref['mis']:5.2f}")

    # ---- A3 per-layer weight distance ----
    L += ["", "=== A3 |dθ|/|θ| by block (mean over pairs) ==="]
    for ds in sorted({r["dataset"] for r in P}):
        for kind in ("base_seed", "jepa_vs_base"):
            rs = [r for r in P if r["dataset"] == ds and r["kind"] == kind]
            if rs:
                blocks = sorted(rs[0]["dtheta_by_layer"])
                L.append(f"{ds:5} {kind:13} " + " ".join(
                    f"{b}:{mean(r['dtheta_by_layer'][b] for r in rs):.4f}" for b in blocks))

    # ---- A4: answer-only held-out NLL and accuracy per cell ----
    L += ["", "=== A4 answer-only test NLL (gold tokens, teacher-forced) and exact accuracy — mean over seeds ==="]
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in P:
        if r["kind"] == "jepa_vs_base":
            per[(r["dataset"], "0.0_0")][r["a"]].append((r["nll_a"], r.get("acc_a")))
            per[(r["dataset"], os.path.basename(r["b"]).rsplit("_", 1)[0])][r["b"]].append((r["nll_b"], r.get("acc_b")))
    for key in sorted(per, key=lambda x: (x[0], sort_key(x[1]))):
        vals = [v[0] for v in per[key].values()]
        nll = [x[0] for x in vals]
        acc = [x[1] for x in vals if x[1] is not None]
        L.append(f"{key[0]:5} {key[1]:8} seeds={len(vals)} nll {mean(nll):.4f} [{min(nll):.4f}-{max(nll):.4f}]  "
                 f"acc {mean(acc):.4f}")

    rep = "\n".join(L)
    with open(os.path.join(a.dir, "report.txt"), "w") as f:
        f.write(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
