"""Consolidated comparison of all evaluation metrics -- DATA-DRIVEN, no hardcoded numbers.

Reads the metric files each eval script writes, so the table can never silently go stale
after a retrain (the previous hardcoded-ROWS version did, twice):

    metrics_sim.json            <- make_figures.py   (held-out simulated)
    metrics_insulin_1MSO.json   <- insulin_eval.py   (wild-type insulin)
    metrics_real.json  (or real_eval_per_chain.csv)  <- eval_real.py   (all PISCES)

Each row is BayesFlow vs exact Forward-Backward vs the MAJORITY-CLASS BASELINE (always
predict the majority class), so accuracy@0.5 is read honestly. Note the baseline is a
per-chain mean where applicable -- mean_i(max(hf_i, 1-hf_i)), NOT max(mean(hf)); the latter
is biased low by Jensen (E[max] >= max(E)).

Insulin note: 1MSO is WILD-TYPE. 1A7F (an earlier choice) is a mutant whose sparse A-chain
annotation inflates its A-chain AUC; we report wild-type.

Run:  python compare_metrics.py   ->  comparison.png, comparison_table.png, printed table
"""

import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_CSV = os.path.join(HERE, "outputs", "real_eval_per_chain.csv")
REAL_JSON = os.path.join(HERE, "outputs", "metrics_real.json")
SIM_JSON = os.path.join(HERE, "outputs", "metrics_sim.json")
INS_JSON = os.path.join(HERE, "outputs", "metrics_insulin_1MSO.json")
OUT_FIG = os.path.join(HERE, "figures", "comparison.png")
OUT_TBL = os.path.join(HERE, "figures", "comparison_table.png")


def _load(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {os.path.basename(path)} -- run the eval script that "
                         f"produces it first (see this file's docstring).")
    with open(path) as f:
        return json.load(f)


def real_from_csv(path=REAL_CSV):
    """Fallback if metrics_real.json is absent: recompute the PISCES row from the CSV.

    Per-chain AUC/acc/baseline are averaged over both-class chains only, so all three
    describe the same population.
    """
    a_bf, a_fb, c_bf, c_fb, base = [], [], [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            u = float(r["auc_bf"])
            if np.isnan(u):                      # single-class chain: no AUC, exclude
                continue
            a_bf.append(u); a_fb.append(float(r["auc_fb"]))
            c_bf.append(float(r["acc_bf"])); c_fb.append(float(r["acc_fb"]))
            hf = float(r["helix_frac"]); base.append(max(hf, 1 - hf))
    return {"auc_bf": float(np.mean(a_bf)), "auc_fb": float(np.mean(a_fb)),
            "acc_bf": float(np.mean(c_bf)), "acc_fb": float(np.mean(c_fb)),
            "baseline": float(np.mean(base)), "n_both_class": len(a_bf)}


def build_rows():
    rows = {}
    sim = _load(SIM_JSON)
    rows[f"Held-out\nsimulated"] = sim

    ins = _load(INS_JSON)
    by_code = {c["code"]: c for c in ins["chains"]}
    for code, lab in [("B", "Insulin B"), ("A", "Insulin A")]:
        if code in by_code:
            rows[f"{lab}\n(wild-type {ins['pdb']})"] = by_code[code]

    real = _load(REAL_JSON) if os.path.exists(REAL_JSON) else real_from_csv()
    rows[f"Real PISCES\n({real['n_both_class']:,} chains)"] = real
    return rows


def main():
    rows = build_rows()
    labels = list(rows.keys())
    g = lambda k: [float(rows[L][k]) for L in labels]
    auc_bf, auc_fb = g("auc_bf"), g("auc_fb")
    acc_bf, acc_fb = g("acc_bf"), g("acc_fb")
    base = g("baseline")

    print("=" * 96)
    print(f"{'setting':<26}{'AUC BF':>8}{'AUC FB':>8}{'acc BF':>8}{'acc FB':>8}"
          f"{'baseline':>10}{'acc beats base?':>16}")
    print("-" * 96)
    for L, a1, a2, c1, c2, b in zip(labels, auc_bf, auc_fb, acc_bf, acc_fb, base):
        v = "yes" if c1 > b + 1e-6 else ("ties it" if abs(c1 - b) < 1e-6 else "NO, below")
        print(f"{L.replace(chr(10), ' '):<26}{a1:>8.3f}{a2:>8.3f}{c1:>8.3f}{c2:>8.3f}"
              f"{b:>10.3f}{v:>16}")
    print("=" * 96)
    print("accuracy@0.5 barely clears the baseline on real data and ties/loses on insulin")
    print("(the model rarely crosses P=0.5) -> AUC is the fair metric, not accuracy.")

    # ---- grouped bar chart --------------------------------------------------
    x = np.arange(len(labels)); w = 0.27
    TEAL, GRAY, RED = "#1D9E75", "#B8B5AA", "#D85A30"
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))

    for rects, vals, off, lab, col in [
            ("auc", auc_bf, -w/2, "BayesFlow", TEAL), ("auc", auc_fb, w/2, "exact FB", GRAY)]:
        b = ax[0].bar(x + off, vals, w, label=lab, color=col)
        for r in b:
            ax[0].annotate(f"{r.get_height():.2f}",
                           (r.get_x() + r.get_width()/2, r.get_height()),
                           ha="center", va="bottom", fontsize=7)
    ax[0].axhline(0.5, color=RED, ls="--", lw=1.2, label="chance (AUC 0.5)")
    ax[0].set(ylim=(0, 1.05), ylabel="AUC", title="AUC vs ground truth")

    for vals, off, lab, col in [(acc_bf, -w, "BayesFlow", TEAL), (acc_fb, 0, "exact FB", GRAY),
                                (base, w, "majority-class baseline", RED)]:
        b = ax[1].bar(x + off, vals, w, label=lab, color=col,
                      alpha=0.85 if lab.startswith("majority") else 1.0)
        for r in b:
            ax[1].annotate(f"{r.get_height():.2f}",
                           (r.get_x() + r.get_width()/2, r.get_height()),
                           ha="center", va="bottom", fontsize=7)
    ax[1].set(ylim=(0, 1.05), ylabel="accuracy@0.5",
              title="accuracy@0.5 vs the trivial baseline")

    for a in ax:
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=9)
        a.legend(fontsize=8, loc="lower right")
    fig.suptitle("BayesFlow vs exact Forward-Backward vs ground truth", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_FIG, dpi=130)
    print(f"saved figure -> {OUT_FIG}")

    # ---- table image --------------------------------------------------------
    cells = [[L.replace("\n", " "), f"{a1:.3f}", f"{a2:.3f}", f"{c1:.3f}", f"{c2:.3f}", f"{b:.3f}"]
             for L, a1, a2, c1, c2, b in zip(labels, auc_bf, auc_fb, acc_bf, acc_fb, base)]
    cols = ["Setting", "AUC\nBayesFlow", "AUC\nexact FB",
            "Acc@0.5\nBayesFlow", "Acc@0.5\nexact FB", "Majority\nbaseline"]
    figt, axt = plt.subplots(figsize=(13.5, 3.2)); axt.axis("off")
    t = axt.table(cellText=cells, colLabels=cols, cellLoc="center", loc="center",
                  colWidths=[0.30, 0.14, 0.14, 0.14, 0.14, 0.14])
    t.auto_set_font_size(False); t.set_fontsize(11.5); t.scale(1, 2.15)
    for j in range(len(cols)):
        c = t[0, j]; c.set_facecolor("#0F6E56"); c.set_text_props(color="white", weight="bold")
    for i in range(1, len(cells) + 1):
        for j in range(len(cols)):
            t[i, j].set_facecolor("#E1F5EE" if i % 2 else "white")
        t[i, 0].set_text_props(weight="bold")
    figt.savefig(OUT_TBL, dpi=150, bbox_inches="tight")
    print(f"saved table  -> {OUT_TBL}")


if __name__ == "__main__":
    main()
