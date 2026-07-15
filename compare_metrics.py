"""Consolidated comparison of all evaluation metrics, with a majority-class baseline.

Rows are BayesFlow (the trained amortized posterior) vs exact Forward-Backward (the
reference Bayesian answer), each scored against ground truth by AUC and accuracy@0.5,
plus the TRIVIAL BASELINE (always predict the majority class) so accuracy is read
honestly.

Insulin note: 1MSO is WILD-TYPE human insulin. 1A7F -- used in an earlier version of
this analysis -- is a MUTANT (B16 Tyr->Glu, B24 Phe->Gly, des-B30) whose A-chain carries
a much sparser helix annotation; its flattering A-chain AUC (0.97) does not survive on
the wild-type structure (0.52). We report the wild-type.

Run:  python compare_metrics.py   ->  comparison.png, comparison_table.png, printed table
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_CSV = os.path.join(HERE, "real_eval_per_chain.csv")
OUT_FIG = os.path.join(HERE, "comparison.png")
OUT_TBL = os.path.join(HERE, "comparison_table.png")

# (auc_bf, auc_fb, acc_bf, acc_fb, helix_fraction)  -- helix_frac gives the baseline
ROWS = {
    "Held-out\nsimulated":            (0.798, 0.798, 0.760, 0.760, 0.325),
    "Insulin B\n(wild-type 1MSO)":    (0.981, 0.986, 19/30, 19/30, 11/30),
    "Insulin A\n(wild-type 1MSO)":    (0.519, 0.509, 9/21, 9/21, 12/21),
}


def real_row(path=REAL_CSV):
    """Mean per-chain AUC / accuracy and helix fraction over all real PISCES chains."""
    a_bf, a_fb, c_bf, c_fb, hf = [], [], [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            c_bf.append(float(r["acc_bf"])); c_fb.append(float(r["acc_fb"]))
            hf.append(float(r["helix_frac"]))
            for lst, k in ((a_bf, "auc_bf"), (a_fb, "auc_fb")):
                v = float(r[k])
                if not np.isnan(v):
                    lst.append(v)
    return (float(np.mean(a_bf)), float(np.mean(a_fb)),
            float(np.mean(c_bf)), float(np.mean(c_fb)),
            float(np.mean(hf)), len(c_bf))


def main():
    rows = dict(ROWS)
    r = real_row()
    rows[f"Real PISCES\n({r[5]:,} chains)"] = (r[0], r[1], r[2], r[3], r[4])

    labels = list(rows.keys())
    auc_bf = [rows[k][0] for k in labels]
    auc_fb = [rows[k][1] for k in labels]
    acc_bf = [rows[k][2] for k in labels]
    acc_fb = [rows[k][3] for k in labels]
    # trivial baseline = always predict the majority class
    base = [max(rows[k][4], 1 - rows[k][4]) for k in labels]

    print("=" * 92)
    print(f"{'setting':<26}{'AUC BF':>8}{'AUC FB':>8}{'acc BF':>8}{'acc FB':>8}"
          f"{'baseline':>10}{'beats base?':>13}")
    print("-" * 92)
    for k, b in zip(labels, base):
        a1, a2, c1, c2, _ = rows[k]
        verdict = "yes" if c1 > b + 1e-6 else ("ties it" if abs(c1 - b) < 1e-6 else "NO, below")
        print(f"{k.replace(chr(10),' '):<26}{a1:>8.3f}{a2:>8.3f}{c1:>8.3f}{c2:>8.3f}"
              f"{b:>10.3f}{verdict:>13}")
    print("=" * 92)
    print("accuracy@0.5 is at or below the trivial baseline on insulin: the model never")
    print("crosses P=0.5 there (uncalibrated to real proteins) -> AUC is the fair metric.")

    # ---- grouped bar chart --------------------------------------------------
    x = np.arange(len(labels)); w = 0.27
    TEAL, GRAY, RED = "#1D9E75", "#B8B5AA", "#D85A30"
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))

    b1 = ax[0].bar(x - w/2, auc_bf, w, label="BayesFlow", color=TEAL)
    b2 = ax[0].bar(x + w/2, auc_fb, w, label="exact FB", color=GRAY)
    ax[0].axhline(0.5, color=RED, ls="--", lw=1.2, label="chance (AUC 0.5)")
    ax[0].set(ylim=(0, 1.05), ylabel="AUC", title="AUC vs ground truth")
    for bars in (b1, b2):
        for rect in bars:
            ax[0].annotate(f"{rect.get_height():.2f}",
                           (rect.get_x() + rect.get_width()/2, rect.get_height()),
                           ha="center", va="bottom", fontsize=7)

    c1 = ax[1].bar(x - w, acc_bf, w, label="BayesFlow", color=TEAL)
    c2 = ax[1].bar(x, acc_fb, w, label="exact FB", color=GRAY)
    c3 = ax[1].bar(x + w, base, w, label="majority-class baseline", color=RED, alpha=0.85)
    ax[1].set(ylim=(0, 1.05), ylabel="accuracy@0.5",
              title="accuracy@0.5 vs the trivial baseline")
    for bars in (c1, c2, c3):
        for rect in bars:
            ax[1].annotate(f"{rect.get_height():.2f}",
                           (rect.get_x() + rect.get_width()/2, rect.get_height()),
                           ha="center", va="bottom", fontsize=7)

    for a in ax:
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=9)
        a.legend(fontsize=8, loc="lower right")
    fig.suptitle("BayesFlow vs exact Forward-Backward vs ground truth", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_FIG, dpi=130)
    print(f"saved figure -> {OUT_FIG}")

    # ---- table image --------------------------------------------------------
    cells = []
    for k, b in zip(labels, base):
        a1, a2, c1, c2, _ = rows[k]
        cells.append([k.replace("\n", " "), f"{a1:.3f}", f"{a2:.3f}",
                      f"{c1:.3f}", f"{c2:.3f}", f"{b:.3f}"])
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
