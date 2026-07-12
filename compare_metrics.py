"""Consolidated comparison of all evaluation metrics.

Puts every result side by side: BayesFlow (the trained neural posterior) vs exact
Forward-Backward (the reference Bayesian answer), each scored against ground truth
by AUC and accuracy@0.5, across four settings:

  - Held-out simulated chains (the model's own world; disjoint tail block)
  - Human insulin 1A7F chain A   (real, held-out)
  - Human insulin 1A7F chain B   (real, held-out)
  - All real PISCES chains        (real, held-out; ~9k chains)

The real-PISCES row is computed live from `real_eval_per_chain.csv`; the other
rows are the verified outputs of `train_bayesflow.py` (validation), `make_figures.py`
and `insulin_eval.py` from the same session.

Run:  python compare_metrics.py   ->  comparison.png  + printed table
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_CSV = os.path.join(HERE, "real_eval_per_chain.csv")
OUT = os.path.join(HERE, "comparison.png")

# --- Results carried over from their respective runs (this session) ----------
# (auc_bf, auc_fb, acc_bf, acc_fb)
ROWS = {
    "Held-out\nsimulated": (0.798, 0.798, 0.760, 0.760),
    "Insulin A": (0.971, 0.971, 0.810, 0.810),
    "Insulin B": (0.984, 0.984, 0.793, 0.793),
}
# Extra note: on held-out simulated, BayesFlow vs exact FB agree at corr=0.999, MAE=0.007.


def real_pisces_row(path=REAL_CSV):
    """Mean per-chain AUC/accuracy over all real PISCES chains, from the saved CSV."""
    auc_bf, auc_fb, acc_bf, acc_fb = [], [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            acc_bf.append(float(r["acc_bf"])); acc_fb.append(float(r["acc_fb"]))
            for lst, key in ((auc_bf, "auc_bf"), (auc_fb, "auc_fb")):
                v = float(r[key])
                if not np.isnan(v):
                    lst.append(v)
    return (float(np.mean(auc_bf)), float(np.mean(auc_fb)),
            float(np.mean(acc_bf)), float(np.mean(acc_fb)), len(acc_bf), len(auc_bf))


def main():
    rows = dict(ROWS)
    r_auc_bf, r_auc_fb, r_acc_bf, r_acc_fb, n_chain, n_auc = real_pisces_row()
    rows[f"Real PISCES\n({n_chain:,} chains)"] = (r_auc_bf, r_auc_fb, r_acc_bf, r_acc_fb)

    labels = list(rows.keys())
    auc_bf = [rows[k][0] for k in labels]
    auc_fb = [rows[k][1] for k in labels]
    acc_bf = [rows[k][2] for k in labels]
    acc_fb = [rows[k][3] for k in labels]

    # ---- printed table ------------------------------------------------------
    print("=" * 68)
    print(f"{'setting':<22}{'AUC BF':>9}{'AUC FB':>9}{'acc BF':>9}{'acc FB':>9}")
    print("-" * 68)
    for k in labels:
        a_bf, a_fb, c_bf, c_fb = rows[k]
        print(f"{k.replace(chr(10),' '):<22}{a_bf:>9.3f}{a_fb:>9.3f}{c_bf:>9.3f}{c_fb:>9.3f}")
    print("=" * 68)
    print(f"real-PISCES row computed from {n_auc:,} chains with both classes")
    print("held-out simulated: BayesFlow vs exact FB agree at corr=0.999, MAE=0.007")

    # ---- grouped bar chart --------------------------------------------------
    x = np.arange(len(labels)); w = 0.38
    TEAL, GRAY = "#1D9E75", "#B8B5AA"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

    for a, (bf, fb, title) in zip(
            ax, [(auc_bf, auc_fb, "AUC vs ground truth"),
                 (acc_bf, acc_fb, "accuracy@0.5 vs ground truth")]):
        b1 = a.bar(x - w/2, bf, w, label="BayesFlow", color=TEAL)
        b2 = a.bar(x + w/2, fb, w, label="exact FB", color=GRAY)
        a.axhline(0.5, color="0.6", ls=":", lw=1, label="chance (0.5)")
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=9)
        a.set_ylim(0, 1.05); a.set_ylabel(title.split(" vs")[0])
        a.set_title(title)
        a.legend(fontsize=8, loc="lower right")
        for bars in (b1, b2):
            for rect in bars:
                a.annotate(f"{rect.get_height():.2f}",
                           (rect.get_x() + rect.get_width()/2, rect.get_height()),
                           ha="center", va="bottom", fontsize=7)
    fig.suptitle("BayesFlow vs exact Forward-Backward, scored against ground truth",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=130)
    print(f"saved figure -> {OUT}")


if __name__ == "__main__":
    main()
