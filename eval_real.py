"""Evaluate the trained BayesFlow posterior on REAL proteins (PISCES ground truth).

For each real chain we run the trained model to get per-residue P(helix), then
score it against the true DSSP label collapsed to the strict alpha-helix
convention: sst8 == 'H' -> helix (1), all other 7 states -> other (0).

We report AUC and accuracy@0.5 per chain (and pooled), and include exact
Forward-Backward on the same real sequences as a reference (the same fixed HMM,
so it shows how far a *perfect* inference of THIS model gets on real data).

Only chains with standard amino acids (has_nonstd_aa == False) are used, since
nonstandard residues are masked with '*' and have no emission probability.

Note on metrics: our fixed two-state HMM is a strong simplification of real
protein structure, so absolute P(helix) is not calibrated to reality -- expect
AUC (ranking) to be informative while accuracy@0.5 may be depressed by that
calibration gap. AUC is the fair metric here.

Run:  python eval_real.py            # first 1000 standard-AA chains
      python eval_real.py --limit 0  # all standard-AA chains (slow)
"""

import argparse
import csv
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from simulate import AA
from forward_backward import build_model, fb_posterior
from train_bayesflow import predict_helix

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "bayesflow_posterior.keras")
PISCES = os.path.join(HERE, "..", "archive", "2018-06-06-pdb-intersect-pisces.csv")
OUT_CSV = os.path.join(HERE, "real_eval_per_chain.csv")
OUT_FIG = os.path.join(HERE, "real_eval_auc_hist.png")
STD_AA = set(AA)


def load_approx(path=MODEL_PATH):
    import bayesflow as bf
    import keras
    return keras.saving.load_model(path)


def helix_labels_from_sst8(sst8):
    """Strict alpha-helix labels: DSSP 'H' -> 1, everything else -> 0."""
    return np.array([1 if c == "H" else 0 for c in sst8], dtype=int)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=PISCES)
    ap.add_argument("--limit", type=int, default=1000,
                    help="max standard-AA chains to score (0 = all)")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--min-len", type=int, default=20)
    args = ap.parse_args(argv)

    approx = load_approx()
    fb_model = build_model()

    rows_out = []
    bf_pool_p, fb_pool_p, pool_true = [], [], []
    aucs_bf, aucs_fb, acc_bf, acc_fb = [], [], [], []
    n_used = n_skip_nonstd = n_skip_degenerate = 0

    with open(args.input, newline="") as f:
        for row in csv.DictReader(f):
            if args.limit and n_used >= args.limit:
                break
            seq = row["seq"]
            if row.get("has_nonstd_aa", "False") == "True" or not set(seq) <= STD_AA:
                n_skip_nonstd += 1
                continue
            if len(seq) < args.min_len:
                continue

            true = helix_labels_from_sst8(row["sst8"])
            both_classes = 0 < true.sum() < len(true)

            bf_p, _ = predict_helix(approx, np.array([AA.index(c) for c in seq]),
                                    num_samples=args.num_samples)
            fb_p, _ = fb_posterior(seq, fb_model)

            a_bf = ((bf_p > 0.5).astype(int) == true).mean()
            a_fb = ((fb_p > 0.5).astype(int) == true).mean()
            u_bf = roc_auc_score(true, bf_p) if both_classes else np.nan
            u_fb = roc_auc_score(true, fb_p) if both_classes else np.nan

            rows_out.append([row["pdb_id"], row["chain_code"], len(seq),
                             f"{true.mean():.3f}", f"{u_bf:.3f}", f"{a_bf:.3f}",
                             f"{u_fb:.3f}", f"{a_fb:.3f}"])
            acc_bf.append(a_bf); acc_fb.append(a_fb)
            if both_classes:
                aucs_bf.append(u_bf); aucs_fb.append(u_fb)
            bf_pool_p.append(bf_p); fb_pool_p.append(fb_p); pool_true.append(true)
            n_used += 1
            if n_used % 200 == 0:
                print(f"  ...{n_used} chains scored")

    if n_used == 0:
        raise SystemExit("no standard-AA chains scored")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb_id", "chain_code", "len", "helix_frac",
                    "auc_bf", "acc_bf", "auc_fb", "acc_fb"])
        w.writerows(rows_out)

    bf_pool_p = np.concatenate(bf_pool_p)
    fb_pool_p = np.concatenate(fb_pool_p)
    pool_true = np.concatenate(pool_true)

    print("\n=== real-protein evaluation (PISCES, sst8 H-only labels) ===")
    print(f"chains scored              : {n_used:,}  "
          f"(skipped {n_skip_nonstd:,} with nonstandard AA)")
    print(f"residues pooled            : {len(pool_true):,}   "
          f"true helix frac {pool_true.mean():.3f}")
    print(f"per-chain AUC   (mean)     : BayesFlow {np.mean(aucs_bf):.3f}   "
          f"FB {np.mean(aucs_fb):.3f}   (n={len(aucs_bf):,} chains w/ both classes)")
    print(f"per-chain acc@0.5 (mean)   : BayesFlow {np.mean(acc_bf):.3f}   "
          f"FB {np.mean(acc_fb):.3f}")
    print(f"pooled AUC (all residues)  : BayesFlow "
          f"{roc_auc_score(pool_true, bf_pool_p):.3f}   "
          f"FB {roc_auc_score(pool_true, fb_pool_p):.3f}")
    print(f"saved per-chain -> {OUT_CSV}")

    # Histogram of per-chain AUC.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(aucs_bf, bins=30, range=(0, 1), color="#1D9E75", alpha=0.75,
            label=f"BayesFlow (mean {np.mean(aucs_bf):.3f})")
    ax.hist(aucs_fb, bins=30, range=(0, 1), histtype="step", color="black", lw=1.5,
            label=f"exact FB (mean {np.mean(aucs_fb):.3f})")
    ax.axvline(0.5, color="0.7", ls=":", lw=1)
    ax.set(xlabel="per-chain AUC (helix vs other)", ylabel="number of chains",
           title=f"Real proteins (PISCES), n={len(aucs_bf):,} chains")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130)
    print(f"saved figure   -> {OUT_FIG}")


if __name__ == "__main__":
    main()
