"""Evaluate the trained BayesFlow posterior on REAL proteins (PISCES ground truth).

For each real chain we run the trained model to get per-residue P(helix), then
score it against the true DSSP label collapsed to the strict alpha-helix
convention: sst8 == 'H' -> helix (1), all other 7 states -> other (0).

We report AUC and accuracy@0.5 per chain (and pooled), and include exact
Forward-Backward on the same real sequences as a reference (the same fixed HMM,
so it shows how far a *perfect* inference of THIS model gets on real data).

Only chains with standard amino acids (has_nonstd_aa == False) are used, since
nonstandard residues are masked with '*' and have no emission probability.

Caveats (also in FINDINGS.md S18):
  * Absolute P(helix) is uncalibrated to real proteins, so AUC (ranking) is the fair
    metric; accuracy@0.5 barely clears the majority baseline.
  * Per-chain AUC and per-chain accuracy are averaged over BOTH-CLASS chains only (a
    single-class chain has no AUC); the two therefore describe the same population.
  * Pooled AUC (all residues) mixes within-chain and between-chain ranking, so it is not
    directly comparable to the per-chain mean -- it runs a little higher.
  * The HMM forces P(helix)=0 at residue 0 (startprob=[0,1]); on a real chain whose true
    first residue is a helix, the model is structurally wrong there. It is one residue per
    chain, kept in the scoring (it is what the model actually predicts) but noted here.

Run:  python eval_real.py            # ALL standard-AA chains (default; ~15 min at n=50)
      python eval_real.py --limit 200 # a quick, NON-representative subset for testing
"""

import argparse
import csv
import json
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from simulate import AA
from forward_backward import build_model, fb_posterior, encode
from train_bayesflow import predict_helix, load_model, NUM_SAMPLES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PISCES = os.path.join(HERE, "..", "archive", "2018-06-06-pdb-intersect-pisces.csv")
OUT_CSV = os.path.join(HERE, "outputs", "real_eval_per_chain.csv")
OUT_FIG = os.path.join(HERE, "figures", "real_eval_auc_hist.png")
OUT_JSON = os.path.join(HERE, "outputs", "metrics_real.json")
STD_AA = set(AA)


def helix_labels_from_sst8(sst8):
    """Strict alpha-helix labels: DSSP 'H' -> 1, everything else -> 0."""
    return np.array([1 if c == "H" else 0 for c in sst8], dtype=int)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=PISCES)
    ap.add_argument("--limit", type=int, default=0,
                    help="max standard-AA chains to score (0 = ALL; use a small N for a quick test)")
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--min-len", type=int, default=20)
    args = ap.parse_args(argv)

    if args.limit:
        print(f"WARNING: --limit {args.limit} scores only the FIRST {args.limit} chains "
              f"(top-of-file, not representative). Use --limit 0 for the reportable number.")
    approx = load_model()
    fb_model = build_model()

    rows_out = []
    bf_pool_p, fb_pool_p, pool_true = [], [], []
    aucs_bf, aucs_fb, acc_bf, acc_fb, base_all = [], [], [], [], []
    n_used = n_skip_nonstd = n_single_class = 0

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

            bf_p, _ = predict_helix(approx, encode(seq),
                                    num_samples=args.num_samples)
            fb_p, _ = fb_posterior(seq, fb_model)

            a_bf = ((bf_p > 0.5).astype(int) == true).mean()
            a_fb = ((fb_p > 0.5).astype(int) == true).mean()
            u_bf = roc_auc_score(true, bf_p) if both_classes else np.nan
            u_fb = roc_auc_score(true, fb_p) if both_classes else np.nan

            base = max(true.mean(), 1 - true.mean())      # this chain's trivial accuracy
            rows_out.append([row["pdb_id"], row["chain_code"], len(seq),
                             f"{true.mean():.3f}", f"{u_bf:.3f}", f"{a_bf:.3f}",
                             f"{u_fb:.3f}", f"{a_fb:.3f}", f"{base:.3f}"])
            if both_classes:                      # AUC and per-chain accuracy on the SAME set
                aucs_bf.append(u_bf); aucs_fb.append(u_fb)
                acc_bf.append(a_bf); acc_fb.append(a_fb)
                base_all.append(max(true.mean(), 1 - true.mean()))
            else:
                n_single_class += 1
            bf_pool_p.append(bf_p); fb_pool_p.append(fb_p); pool_true.append(true)
            n_used += 1
            if n_used % 200 == 0:
                print(f"  ...{n_used} chains scored")

    if n_used == 0:
        raise SystemExit("no standard-AA chains scored")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb_id", "chain_code", "len", "helix_frac",
                    "auc_bf", "acc_bf", "auc_fb", "acc_fb", "baseline"])
        w.writerows(rows_out)

    bf_pool_p = np.concatenate(bf_pool_p)
    fb_pool_p = np.concatenate(fb_pool_p)
    pool_true = np.concatenate(pool_true)

    print("\n=== real-protein evaluation (PISCES, sst8 H-only labels) ===")
    pooled_bf = float(roc_auc_score(pool_true, bf_pool_p))
    pooled_fb = float(roc_auc_score(pool_true, fb_pool_p))
    print(f"chains scored              : {n_used:,}  "
          f"(skipped {n_skip_nonstd:,} nonstandard-AA; {n_single_class:,} single-class "
          f"excluded from per-chain AUC/acc)")
    print(f"residues pooled            : {len(pool_true):,}   "
          f"true helix frac {pool_true.mean():.3f}")
    print(f"[per-chain, both-class only, n={len(aucs_bf):,}]")
    print(f"  AUC   : BayesFlow {np.mean(aucs_bf):.3f}   FB {np.mean(aucs_fb):.3f}")
    print(f"  acc@0.5: BayesFlow {np.mean(acc_bf):.3f}   FB {np.mean(acc_fb):.3f}   "
          f"majority baseline {np.mean(base_all):.3f}")
    print(f"[pooled over all residues]  AUC: BayesFlow {pooled_bf:.3f}   FB {pooled_fb:.3f}")
    print("  NB pooled AUC mixes within-chain and between-chain ranking, so it is not")
    print("  directly comparable to the per-chain mean above (it runs a touch higher).")
    print(f"saved per-chain -> {OUT_CSV}")
    with open(OUT_JSON, "w") as jf:
        json.dump({"setting": "real_pisces", "n_chains": int(n_used),
                   "n_both_class": int(len(aucs_bf)), "n_single_class": int(n_single_class),
                   "num_samples": int(args.num_samples),
                   "auc_bf": float(np.mean(aucs_bf)), "auc_fb": float(np.mean(aucs_fb)),
                   "acc_bf": float(np.mean(acc_bf)), "acc_fb": float(np.mean(acc_fb)),
                   "baseline": float(np.mean(base_all)),
                   "pooled_auc_bf": pooled_bf, "pooled_auc_fb": pooled_fb}, jf, indent=2)
    print(f"saved metrics   -> {OUT_JSON}")

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
