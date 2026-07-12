"""Step 4: test the trained model on REAL human insulin and compare to ground truth.

Pulls human insulin (default PDB 1A7F, both chains) straight from the real
dataset -- sequence AND its DSSP secondary-structure annotation -- and scores
the trained BayesFlow posterior against the true helix labels (strict: sst8 'H'
-> helix, all other states -> other). Exact Forward-Backward on the same
sequences is shown as a reference.

The model was trained ONLY on simulated chains, so insulin is a genuinely
held-out real protein (no leakage of any kind).

Produces `insulin_prediction.png` (per-residue: annotated helix shaded, exact FB,
BayesFlow mean + uncertainty band) and prints AUC / accuracy per chain.

Run:  python insulin_eval.py
      python insulin_eval.py --pdb-id 1MSO      # a crystal structure instead
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
from forward_backward import build_model, fb_posterior, encode
from train_bayesflow import predict_helix

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "bayesflow_posterior.keras")
DATA = os.path.join(HERE, "..", "archive", "2018-06-06-ss.cleaned.csv")
OUT_FIG = os.path.join(HERE, "insulin_prediction.png")


def load_approx(path=MODEL_PATH):
    import bayesflow as bf
    import keras
    return keras.saving.load_model(path)


def helix_runs(mask):
    """Contiguous (start, end_exclusive) index ranges where mask == 1."""
    d = np.diff(np.concatenate([[0], mask, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def fetch_chains(pdb_id, data=DATA):
    """All chains for a PDB id: list of (chain_code, seq, sst8)."""
    out = []
    with open(data, newline="") as f:
        for row in csv.DictReader(f):
            if row["pdb_id"] == pdb_id and row.get("has_nonstd_aa") != "True":
                out.append((row["chain_code"], row["seq"], row["sst8"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb-id", default="1A7F")
    ap.add_argument("--input", default=DATA)
    ap.add_argument("--num-samples", type=int, default=500)
    args = ap.parse_args(argv)

    chains = fetch_chains(args.pdb_id, args.input)
    if not chains:
        raise SystemExit(f"{args.pdb_id} not found (or only nonstandard-AA chains)")

    approx = load_approx()
    fb_model = build_model()

    fig, axes = plt.subplots(len(chains), 1, figsize=(11, 3.4 * len(chains)),
                             squeeze=False)
    print(f"=== human insulin {args.pdb_id}: BayesFlow vs ground truth (sst8 H-only) ===")
    for ax, (code, seq, sst8) in zip(axes[:, 0], chains):
        obs = encode(seq)
        true = np.array([1 if c == "H" else 0 for c in sst8], dtype=int)
        both = 0 < true.sum() < len(true)

        bf_p, bf_s = predict_helix(approx, obs, num_samples=args.num_samples)
        fb_p, _ = fb_posterior(seq, fb_model)

        auc_bf = roc_auc_score(true, bf_p) if both else float("nan")
        auc_fb = roc_auc_score(true, fb_p) if both else float("nan")
        acc_bf = ((bf_p > 0.5).astype(int) == true).mean()
        acc_fb = ((fb_p > 0.5).astype(int) == true).mean()
        print(f"chain {code} (len {len(seq)}, helix residues {int(true.sum())}): "
              f"BayesFlow AUC={auc_bf:.3f} acc={acc_bf:.3f} | "
              f"FB AUC={auc_fb:.3f} acc={acc_fb:.3f}")

        xs = np.arange(len(seq))
        for s, e in helix_runs(true):
            ax.axvspan(s - 0.5, e - 0.5, color="gold", alpha=0.30,
                       label="annotated helix (truth)" if s == helix_runs(true)[0][0] else None)
        ax.plot(xs, fb_p, "-o", color="black", ms=3, lw=1.4, label="exact FB")
        ax.plot(xs, bf_p, "-s", color="#1D9E75", ms=3, lw=1.4, label="BayesFlow mean")
        ax.fill_between(xs, np.clip(bf_p - bf_s, 0, 1), np.clip(bf_p + bf_s, 0, 1),
                        color="#1D9E75", alpha=0.22)
        ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels(list(seq), fontsize=7)
        ax.set(ylabel="P(helix)", ylim=(-0.05, 1.05),
               title=f"insulin {args.pdb_id} chain {code}   "
                     f"(BayesFlow AUC={auc_bf:.2f}, FB AUC={auc_fb:.2f})")
        ax.legend(fontsize=8, loc="upper right", ncol=3)
    axes[-1, 0].set_xlabel("residue")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130)
    print(f"saved figure -> {OUT_FIG}")


if __name__ == "__main__":
    main()
