"""Validation figures for the trained BayesFlow posterior.

Produces `validation_figure.png` with two panels:
  (left)  BayesFlow posterior mean vs exact Forward-Backward P(helix), per residue,
          over held-out simulated chains (the disjoint tail block, never trained on).
  (right) one example chain: true hidden state, exact FB, BayesFlow mean, and the
          BayesFlow +/-1 std uncertainty band drawn under each residue.

Also writes `metrics_sim.json` (AUC/accuracy vs the true hidden state, plus r/MAE vs FB
and the majority baseline) so `compare_metrics.py` can read these numbers instead of
carrying a hardcoded copy.

Run:  python make_figures.py
"""

import json
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_bayesflow import load_chains, total_chains, predict_helix, load_model, NUM_SAMPLES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "figures", "validation_figure.png")
METRICS = os.path.join(HERE, "outputs", "metrics_sim.json")
N_SCATTER_CHAINS = 500   # the FULL held-out tail block (matches train/val split)


def main():
    approx = load_model()

    total = total_chains()
    val_idx = list(range(total - N_SCATTER_CHAINS, total))     # held-out tail block
    triples, _ = load_chains(val_idx)
    chains = [triples[i] for i in val_idx]

    # ---- BayesFlow mean vs exact FB (and vs the TRUE hidden state) -----------
    bf_all, fb_all, true_all = [], [], []
    example = None
    for obs, gamma, true in chains:
        mean, std = predict_helix(approx, obs, num_samples=NUM_SAMPLES)
        bf_all.append(mean); fb_all.append(gamma); true_all.append(true)
        if example is None and 45 <= len(obs) <= 85 and true.sum() >= 5:
            example = (obs, gamma, true, mean, std)
    bf_all = np.concatenate(bf_all); fb_all = np.concatenate(fb_all)
    true_all = np.concatenate(true_all)
    if example is None:                       # fallback to the first chain
        obs, gamma, true = chains[0]
        mean, std = predict_helix(approx, obs, num_samples=NUM_SAMPLES)
        example = (obs, gamma, true, mean, std)

    mae = float(np.abs(bf_all - fb_all).mean())
    corr = float(np.corrcoef(bf_all, fb_all)[0, 1])

    # ---- metrics vs ground truth (for compare_metrics.py) -------------------
    hf = float(true_all.mean())
    metrics = {
        "setting": "held_out_simulated",
        "n_chains": N_SCATTER_CHAINS, "n_residues": int(len(true_all)),
        "num_samples": NUM_SAMPLES,
        "auc_bf": float(roc_auc_score(true_all, bf_all)),
        "auc_fb": float(roc_auc_score(true_all, fb_all)),
        "acc_bf": float(((bf_all > 0.5).astype(int) == true_all).mean()),
        "acc_fb": float(((fb_all > 0.5).astype(int) == true_all).mean()),
        "baseline": float(max(hf, 1 - hf)),      # pooled majority-class accuracy
        "corr_bf_fb": corr, "mae_bf_fb": mae,
    }
    with open(METRICS, "w") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # Left: scatter
    ax[0].scatter(fb_all, bf_all, s=5, alpha=0.15, color="#1D9E75", edgecolors="none")
    ax[0].plot([0, 1], [0, 1], "--", color="0.4", lw=1)
    ax[0].set(xlabel="exact Forward-Backward  P(helix)",
              ylabel="BayesFlow posterior mean  P(helix)",
              xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
              title=f"Held-out residues (n={len(bf_all):,})\nMAE={mae:.3f}   r={corr:.3f}")
    ax[0].set_aspect("equal")

    # Right: example chain with uncertainty band
    obs, gamma, true, mean, std = example
    xs = np.arange(len(obs))
    ax[1].step(xs, true, where="mid", color="0.6", lw=1.2, label="true hidden state")
    ax[1].plot(xs, gamma, "-", color="black", lw=1.8, label="exact FB")
    ax[1].plot(xs, mean, "-", color="#1D9E75", lw=1.6, label="BayesFlow mean")
    ax[1].fill_between(xs, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                       color="#1D9E75", alpha=0.25, label="BayesFlow +/-1 std")
    ax[1].axhline(0.5, color="0.8", lw=0.8, ls=":")
    ax[1].set(xlabel="residue position", ylabel="P(helix)", ylim=(-0.05, 1.05),
              title=f"Example held-out chain (len {len(obs)})")
    ax[1].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"MAE={mae:.4f}  corr={corr:.4f}  residues={len(bf_all):,}")
    print(f"AUC vs true: BF={metrics['auc_bf']:.3f} FB={metrics['auc_fb']:.3f}  "
          f"acc: BF={metrics['acc_bf']:.3f} baseline={metrics['baseline']:.3f}")
    print(f"saved figure -> {OUT}")
    print(f"saved metrics -> {METRICS}")


if __name__ == "__main__":
    main()
