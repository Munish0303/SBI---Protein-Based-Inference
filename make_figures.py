"""Validation figures for the trained BayesFlow posterior.

Produces `validation_figure.png` with two panels:
  (left)  BayesFlow posterior mean vs exact Forward-Backward P(helix), per residue,
          over held-out simulated chains (the disjoint tail block, never trained on).
  (right) one example chain: true hidden state, exact FB, BayesFlow mean, and the
          BayesFlow +/-1 std uncertainty band drawn under each residue.

Run:  python make_figures.py
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_bayesflow import load_chains, total_chains, predict_helix

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "bayesflow_posterior.keras")
OUT = os.path.join(HERE, "validation_figure.png")
N_SCATTER_CHAINS = 80
NUM_SAMPLES = 300


def load_approx(path=MODEL_PATH):
    import bayesflow as bf          # registers serializable classes
    import keras
    return keras.saving.load_model(path)


def main():
    approx = load_approx()

    total = total_chains()
    val_idx = list(range(total - N_SCATTER_CHAINS, total))     # held-out tail block
    triples, _ = load_chains(val_idx)
    chains = [triples[i] for i in val_idx]

    # ---- Panel data: BayesFlow mean vs exact FB over all residues -----------
    bf_all, fb_all = [], []
    example = None
    for obs, gamma, true in chains:
        mean, std = predict_helix(approx, obs, num_samples=NUM_SAMPLES)
        bf_all.append(mean); fb_all.append(gamma)
        # pick a readable example: medium length with a real helix run
        if example is None and 45 <= len(obs) <= 85 and true.sum() >= 5:
            example = (obs, gamma, true, mean, std)
    bf_all = np.concatenate(bf_all); fb_all = np.concatenate(fb_all)
    if example is None:                       # fallback to the first chain
        obs, gamma, true = chains[0]
        mean, std = predict_helix(approx, obs, num_samples=NUM_SAMPLES)
        example = (obs, gamma, true, mean, std)

    mae = np.abs(bf_all - fb_all).mean()
    corr = np.corrcoef(bf_all, fb_all)[0, 1]

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
    print(f"saved figure -> {OUT}")


if __name__ == "__main__":
    main()
