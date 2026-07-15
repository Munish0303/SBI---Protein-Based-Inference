"""SBI diagnostics for the trained BayesFlow amortized posterior.

The predictive metrics elsewhere (AUC, accuracy, correlation with FB) say whether the
point estimate is *useful*. These diagnostics say whether the *posterior itself* is
honest -- which is what a simulation-based-inference workflow actually requires:

  1. Convergence      -- training loss curve (from the actual run that produced the model)
  2. Calibration      -- simulation-based calibration (SBC) rank ECDF
  3. Recovery         -- estimated vs true theta, with uncertainty
  4. Contraction      -- posterior contraction / z-scores

What "SBC" means here
---------------------
The amortized target is theta = logit gamma at [prev, centre, next], where gamma is the
exact Forward-Backward posterior of the WHOLE chain; the condition x is the local
31-residue window. Simulating a chain and picking a position draws an exact sample from
the joint p(theta, x). SBC then asks: is the rank of theta_true among draws from
q(theta | x) uniform? If yes, the learned posterior is calibrated -- i.e. the +/-1 std
band is honest rather than decorative.

All diagnostic windows come from the HELD-OUT tail block of chains (never trained on).

Run:  python diagnostics.py
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bayesflow as bf
import keras

from train_bayesflow import (load_chains, total_chains, build_training_data,
                             build_approximator, _sigmoid)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "bayesflow_posterior.keras")
TRAIN_LOG = os.path.join(HERE, "train_bf2.log")

N_DIAG_CHAINS = 500        # held-out tail block (same block used for validation)
N_WINDOWS = 2000           # diagnostic datasets
N_DRAWS = 250              # posterior draws per dataset
VAR_NAMES = [r"logit $\gamma$ (prev)", r"logit $\gamma$ (centre)", r"logit $\gamma$ (next)"]
TEAL, DEEP = "#1D9E75", "#0F6E56"


def held_out_diag_set(seed=11):
    """(cond, theta_true) pairs drawn from the held-out TAIL block of chains."""
    total = total_chains()
    val_idx = list(range(total - N_DIAG_CHAINS, total))
    triples, _ = load_chains(val_idx)
    chains = [triples[i] for i in val_idx]
    rng = np.random.default_rng(seed)
    data = build_training_data(chains, N_WINDOWS, rng)   # {"cond": (N,651), "theta": (N,3)}
    print(f"diagnostic set: {data['theta'].shape[0]} windows from {len(chains)} held-out chains")
    return data


def parse_loss_curve(path=TRAIN_LOG):
    """Per-epoch training loss from the log of the run that produced the model."""
    if not os.path.exists(path):
        return []
    losses = []
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.search(r"-\s*loss:\s*(-?[\d.]+e?[-+]?\d*)", line)
            if m:
                try:
                    losses.append(float(m.group(1)))
                except ValueError:
                    pass
    return losses


def main():
    approx = keras.saving.load_model(MODEL_PATH)
    data = held_out_diag_set()
    targets = np.asarray(data["theta"], dtype=np.float32)                 # (N, 3)

    print("drawing posterior samples ...")
    draws = approx.sample(num_samples=N_DRAWS,
                          conditions={"cond": data["cond"]})["theta"]     # (N, N_DRAWS, 3)
    draws = np.asarray(draws, dtype=np.float32)
    print("estimates:", draws.shape, " targets:", targets.shape)

    # ---- 1. Convergence: training loss curve -------------------------------
    losses = parse_loss_curve()
    if losses:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(range(1, len(losses) + 1), losses, "-o", ms=3, color=DEEP, lw=1.8)
        ax.set(xlabel="epoch", ylabel="training loss (negative log-density)",
               title=f"Convergence — {len(losses)} epochs")
        ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig("diag_loss.png", dpi=130); plt.close(fig)
        print(f"loss: first={losses[0]:.3f}  last={losses[-1]:.3f}  -> diag_loss.png")
    else:
        print("(no training log found — skipping loss curve)")

    # ---- 2. Calibration: SBC rank ECDF -------------------------------------
    fig = bf.diagnostics.calibration_ecdf(draws, targets, variable_names=VAR_NAMES,
                                          difference=True, rank_type="fractional")
    fig.savefig("diag_sbc_ecdf.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved -> diag_sbc_ecdf.png  (ranks inside the band = calibrated)")

    # ---- 3. Recovery: theta_true vs estimate -------------------------------
    fig = bf.diagnostics.recovery(draws, targets, variable_names=VAR_NAMES)
    fig.savefig("diag_recovery.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved -> diag_recovery.png")

    # ---- 4. Posterior contraction / z-scores -------------------------------
    fig = bf.diagnostics.z_score_contraction(draws, targets, variable_names=VAR_NAMES)
    fig.savefig("diag_contraction.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved -> diag_contraction.png")

    # ---- numeric summaries --------------------------------------------------
    post_mean = draws.mean(axis=1)                       # (N, 3)
    post_sd = draws.std(axis=1)                          # (N, 3)
    prior_sd = targets.std(axis=0)                       # (3,)  induced "prior" spread
    contraction = 1.0 - (post_sd.mean(axis=0) ** 2) / (prior_sd ** 2)
    z = (post_mean - targets) / np.maximum(post_sd, 1e-8)

    print("\n=== numeric diagnostics (held-out) ===")
    for i, n in enumerate(["prev", "centre", "next"]):
        print(f"  {n:>6}: contraction={contraction[i]:.3f}   "
              f"mean|z|={np.abs(z[:, i]).mean():.2f}   "
              f"posterior sd={post_sd[:, i].mean():.3f}")

    # calibration error (deviation of rank ECDF from uniform)
    try:
        ce = bf.diagnostics.calibration_error(draws, targets)
        print("  calibration error:", {k: np.round(np.asarray(v), 4).tolist()
                                       for k, v in ce.items()} if isinstance(ce, dict)
              else np.round(np.asarray(ce), 4).tolist())
    except Exception as e:
        print("  (calibration_error unavailable:", e, ")")

    # on the probability scale (what we actually report)
    p_true = _sigmoid(targets[:, 1])
    p_mean = _sigmoid(draws[:, :, 1]).mean(axis=1)
    print(f"\n  centre residue on probability scale: "
          f"MAE={np.abs(p_mean - p_true).mean():.4f}  "
          f"corr={np.corrcoef(p_mean, p_true)[0,1]:.4f}")


if __name__ == "__main__":
    main()
