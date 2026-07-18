"""Improved BayesFlow variant to attack the SBC calibration failure (audit items #1, #3).

Three changes vs the main model, all aimed at the diagnosed cause (a near-deterministic,
near-degenerate 3-D target fitted by a coupling flow):

  1. TARGET -> 1-D. Predict only the centre-residue logit gamma. This removes the
     0.89-0.96 inter-dimension correlation that a coupling flow handled poorly. (A coupling
     flow needs >=2 dims to split, which is *why* the original used 3-D; FlowMatching has no
     such constraint, so 1-D becomes possible.)
  2. INFERENCE NET -> FlowMatching (instead of CouplingFlow). Better suited to sharp /
     low-dimensional densities.
  3. SUMMARY NET -> a small Conv+GRU (TimeSeriesNetwork) over the (31 x 21) window, instead
     of flattening to 651 and using a plain MLP. This is the canonical BayesFlow component
     the main model omitted (audit #11 / Block C item 1); it also gives the window
     translation structure a proper inductive bias.

If SBC now passes, the degenerate-target hypothesis is confirmed and this becomes the
reportable "calibrated" model. If it still fails, the written near-deterministic argument
stands and item #3 (summary network) is still delivered.

Run:  python train_calibrated.py            # train + SBC
      python train_calibrated.py --smoke    # tiny end-to-end check
"""

import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_bayesflow import (load_chains, total_chains, windows_for_sequence,
                             _logit, _sigmoid, WINDOW, N_CH, SKIP_START)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_OUT = os.path.join(HERE, "bayesflow_calibrated.keras")  # overwritten per-run; window in log
VAR_NAME = [r"logit $\gamma$ (centre)"]


def build_data_1d(chains, max_windows, rng, skip_start=SKIP_START, W=WINDOW):
    """{'window': (N, WINDOW, N_CH), 'theta': (N, 1)} -- 1-D centre target, window kept 2-D."""
    per = max(1, max_windows // max(1, len(chains)))
    Xs, Ys = [], []
    for obs, gamma, _ in chains:
        L = len(obs)
        lo = min(skip_start, max(0, L - 1))
        avail = L - lo
        if avail <= 0:
            continue
        k = min(avail, per)
        idx = lo + rng.choice(avail, size=k, replace=False)
        Xs.append(windows_for_sequence(obs, W)[idx])            # (k, W, N_CH)
        Ys.append(gamma[idx][:, None])                          # (k, 1)
    X = np.concatenate(Xs, 0).astype(np.float32)
    y = np.concatenate(Ys, 0)
    if len(y) > max_windows:
        keep = rng.choice(len(y), size=max_windows, replace=False)
        X, y = X[keep], y[keep]
    return {"window": X, "theta": _logit(y).astype(np.float32)}


def build_approximator():
    import bayesflow as bf
    inference = bf.networks.FlowMatching(subnet_kwargs={"widths": (256, 256)})
    summary = bf.networks.TimeSeriesNetwork(summary_dim=32, filters=(32, 64),
                                            kernel_sizes=(5, 3))
    adapter = (bf.Adapter()
               .rename("theta", "inference_variables")
               .rename("window", "summary_variables"))
    approx = bf.approximators.ContinuousApproximator(
        inference_network=inference, summary_network=summary, adapter=adapter)
    return approx, adapter


def predict_centre(approx, obs, num_samples=50):
    win = windows_for_sequence(obs, WINDOW).astype(np.float32)   # (L, WINDOW, N_CH)
    draws = approx.sample(num_samples=num_samples, conditions={"window": win})["theta"]
    p = _sigmoid(np.asarray(draws)[..., 0])                      # (L, num_samples)
    return p.mean(1), p.std(1)


def main(argv=None):
    import bayesflow as bf
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-chains", type=int, default=15000)
    ap.add_argument("--val-chains", type=int, default=500)
    ap.add_argument("--max-windows", type=int, default=300000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        args.train_chains, args.val_chains, args.max_windows, args.epochs = 400, 60, 8000, 2

    rng = np.random.default_rng(args.seed)
    total = total_chains()
    train_idx = list(range(0, args.train_chains))
    val_idx = list(range(total - args.val_chains, total))
    assert set(train_idx).isdisjoint(val_idx)

    print(f"WINDOW = {args.window}")
    print(f"loading {args.train_chains} train + {args.val_chains} val chains ...")
    triples, _ = load_chains(train_idx + val_idx)
    train_chains = [triples[i] for i in train_idx]
    val_chains = [triples[i] for i in val_idx]

    print("building 1-D training data ...")
    train = build_data_1d(train_chains, args.max_windows, rng, W=args.window)
    print(f"windows: {train['window'].shape}, theta: {train['theta'].shape}")

    approx, adapter = build_approximator()
    approx.compile(optimizer="adam")
    dataset = bf.OfflineDataset(data=train, batch_size=args.batch_size, adapter=adapter)
    approx.fit(dataset=dataset, epochs=args.epochs, verbose=2)
    try:
        approx.save(MODEL_OUT); print(f"saved -> {MODEL_OUT}")
    except Exception as e:
        print("(save skipped:", e, ")")

    # ---- SBC on held-out ----------------------------------------------------
    print("building held-out diagnostic set ...")
    vd = build_data_1d(val_chains, 1500, np.random.default_rng(11), W=args.window)
    th = vd["theta"]
    draws = np.asarray(approx.sample(num_samples=200,
                                     conditions={"window": vd["window"]})["theta"], np.float32)
    fig = bf.diagnostics.calibration_ecdf(draws, th, variable_names=VAR_NAME, difference=True)
    fig.savefig(os.path.join(HERE, "figures", f"diag_sbc_w{args.window}.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    fig = bf.diagnostics.recovery(draws, th, variable_names=VAR_NAME)
    fig.savefig(os.path.join(HERE, "figures", f"diag_recovery_w{args.window}.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    pm, ps = draws.mean(1), draws.std(1)
    z = (pm - th) / np.maximum(ps, 1e-8)
    # crude SBC verdict: fraction of the rank-ECDF expected inside the band is proxied by |z|
    print("\n=== calibrated model diagnostics (held-out) ===")
    print(f"  centre: bias={float((pm-th).mean()):+.4f}  mean|z|={float(np.abs(z).mean()):.2f} "
          f"(calibrated ~0.80)  post.sd={float(ps.mean()):.4f}")
    print("  -> inspect diag_sbc_calibrated.png: rank ECDF inside the grey band = PASS")


if __name__ == "__main__":
    main()
