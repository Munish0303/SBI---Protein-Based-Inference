"""NEW FINDING: the accuracy <-> calibration Pareto frontier (window-size sweep).

Our W=31 vs W=7 comparison showed calibration is controllable by window size. This
generalises it to a curve: for several window sizes we measure BOTH the point accuracy
(correlation of the posterior mean with the true gamma) AND the calibration quality
(a scalar SBC deviation), on held-out chains. The result is a frontier: you cannot have
a near-perfect point estimate AND a calibrated posterior at once, because a larger window
makes the estimand near-deterministic.

Model at each window: FlowMatching + Conv/GRU summary net + 1-D centre target (reused from
train_calibrated). Writes metrics_calib_w{W}.json for each; plot with the companion figure.

Run:  python experiment_pareto.py
"""

import json
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from train_bayesflow import load_chains, total_chains, _sigmoid
from train_calibrated import build_data_1d, build_approximator

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = [7, 11, 15, 21, 31]
TRAIN_CHAINS = 8000
MAX_WINDOWS = 150_000
EPOCHS = 15
VAL_CHAINS = 400
DRAWS = 200


def ks_deviation(draws_1d, true_1d):
    """Max |rank-ECDF - uniform| over the SBC fractional ranks (0 = perfectly calibrated)."""
    # rank of each true value among its posterior draws
    ranks = (draws_1d < true_1d[:, None]).mean(axis=1)          # (N,)
    r = np.sort(ranks)
    ecdf = np.arange(1, len(r) + 1) / len(r)
    return float(np.max(np.abs(ecdf - r)))


def main():
    import bayesflow as bf
    total = total_chains()
    train_idx = list(range(0, TRAIN_CHAINS))
    val_idx = list(range(total - VAL_CHAINS, total))
    triples, _ = load_chains(train_idx + val_idx)
    train_chains = [triples[i] for i in train_idx]
    val_chains = [triples[i] for i in val_idx]

    results = []
    for W in WINDOWS:
        print(f"\n===== WINDOW {W} =====")
        rng = np.random.default_rng(7)
        train = build_data_1d(train_chains, MAX_WINDOWS, rng, W=W)
        approx, adapter = build_approximator()
        approx.compile(optimizer="adam")
        ds = bf.OfflineDataset(data=train, batch_size=512, adapter=adapter)
        approx.fit(dataset=ds, epochs=EPOCHS, verbose=2)

        vd = build_data_1d(val_chains, 1500, np.random.default_rng(11), W=W)
        th = vd["theta"][:, 0]                                   # logit gamma (centre)
        draws = np.asarray(approx.sample(num_samples=DRAWS,
                                         conditions={"window": vd["window"]})["theta"], np.float32)[..., 0]
        p_true = _sigmoid(th)
        p_est = _sigmoid(draws).mean(1)
        r = float(np.corrcoef(p_est, p_true)[0, 1])
        mae = float(np.abs(p_est - p_true).mean())
        ks = ks_deviation(draws, th)
        sd = float(draws.std(1).mean())
        rec = {"window": W, "r_prob": r, "mae_prob": mae, "sbc_ks": ks, "post_sd_logit": sd}
        results.append(rec)
        json.dump(rec, open(os.path.join(HERE, "outputs", f"metrics_calib_w{W}.json"), "w"), indent=2)
        print(f"  W={W}: r={r:.3f}  MAE={mae:.4f}  SBC-KS={ks:.3f}  post.sd={sd:.3f}")

    json.dump(results, open(os.path.join(HERE, "outputs", "metrics_pareto.json"), "w"), indent=2)
    print("\n=== PARETO SUMMARY (accuracy vs calibration) ===")
    print(f"{'W':>4}{'r(prob)':>9}{'MAE':>8}{'SBC-KS':>9}{'post.sd':>9}")
    for x in results:
        print(f"{x['window']:>4}{x['r_prob']:>9.3f}{x['mae_prob']:>8.4f}{x['sbc_ks']:>9.3f}{x['post_sd_logit']:>9.3f}")
    print("saved -> metrics_pareto.json")


if __name__ == "__main__":
    main()
