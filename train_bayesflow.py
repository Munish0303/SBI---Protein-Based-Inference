"""Amortized neural posterior for per-residue alpha-helix probability (BayesFlow).

Goal
----
Learn a neural network that maps an amino-acid sequence to the per-residue helix
posterior P(helix), emulating the exact Forward-Backward answer -- but instantly,
without running the HMM. This is the "fast robot copies the slow detective" step.

The modeling problem and our solution
-------------------------------------
A normalizing flow needs a FIXED-dimension target, but chains (and their gamma
vectors) have variable length. So we do NOT model the whole chain at once.
Instead we amortize the posterior of a SINGLE residue, conditioned on a
fixed-size local window of sequence around it:

    inference variables : theta = logit P(helix) at [prev, centre, next] residue   (3-D)
    inference conditions : a flattened W-residue one-hot window centred on it

Because the window is only a LOCAL view, residues outside it still influence the
true gamma -- so window -> gamma is genuinely stochastic and the posterior is
non-degenerate. That is exactly what BayesFlow is built to capture. At prediction
time we slide the window along a chain and read off the centre residue.

Targets come from `fb_targets.npz` (exact Forward-Backward, already computed), so
BayesFlow is trained to reproduce the exact Bayesian posterior.

Run
---
    python train_bayesflow.py                       # default train + validate
    python train_bayesflow.py --train-chains 30000 --max-windows 600000 --epochs 40
    python train_bayesflow.py --smoke               # tiny fast end-to-end check

Train uses a FRONT block of chains, validation a disjoint TAIL block, so the two
never overlap however large training grows (verified at runtime + printed).
Memory note: the flattened one-hot windows are float32, so RAM ~ max_windows x
651 x 4 bytes (300k windows ~ 0.78 GB, 600k ~ 1.6 GB).
"""

import argparse
import csv
import os
os.environ.setdefault("KERAS_BACKEND", "torch")   # must precede keras/bayesflow import

import numpy as np

from simulate import AA, N_AA, H, O
from forward_backward import encode

# --- One-hot encoding of a window --------------------------------------------
PAD_CHANNEL = N_AA          # channel 20 marks "outside the chain" (padding)
N_CH = N_AA + 1             # 20 amino acids + 1 padding channel = 21
WINDOW = 31                 # +/-15 residues: comfortably spans a helix + context
_EPS = 1e-3                 # keeps logit finite when gamma hits 0 or 1

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "simulated_chains.csv")
NPZ_PATH = os.path.join(HERE, "fb_targets.npz")


def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def windows_for_sequence(obs, W=WINDOW):
    """(L,) amino-acid indices -> (L, W, N_CH) one-hot windows centred on each residue."""
    obs = np.asarray(obs)
    L = len(obs)
    half = W // 2
    padded = np.full(L + 2 * half, -1, dtype=np.int64)      # -1 = padding
    padded[half:half + L] = obs
    pos = np.arange(L)[:, None] + np.arange(W)[None, :]      # (L, W)
    vals = padded[pos]
    X = np.zeros((L, W, N_CH), dtype=np.float32)
    real = vals >= 0
    rl, rj = np.where(real)
    X[rl, rj, vals[rl, rj]] = 1.0                            # amino-acid channels
    pl, pj = np.where(~real)
    X[pl, pj, PAD_CHANNEL] = 1.0                             # padding channel
    return X


def total_chains(npz_path=NPZ_PATH):
    """Number of chains available in the Forward-Backward targets."""
    with np.load(npz_path) as d:
        return int(len(d["lengths"]))


def load_chains(indices, csv_path=CSV_PATH, npz_path=NPZ_PATH):
    """Load specific chain indices as ({i: (obs, gamma, true)}, {i: seq}).

    CSV rows and npz entries share the same order (both come from the same file),
    so chain i in the CSV aligns with slice i of the concatenated FB posteriors.
    Loading by an explicit index set lets train and validation come from DISJOINT
    blocks of the file (front block vs tail block) with no possibility of overlap.
    """
    want = {int(i) for i in indices}
    hi = max(want)
    d = np.load(npz_path)
    lengths = d["lengths"]
    p_helix = d["p_helix"]
    offs = np.concatenate([[0], np.cumsum(lengths)])
    triples, seqs = {}, {}
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i > hi:
                break
            if i in want:
                L = int(row["len"])
                obs = encode(row["seq"])
                gamma = p_helix[offs[i]:offs[i] + L].astype(np.float64)
                true = (np.frombuffer(row["states"].encode(), np.uint8) == ord("H")).astype(int)
                triples[i] = (obs, gamma, true)
                seqs[i] = row["seq"]
    return triples, seqs


def build_training_data(train_chains, max_windows, rng, W=WINDOW):
    """Collect (window, logit-gamma) pairs, sampling positions to bound memory.

    We sample a handful of positions per chain rather than materialising windows
    for all ~24M residues -- this keeps many DISTINCT chains represented while
    capping the training set at `max_windows`.
    """
    per_chain = max(1, max_windows // max(1, len(train_chains)))
    Xs, Ys = [], []
    for obs, gamma, _ in train_chains:
        L = len(obs)
        k = min(L, per_chain)
        idx = rng.choice(L, size=k, replace=False)
        Xw = windows_for_sequence(obs, W)[idx]              # (k, W, N_CH)
        g_prev = np.concatenate([gamma[:1], gamma[:-1]])
        g_next = np.concatenate([gamma[1:], gamma[-1:]])
        y = np.stack([g_prev, gamma, g_next], axis=1)[idx]  # (k, 3)
        Xs.append(Xw)
        Ys.append(y)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(Ys, axis=0)
    if len(y) > max_windows:
        keep = rng.choice(len(y), size=max_windows, replace=False)
        X, y = X[keep], y[keep]
    cond = X.reshape(len(X), -1).astype(np.float32)         # flatten window
    theta = _logit(y).astype(np.float32)                    # (N, 3)
    return {"cond": cond, "theta": theta}


def build_approximator(depth=6, widths=(256, 256)):
    """ContinuousApproximator with a CouplingFlow posterior over theta.

    The flattened one-hot window is fed directly as inference_conditions: an MLP
    over the fixed window preserves position and avoids a slow recurrent summary
    net. A coupling flow needs >=2 target dims to split, which is also why theta
    is 3-D (prev/centre/next) rather than a lone scalar.
    """
    import bayesflow as bf
    inference = bf.networks.CouplingFlow(depth=depth, subnet_kwargs={"widths": widths})
    adapter = (bf.Adapter()
               .rename("cond", "inference_conditions")
               .rename("theta", "inference_variables"))
    approx = bf.approximators.ContinuousApproximator(
        inference_network=inference, adapter=adapter)
    return approx, adapter


def predict_helix(approx, obs, num_samples=300, W=WINDOW):
    """Per-residue posterior mean and std of P(helix) for one chain."""
    cond = windows_for_sequence(obs, W).reshape(len(obs), -1).astype(np.float32)
    draws = approx.sample(num_samples=num_samples, conditions={"cond": cond})["theta"]
    p = _sigmoid(draws[..., 1])                             # centre residue, (L, num_samples)
    return p.mean(axis=1), p.std(axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-chains", type=int, default=15_000)
    ap.add_argument("--val-chains", type=int, default=500)
    ap.add_argument("--max-windows", type=int, default=300_000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save", default=os.path.join(HERE, "bayesflow_posterior.keras"))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny settings for a fast end-to-end sanity run")
    args = ap.parse_args(argv)

    if args.smoke:
        args.train_chains, args.val_chains = 300, 30
        args.max_windows, args.epochs = 5000, 2

    rng = np.random.default_rng(args.seed)

    # ---- Disjoint train/val split by chain index (no leakage) ---------------
    total = total_chains()
    if args.train_chains + args.val_chains > total:
        raise SystemExit(
            f"train_chains + val_chains ({args.train_chains}+{args.val_chains}) "
            f"exceeds available chains ({total})")
    train_idx = list(range(0, args.train_chains))                 # front block
    val_idx = list(range(total - args.val_chains, total))         # tail block
    assert set(train_idx).isdisjoint(val_idx), "train/val indices overlap -> leakage!"
    assert max(train_idx) < min(val_idx)

    print(f"loading {args.train_chains} train + {args.val_chains} val chains ...")
    triples, seqs = load_chains(train_idx + val_idx)
    train_chains = [triples[i] for i in train_idx]
    val_chains = [triples[i] for i in val_idx]

    # Explicit, visible leakage check: disjoint indices AND no shared sequence.
    train_seqset = {seqs[i] for i in train_idx}
    seq_overlap = sum(seqs[i] in train_seqset for i in val_idx)
    assert seq_overlap == 0, f"{seq_overlap} validation sequences also appear in train!"
    print(f"no-leakage: train idx [0,{args.train_chains}) | val idx "
          f"[{total - args.val_chains},{total}) | disjoint OK | "
          f"train/val sequence overlap = {seq_overlap} OK")

    print("building training windows ...")
    train = build_training_data(train_chains, args.max_windows, rng)
    print(f"training pairs: cond={train['cond'].shape}, theta={train['theta'].shape}")
    print(f"target helix-prob range: "
          f"[{_sigmoid(train['theta']).min():.3f}, {_sigmoid(train['theta']).max():.3f}]")

    import bayesflow as bf
    approx, adapter = build_approximator()
    approx.compile(optimizer="adam")
    dataset = bf.OfflineDataset(data=train, batch_size=args.batch_size, adapter=adapter)
    approx.fit(dataset=dataset, epochs=args.epochs, verbose=2)

    # ---- Validation on held-out chains: BayesFlow vs exact FB vs true state ----
    from sklearn.metrics import roc_auc_score
    bf_all, fb_all, true_all, std_all = [], [], [], []
    for obs, gamma, true in val_chains:
        mean, std = predict_helix(approx, obs, num_samples=args.num_samples)
        bf_all.append(mean); fb_all.append(gamma)
        true_all.append(true); std_all.append(std)
    bf_all = np.concatenate(bf_all); fb_all = np.concatenate(fb_all)
    true_all = np.concatenate(true_all); std_all = np.concatenate(std_all)

    mae = np.abs(bf_all - fb_all).mean()
    corr = np.corrcoef(bf_all, fb_all)[0, 1]
    print("\n=== validation (held-out chains) ===")
    print(f"BayesFlow vs exact FB : MAE={mae:.4f}  corr={corr:.4f}")
    print(f"mean posterior std     : {std_all.mean():.4f}")
    print(f"AUC vs TRUE state      : FB={roc_auc_score(true_all, fb_all):.3f}  "
          f"BayesFlow={roc_auc_score(true_all, bf_all):.3f}")
    print(f"acc@0.5 vs TRUE state  : FB={((fb_all>.5).astype(int)==true_all).mean():.3f}  "
          f"BayesFlow={((bf_all>.5).astype(int)==true_all).mean():.3f}")

    try:
        approx.save(args.save)
        print(f"saved model -> {args.save}")
    except Exception as e:
        print(f"(model save skipped: {e})")


if __name__ == "__main__":
    main()
