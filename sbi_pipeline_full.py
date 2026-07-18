"""SBI Protein-Based Inference -- THE ENTIRE PIPELINE IN ONE FILE.

Predict per-residue alpha-helix probability from an amino-acid sequence with a fixed
two-state HMM, amortized by BayesFlow. This single script inlines every stage that the
modular scripts (simulate.py, forward_backward.py, train_bayesflow.py, make_figures.py,
insulin_eval.py, eval_real.py, diagnostics.py) implement separately -- self-contained,
no local imports.

Stages (run in order):
    simulate  -> simulated_chains.csv       (run the fixed HMM)
    fb        -> fb_targets.npz              (exact Forward-Backward gamma, via hmmlearn)
    train     -> bayesflow_posterior.keras   (windowed CouplingFlow amortized posterior)
    eval      -> held-out sim + insulin(1MSO) + PISCES, with AUC/accuracy
    diag      -> SBC / loss / recovery (the SBI diagnostics)

Usage:
    python sbi_pipeline_full.py                 # DEMO scale, all stages (~a few minutes)
    python sbi_pipeline_full.py --full          # real scale (100k chains, 30 epochs; ~1 hour+)
    python sbi_pipeline_full.py --stages train,eval
    python sbi_pipeline_full.py --force         # recompute even if outputs exist

Requires: numpy, hmmlearn, scikit-learn, matplotlib, bayesflow>=2, keras, torch.
Ground-truth CSVs expected in ../archive/ (Kaggle protein-secondary-structure dataset).
"""

import argparse
import csv
import json
import os
os.environ.setdefault("KERAS_BACKEND", "torch")   # must precede keras/bayesflow import

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "..", "archive")
CSV_PATH = os.path.join(HERE, "simulated_chains.csv")
NPZ_PATH = os.path.join(HERE, "fb_targets.npz")
MODEL_PATH = os.path.join(HERE, "bayesflow_posterior.keras")
PISCES = os.path.join(ARCHIVE, "2018-06-06-pdb-intersect-pisces.csv")
SSCLEAN = os.path.join(ARCHIVE, "2018-06-06-ss.cleaned.csv")

# =============================================================================
# 0.  THE MODEL  (given, fixed -- not inferred)
# =============================================================================
AA = list("ARNDCEQGHILKMFPSTWYV")            # 20 amino acids, emission-table order
AA_INDEX = {a: i for i, a in enumerate(AA)}
N_AA = len(AA)
H, O = 0, 1                                  # hidden states: helix, other
N_STATES = 2
START_PROB = np.array([0.0, 1.0])            # always starts in "other"
TRANS = np.array([[0.90, 0.10],              # from helix
                  [0.05, 0.95]])             # from other
_HELIX = [12, 6, 3, 5, 1, 9, 5, 4, 2, 7, 12, 6, 3, 4, 2, 5, 4, 1, 3, 6]
_OTHER = [6, 5, 5, 6, 2, 5, 3, 9, 3, 5, 8, 6, 2, 4, 6, 7, 6, 1, 4, 7]
EMIT = np.array([_HELIX, _OTHER], dtype=float) / 100.0
assert np.allclose(EMIT.sum(1), 1) and np.allclose(TRANS.sum(1), 1)

# windowing / encoding
WINDOW = 31                                  # +/-15 residues
PAD_CHANNEL = N_AA
N_CH = N_AA + 1                              # 20 amino acids + 1 padding channel
_EPS = 1e-3
SKIP_START = 2                               # gamma_0 = 0 exactly -> drop the start atom
NUM_SAMPLES = 50                             # posterior draws per residue (see FINDINGS)

_P_TO_HELIX = TRANS[:, H]
_EMIT_CUM = np.cumsum(EMIT, axis=1)
_AA_ARR = np.array(AA)


def encode(seq):
    return np.fromiter((AA_INDEX[c] for c in seq), dtype=np.int64, count=len(seq))


def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# =============================================================================
# 1.  SIMULATOR
# =============================================================================
def simulate_chain(length, rng):
    """One (state_str, seq_str, n_helix) draw from the HMM."""
    if length <= 0:
        return "", "", 0
    states = np.empty(length, dtype=np.int64)
    s = int(rng.choice(N_STATES, p=START_PROB))
    states[0] = s
    if length > 1:
        u = rng.random(length - 1)
        for t in range(1, length):
            s = H if u[t - 1] < _P_TO_HELIX[s] else O
            states[t] = s
    obs = np.empty(length, dtype=np.int64)
    ue = rng.random(length)
    for st in (H, O):
        mask = states == st
        if mask.any():
            obs[mask] = np.searchsorted(_EMIT_CUM[st], ue[mask], side="right")
    obs = np.clip(obs, 0, N_AA - 1)
    return ("".join("H" if x == H else "O" for x in states),
            "".join(_AA_ARR[obs]), int((states == H).sum()))


def stage_simulate(n_chains, seed=0):
    banner(f"1 · SIMULATE  ({n_chains:,} chains)")
    rng = np.random.default_rng(seed)
    gt = []
    with open(PISCES, newline="") as f:
        for r in csv.DictReader(f):
            try:
                gt.append(int(r["len"]))
            except (KeyError, ValueError):
                pass
    lengths = rng.choice(np.array(gt), size=n_chains, replace=True)
    tot = helix = 0
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain_id", "len", "states", "seq"])
        for i, L in enumerate(lengths):
            st, seq, nh = simulate_chain(int(L), rng)
            w.writerow([i, int(L), st, seq])
            tot += int(L); helix += nh
    print(f"  wrote {CSV_PATH}  ({tot:,} residues, helix frac {helix/tot:.3f} vs theory 0.333)")


# =============================================================================
# 2.  FORWARD-BACKWARD  (exact per-residue gamma, via hmmlearn)
# =============================================================================
def build_hmm():
    from hmmlearn.hmm import CategoricalHMM
    m = CategoricalHMM(n_components=N_STATES, init_params="", params="")
    m.startprob_, m.transmat_, m.emissionprob_ = START_PROB, TRANS, EMIT
    m.n_features = N_AA
    return m


def fb_posterior(seq, model):
    obs = encode(seq).reshape(-1, 1)
    loglik, post = model.score_samples(obs)          # post = gamma, (L, 2)
    return post[:, H], loglik


def stage_fb():
    banner("2 · FORWARD-BACKWARD  (exact gamma targets)")
    model = build_hmm()
    lengths, p_all = [], []
    n = 0
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            p, _ = fb_posterior(row["seq"], model)
            lengths.append(len(row["seq"])); p_all.append(p.astype(np.float32))
            n += 1
            if n % 20000 == 0:
                print(f"  ...{n:,} chains")
    p_all = np.concatenate(p_all)
    np.savez_compressed(NPZ_PATH, lengths=np.array(lengths, dtype=np.int64), p_helix=p_all)
    print(f"  wrote {NPZ_PATH}  ({len(p_all):,} residues, mean P(helix) {p_all.mean():.3f})")


# =============================================================================
# 3.  WINDOWING + BAYESFLOW
# =============================================================================
def windows_for_sequence(obs, W=WINDOW):
    obs = np.asarray(obs); L = len(obs); half = W // 2
    padded = np.full(L + 2 * half, -1, dtype=np.int64)
    padded[half:half + L] = obs
    vals = padded[np.arange(L)[:, None] + np.arange(W)[None, :]]
    X = np.zeros((L, W, N_CH), dtype=np.float32)
    rl, rj = np.where(vals >= 0); X[rl, rj, vals[rl, rj]] = 1.0
    pl, pj = np.where(vals < 0); X[pl, pj, PAD_CHANNEL] = 1.0
    return X


def load_chains(indices):
    """{i: (obs, gamma, true)} for the given chain indices, from csv + npz (row-aligned)."""
    want = {int(i) for i in indices}; hi = max(want)
    d = np.load(NPZ_PATH); lengths = d["lengths"]; p = d["p_helix"]
    offs = np.concatenate([[0], np.cumsum(lengths)])
    out = {}
    with open(CSV_PATH, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i > hi:
                break
            if i in want:
                L = int(row["len"]); obs = encode(row["seq"])
                assert len(obs) == L == int(lengths[i]), f"csv/npz misalign at {i}"
                gamma = p[offs[i]:offs[i] + L].astype(np.float64)
                true = (np.frombuffer(row["states"].encode(), np.uint8) == ord("H")).astype(int)
                out[i] = (obs, gamma, true)
    return out


def total_chains():
    with np.load(NPZ_PATH) as d:
        return int(len(d["lengths"]))


def build_training_data(chains, max_windows, rng):
    per = max(1, max_windows // max(1, len(chains)))
    Xs, Ys = [], []
    for obs, gamma, _ in chains:
        L = len(obs); lo = min(SKIP_START, max(0, L - 1)); avail = L - lo
        if avail <= 0:
            continue
        idx = lo + rng.choice(avail, size=min(avail, per), replace=False)
        Xs.append(windows_for_sequence(obs)[idx])
        g_prev = np.concatenate([gamma[:1], gamma[:-1]])
        g_next = np.concatenate([gamma[1:], gamma[-1:]])
        Ys.append(np.stack([g_prev, gamma, g_next], 1)[idx])
    X = np.concatenate(Xs, 0); y = np.concatenate(Ys, 0)
    if len(y) > max_windows:
        k = rng.choice(len(y), max_windows, replace=False); X, y = X[k], y[k]
    return {"cond": X.reshape(len(X), -1).astype(np.float32), "theta": _logit(y).astype(np.float32)}


def build_approximator():
    import bayesflow as bf
    net = bf.networks.CouplingFlow(depth=6, subnet_kwargs={"widths": (256, 256)})
    adapter = (bf.Adapter().rename("cond", "inference_conditions")
               .rename("theta", "inference_variables"))
    return bf.approximators.ContinuousApproximator(inference_network=net, adapter=adapter), adapter


def load_model(path=MODEL_PATH):
    import bayesflow as bf  # noqa: F401 (registers serializable classes)
    import keras
    return keras.saving.load_model(path)


def predict_helix(approx, obs, num_samples=NUM_SAMPLES):
    cond = windows_for_sequence(obs).reshape(len(obs), -1).astype(np.float32)
    draws = approx.sample(num_samples=num_samples, conditions={"cond": cond})["theta"]
    p = _sigmoid(np.asarray(draws)[..., 1])
    return p.mean(1), p.std(1)


def stage_train(train_chains, max_windows, epochs, seed=7):
    banner(f"3 · TRAIN BAYESFLOW  ({train_chains:,} chains, {max_windows:,} windows, {epochs} epochs)")
    import bayesflow as bf
    rng = np.random.default_rng(seed)
    idx = list(range(0, train_chains))
    chains = [load_chains(idx)[i] for i in idx]
    data = build_training_data(chains, max_windows, rng)
    print(f"  training windows: cond {data['cond'].shape}, theta {data['theta'].shape}")
    approx, adapter = build_approximator()
    approx.compile(optimizer="adam")
    ds = bf.OfflineDataset(data=data, batch_size=512, adapter=adapter)
    approx.fit(dataset=ds, epochs=epochs, verbose=2)
    approx.save(MODEL_PATH)
    print(f"  saved {MODEL_PATH}")


# =============================================================================
# 4.  EVALUATION
# =============================================================================
def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if 0 < np.sum(y) < len(y) else float("nan")


def stage_eval(val_chains, pisces_limit):
    banner("4 · EVALUATE")
    from sklearn.metrics import roc_auc_score
    approx = load_model(); fb = build_hmm()
    total = total_chains()

    # (a) held-out simulated tail block: BayesFlow vs exact FB vs true state
    vi = list(range(total - val_chains, total))
    chains = [load_chains(vi)[i] for i in vi]
    bf_all, fb_all, tr_all = [], [], []
    for obs, gamma, true in chains:
        m, _ = predict_helix(approx, obs)
        bf_all.append(m); fb_all.append(gamma); tr_all.append(true)
    bf_all = np.concatenate(bf_all); fb_all = np.concatenate(fb_all); tr_all = np.concatenate(tr_all)
    print(f"  [held-out sim]  BayesFlow vs FB: r={np.corrcoef(bf_all, fb_all)[0,1]:.3f} "
          f"MAE={np.abs(bf_all-fb_all).mean():.3f} | AUC vs true: "
          f"BF={roc_auc_score(tr_all, bf_all):.3f} FB={roc_auc_score(tr_all, fb_all):.3f}")

    # (b) wild-type insulin 1MSO
    print("  [insulin 1MSO]")
    for pid, want in [("1MSO", True)]:
        with open(SSCLEAN, newline="") as f:
            for row in csv.DictReader(f):
                if row["pdb_id"] != "1MSO" or row.get("has_nonstd_aa") == "True":
                    continue
                seq, sst8 = row["seq"], row["sst8"]
                true = np.array([1 if c == "H" else 0 for c in sst8])
                bfp, _ = predict_helix(approx, encode(seq))
                print(f"    chain {row['chain_code']} (len {len(seq)}, helix {int(true.sum())}): "
                      f"AUC BF={_auc(true, bfp):.3f}")

    # (c) real PISCES proteins
    lim = "all" if pisces_limit == 0 else pisces_limit
    print(f"  [PISCES real proteins, limit={lim}]")
    aucs, n = [], 0
    std = set(AA)
    with open(PISCES, newline="") as f:
        for row in csv.DictReader(f):
            if pisces_limit and n >= pisces_limit:
                break
            seq = row["seq"]
            if row.get("has_nonstd_aa") == "True" or not set(seq) <= std or len(seq) < 20:
                continue
            true = np.array([1 if c == "H" else 0 for c in row["sst8"]])
            bfp, _ = predict_helix(approx, encode(seq))
            u = _auc(true, bfp)
            if not np.isnan(u):
                aucs.append(u)
            n += 1
            if n % 200 == 0:
                print(f"    ...{n} chains")
    print(f"    per-chain AUC {np.mean(aucs):.3f} over {len(aucs):,} chains "
          f"({(np.array(aucs) > 0.5).mean():.1%} beat chance)")


# =============================================================================
# 5.  SBI DIAGNOSTICS  (SBC / recovery)
# =============================================================================
def stage_diag(val_chains):
    banner("5 · SBI DIAGNOSTICS  (SBC + recovery)")
    import bayesflow as bf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    approx = load_model()
    total = total_chains()
    vi = list(range(total - val_chains, total))
    chains = [load_chains(vi)[i] for i in vi]
    data = build_training_data(chains, 1500, np.random.default_rng(11))
    th = data["theta"]
    draws = np.asarray(approx.sample(num_samples=200, conditions={"cond": data["cond"]})["theta"], np.float32)
    names = [r"logit $\gamma$ (prev)", r"logit $\gamma$ (centre)", r"logit $\gamma$ (next)"]
    for fn, out in [(bf.diagnostics.calibration_ecdf, "diag_sbc_full.png"),
                    (bf.diagnostics.recovery, "diag_recovery_full.png")]:
        try:
            fig = fn(draws, th, variable_names=names)
            fig.savefig(os.path.join(HERE, "figures", out), dpi=120, bbox_inches="tight"); plt.close(fig)
            print(f"  saved {out}")
        except Exception as e:
            print(f"  ({out} skipped: {e})")
    z = (draws.mean(1) - th) / np.maximum(draws.std(1), 1e-8)
    print(f"  mean|z| = {np.abs(z).mean():.2f} (calibrated ~0.80) -- SBC fails slightly; "
          f"see FINDINGS §21 (near-deterministic estimand).")


# =============================================================================
#  MAIN
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="real scale (slow) instead of demo")
    ap.add_argument("--stages", default="simulate,fb,train,eval,diag",
                    help="comma list of stages to run")
    ap.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    args = ap.parse_args(argv)

    if args.full:
        cfg = dict(n_sim=100_000, train_chains=15_000, max_windows=300_000, epochs=30,
                   val_chains=500, pisces_limit=0)
    else:                                    # DEMO -- fast end-to-end
        cfg = dict(n_sim=4_000, train_chains=3_000, max_windows=40_000, epochs=8,
                   val_chains=200, pisces_limit=300)
    stages = [s.strip() for s in args.stages.split(",")]
    print(f"config: {'FULL' if args.full else 'DEMO'}  {cfg}\nstages: {stages}")

    if "simulate" in stages and (args.force or not os.path.exists(CSV_PATH)):
        stage_simulate(cfg["n_sim"])
    elif "simulate" in stages:
        print(f"\n[skip simulate] {os.path.basename(CSV_PATH)} exists (use --force)")

    if "fb" in stages and (args.force or not os.path.exists(NPZ_PATH)):
        stage_fb()
    elif "fb" in stages:
        print(f"[skip fb] {os.path.basename(NPZ_PATH)} exists (use --force)")

    if "train" in stages and (args.force or not os.path.exists(MODEL_PATH)):
        stage_train(cfg["train_chains"], cfg["max_windows"], cfg["epochs"])
    elif "train" in stages:
        print(f"[skip train] {os.path.basename(MODEL_PATH)} exists (use --force)")

    if "eval" in stages:
        stage_eval(cfg["val_chains"], cfg["pisces_limit"])
    if "diag" in stages:
        stage_diag(cfg["val_chains"])
    banner("DONE")


if __name__ == "__main__":
    main()
