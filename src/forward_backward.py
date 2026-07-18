"""Exact per-residue helix posteriors via the Forward-Backward algorithm (hmmlearn).

Given the fixed two-state HMM (start / transition / emission tables defined in
`simulate.py`), this runs the Forward-Backward algorithm on amino-acid sequences
to obtain, for every residue, the posterior probability P(helix | whole chain).
These "gamma" targets are the exact Bayesian answer that a BayesFlow estimator
can later be trained to emulate.

We use `hmmlearn`'s `CategoricalHMM.predict_proba`, which is exactly the
Forward-Backward posterior of the hidden state at each position.

Usage
-----
    python forward_backward.py                       # all chains in simulated_chains.csv
    python forward_backward.py --limit 3000          # quick subset
    python forward_backward.py --input simulated_chains.csv --output fb_targets.npz

Outputs
-------
    <output>.npz  : lengths (int), p_helix (float32, concatenated per residue),
                    true_states (uint8, 1=helix/0=other, concatenated), chain_id.
                    Split back into chains with:
                        np.split(p_helix, np.cumsum(lengths)[:-1])
    fb_sample.csv : first few chains rendered residue-by-residue for eyeballing.
"""

import argparse
import csv
import os
import sys

import numpy as np
from hmmlearn.hmm import CategoricalHMM
from sklearn.metrics import roc_auc_score

# Reuse the exact model definition (single source of truth).
from simulate import AA, START_PROB, TRANS, EMIT, N_AA, N_STATES, H, O

AA_INDEX = {a: i for i, a in enumerate(AA)}

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(HERE, "simulated_chains.csv")
DEFAULT_OUTPUT = os.path.join(HERE, "fb_targets.npz")
DEFAULT_SAMPLE = os.path.join(HERE, "outputs", "fb_sample.csv")


def build_model():
    """A CategoricalHMM wired to our fixed start/transition/emission parameters.

    init_params='' and params='' tell hmmlearn NOT to initialise or re-estimate
    anything -- we are only doing inference (Forward-Backward), never training.
    """
    model = CategoricalHMM(n_components=N_STATES, init_params="", params="")
    model.startprob_ = START_PROB          # always starts in "other"
    model.transmat_ = TRANS                # 0.90/0.10, 0.05/0.95
    model.emissionprob_ = EMIT             # the two 20-residue tables
    model.n_features = N_AA
    return model


def encode(seq):
    """Map an amino-acid string to integer symbol indices (0..19)."""
    try:
        return np.fromiter((AA_INDEX[c] for c in seq), dtype=np.int64, count=len(seq))
    except KeyError as e:
        raise ValueError(
            f"sequence contains non-standard symbol {e!s}; only the 20 standard "
            f"amino acids {''.join(AA)} are supported (mask/drop others first)."
        )


def fb_posterior(seq, model):
    """Forward-Backward posterior P(helix) for each residue of `seq`.

    Returns
    -------
    p_helix : (L,) float array, posterior probability of the helix state.
    loglik  : float, log P(seq) under the model.
    """
    obs = encode(seq).reshape(-1, 1)
    loglik, posteriors = model.score_samples(obs)   # posteriors = gamma, shape (L, 2)
    return posteriors[:, H], loglik


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help="CSV with a 'seq' column (and optional 'states' truth)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="output .npz path")
    ap.add_argument("--sample-csv", default=DEFAULT_SAMPLE,
                    help="readable per-residue CSV of the first few chains")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N chains")
    ap.add_argument("--sample-n", type=int, default=5,
                    help="how many chains to write to the readable sample CSV")
    args = ap.parse_args(argv)

    if not os.path.exists(args.input):
        sys.exit(f"input CSV not found: {args.input}")

    model = build_model()

    lengths, chain_ids = [], []
    p_all, true_all = [], []
    sample_rows = []
    n = 0

    with open(args.input, newline="") as f:
        for row in csv.DictReader(f):
            if args.limit is not None and n >= args.limit:
                break
            seq = row["seq"]
            if not seq:
                continue

            p_helix, _ = fb_posterior(seq, model)

            states = row.get("states", "")
            true = np.array([1 if c == "H" else 0 for c in states], dtype=np.uint8) \
                if states else np.zeros(len(seq), dtype=np.uint8)

            lengths.append(len(seq))
            chain_ids.append(row.get("chain_id", str(n)))
            p_all.append(p_helix.astype(np.float32))
            true_all.append(true)

            if len(sample_rows) < args.sample_n:
                sample_rows.append((row.get("chain_id", str(n)), seq, states, p_helix))

            n += 1
            if n % 10_000 == 0:
                print(f"  ...{n:,} chains processed")

    if n == 0:
        sys.exit("no chains processed")

    lengths = np.array(lengths, dtype=np.int64)
    p_all = np.concatenate(p_all)
    true_all = np.concatenate(true_all)

    np.savez_compressed(args.output,
                        lengths=lengths,
                        chain_id=np.array(chain_ids),
                        p_helix=p_all,
                        true_states=true_all)

    # Readable per-residue sample.
    with open(args.sample_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain_id", "position", "residue", "true_state", "P_helix"])
        for cid, seq, states, p in sample_rows:
            for i, (aa, pv) in enumerate(zip(seq, p)):
                st = states[i] if i < len(states) else ""
                w.writerow([cid, i, aa, st, f"{pv:.4f}"])

    # Validation: does the exact posterior recover the KNOWN hidden states?
    have_truth = true_all.sum() > 0
    print(f"\ninput            : {args.input}")
    print(f"chains processed : {n:,}")
    print(f"total residues   : {len(p_all):,}")
    print(f"mean P(helix)    : {p_all.mean():.3f}  (simulator true helix frac "
          f"{true_all.mean():.3f})")
    print(f"saved targets    : {args.output}")
    print(f"saved sample     : {args.sample_csv}")
    if have_truth:
        acc = ((p_all > 0.5).astype(np.uint8) == true_all).mean()
        auc = roc_auc_score(true_all, p_all)
        print(f"\nForward-Backward vs TRUE hidden states (sanity check):")
        print(f"  accuracy @0.5 : {acc:.3f}")
        print(f"  AUC           : {auc:.3f}")


if __name__ == "__main__":
    main()
