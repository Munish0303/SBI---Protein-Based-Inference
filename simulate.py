"""Two-state HMM protein simulator (alpha-helix vs other).

Self-contained: the transition and emission probabilities are baked in exactly
as specified. It simulates N chains (default 100,000) whose lengths are drawn
(with replacement) from the empirical length distribution of a ground-truth CSV,
so the synthetic set mirrors the real data's length profile.

Model
-----
States: 0 = alpha-helix ("H"), 1 = other ("O"). Every chain starts in "other".
Transitions   from helix : ->helix 0.90, ->other 0.10
              from other : ->helix 0.05, ->other 0.95
Emissions are the two 20-amino-acid tables provided (each row sums to 1).

Usage
-----
    python simulate.py                     # 100,000 chains -> simulated_chains.csv
    python simulate.py --n 5000 --seed 1   # smaller, different seed
"""

import argparse
import csv
import os
import sys

import numpy as np

# --- Amino-acid alphabet, in the exact column order of the emission tables ---
AA = list("ARNDCEQGHILKMFPSTWYV")            # 20 residues
N_AA = len(AA)
_AA_ARR = np.array(AA)

# --- Hidden states -----------------------------------------------------------
H, O = 0, 1                                  # helix, other
N_STATES = 2
STATE_CHAR = {H: "H", O: "O"}

# --- Start distribution: chains ALWAYS start in "other" ----------------------
START_PROB = np.array([0.0, 1.0])            # [helix, other]

# --- Transition matrix (row = current state, col = next state) ---------------
TRANS = np.array([
    [0.90, 0.10],                            # from helix
    [0.05, 0.95],                            # from other
])

# --- Emission tables (row = state, col = amino acid), as fractions -----------
_HELIX = [12, 6, 3, 5, 1, 9, 5, 4, 2, 7, 12, 6, 3, 4, 2, 5, 4, 1, 3, 6]
_OTHER = [6, 5, 5, 6, 2, 5, 3, 9, 3, 5, 8, 6, 2, 4, 6, 7, 6, 1, 4, 7]
EMIT = np.array([_HELIX, _OTHER], dtype=float) / 100.0

# --- Sanity checks on the transcribed numbers --------------------------------
assert EMIT.shape == (N_STATES, N_AA)
assert np.allclose(EMIT.sum(axis=1), 1.0), "emission rows must sum to 1"
assert np.allclose(TRANS.sum(axis=1), 1.0), "transition rows must sum to 1"
assert np.isclose(START_PROB.sum(), 1.0)

# Precomputed helpers for fast sampling.
_P_TO_HELIX = TRANS[:, H]                     # [P(helix->helix), P(other->helix)]
_EMIT_CUM = np.cumsum(EMIT, axis=1)           # (2, 20) cumulative emission

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "..", "archive",
                             "2018-06-06-pdb-intersect-pisces.csv")
DEFAULT_OUTPUT = os.path.join(HERE, "simulated_chains.csv")


def load_lengths(path):
    """Return a numpy array of chain lengths from the ground-truth CSV."""
    lens = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                lens.append(int(row["len"]))
            except (KeyError, ValueError):
                pass
    if not lens:
        sys.exit(f"no lengths found in {path}")
    return np.array(lens, dtype=np.int64)


def simulate_chain(length, rng):
    """Draw one (state_str, seq_str, n_helix) from the generative model."""
    if length <= 0:
        return "", "", 0

    # Hidden state path (sequential, first-order Markov; starts in "other").
    states = np.empty(length, dtype=np.int64)
    s = int(rng.choice(N_STATES, p=START_PROB))
    states[0] = s
    if length > 1:
        u = rng.random(length - 1)
        for t in range(1, length):
            s = H if u[t - 1] < _P_TO_HELIX[s] else O
            states[t] = s

    # Emissions (vectorised inverse-transform sampling per state).
    obs = np.empty(length, dtype=np.int64)
    ue = rng.random(length)
    for st in (H, O):
        mask = states == st
        if mask.any():
            obs[mask] = np.searchsorted(_EMIT_CUM[st], ue[mask], side="right")
    obs = np.clip(obs, 0, N_AA - 1)

    state_str = "".join(STATE_CHAR[int(x)] for x in states)
    seq_str = "".join(_AA_ARR[obs])
    return state_str, seq_str, int((states == H).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100_000,
                    help="number of chains to simulate (default 100000)")
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help="ground-truth CSV to draw chain lengths from")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="output CSV path")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = ap.parse_args(argv)

    if not os.path.exists(args.input):
        sys.exit(f"input CSV not found: {args.input}")

    rng = np.random.default_rng(args.seed)

    gt_lengths = load_lengths(args.input)
    # Match the real length profile: sample lengths with replacement.
    lengths = rng.choice(gt_lengths, size=args.n, replace=True)

    total_res = 0
    sim_helix = 0
    with open(args.output, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["chain_id", "len", "states", "seq"])
        for i, L in enumerate(lengths):
            L = int(L)
            state_str, seq_str, n_h = simulate_chain(L, rng)
            writer.writerow([i, L, state_str, seq_str])
            total_res += L
            sim_helix += n_h

    print(f"input (lengths from): {args.input}")
    print(f"output              : {args.output}")
    print(f"chains simulated    : {args.n:,}")
    print(f"total residues      : {total_res:,}")
    print(f"length  min/max     : {lengths.min()} / {lengths.max()}")
    print(f"length  mean/median : {lengths.mean():.1f} / {int(np.median(lengths))}")
    print(f"sim helix fraction  : {sim_helix / total_res:.3f} "
          f"(model stationary ~ {0.05 / (0.05 + 0.10):.3f})")


if __name__ == "__main__":
    main()
