"""Block B: are the GIVEN emission tables realistic? (empirical check against real data)

We already verified the transition probabilities against real DSSP annotations. This does
the same for the *emission* tables: measure the empirical P(amino acid | helix) and
P(amino acid | other) from the real PISCES chains -- using the strict H-only helix
definition -- and compare them to the tables the model was handed.

Run:  python emission_check.py   ->  emission_check.png + printed table
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate import AA, EMIT, N_AA

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PISCES = os.path.join(HERE, "..", "archive", "2018-06-06-pdb-intersect-pisces.csv")
OUT = os.path.join(HERE, "figures", "emission_check.png")
STD = set(AA)
TEAL, GRAY, DEEP = "#1D9E75", "#B8B5AA", "#0F6E56"


def empirical_emissions(path=PISCES):
    """Empirical P(aa | helix) and P(aa | other) from real sst8 (H-only helix)."""
    lut = np.full(128, -1, dtype=np.int64)
    for i, a in enumerate(AA):
        lut[ord(a)] = i
    counts = np.zeros((2, N_AA), dtype=np.int64)      # row 0 = helix, 1 = other
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            seq, sst8 = row["seq"], row["sst8"]
            if row.get("has_nonstd_aa") == "True" or not set(seq) <= STD:
                continue
            aa = lut[np.frombuffer(seq.encode(), np.uint8)]
            is_h = (np.frombuffer(sst8.encode(), np.uint8) == ord("H")).astype(np.int64)
            # state index: helix -> 0, other -> 1  (matches EMIT row order)
            st = 1 - is_h
            counts += np.bincount(st * N_AA + aa, minlength=2 * N_AA).reshape(2, N_AA)
    return counts / counts.sum(axis=1, keepdims=True)


def main():
    emp = empirical_emissions()
    print(f"{'AA':>3}  {'helix model':>11} {'helix real':>10}  |  {'other model':>11} {'other real':>10}")
    print("-" * 62)
    for i, a in enumerate(AA):
        print(f"{a:>3}  {EMIT[0,i]*100:>10.1f}% {emp[0,i]*100:>9.1f}%  |  "
              f"{EMIT[1,i]*100:>10.1f}% {emp[1,i]*100:>9.1f}%")
    print("-" * 62)
    dh = np.abs(EMIT[0] - emp[0]).max() * 100
    do = np.abs(EMIT[1] - emp[1]).max() * 100
    print(f"max |model - real|:  helix {dh:.1f} pp   other {do:.1f} pp")
    ch = np.corrcoef(EMIT[0], emp[0])[0, 1]
    co = np.corrcoef(EMIT[1], emp[1])[0, 1]
    print(f"correlation model vs real:  helix r={ch:.3f}   other r={co:.3f}")

    x = np.arange(N_AA); w = 0.38
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
    for k, (name, r) in enumerate([("α-helix state", ch), ("other state", co)]):
        ax[k].bar(x - w/2, EMIT[k]*100, w, label="model (given table)", color=TEAL)
        ax[k].bar(x + w/2, emp[k]*100, w, label="real (PISCES, sst8 H-only)", color=GRAY)
        ax[k].set(ylabel="P(amino acid | state)  %",
                  title=f"{name}   —   model vs real   (r = {r:.3f})")
        ax[k].legend(fontsize=9)
        ax[k].grid(axis="y", alpha=0.25)
    ax[1].set_xticks(x); ax[1].set_xticklabels(AA)
    ax[1].set_xlabel("amino acid")
    fig.tight_layout(); fig.savefig(OUT, dpi=130)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
