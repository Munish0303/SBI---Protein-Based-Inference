"""Assemble the end-to-end pipeline notebook with nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):  cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# α-Helix Secondary Structure — End-to-End Pipeline
### Two-state HMM → Forward–Backward → BayesFlow → real-protein evaluation

This notebook runs **every step of the project in order, with diagrams**. It reuses the
project modules (`simulate.py`, `forward_backward.py`, `train_bayesflow.py`,
`insulin_eval.py`) so the model is defined in exactly **one** place.

**How to run:** open this notebook from the `restart/` folder and choose
*Kernel → Restart & Run All*. Heavy steps use a small demo scale for interactivity;
the full-scale results are shown from the saved runs.

| Step | What happens |
|------|--------------|
| 0 | The statistical model — a two-state HMM |
| 1 | Simulator — generate chains matched to real lengths |
| 2 | Forward–Backward — exact per-residue P(helix) targets |
| 3 | BayesFlow — amortized neural posterior (windowing + encoding) |
| 4 | Insulin & real-protein evaluation |
| 5 | Comparison & results |
""")

code(r"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")   # must precede keras/bayesflow import
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
from sklearn.metrics import roc_auc_score

TEAL, DEEP = "#1D9E75", "#0F6E56"
rng = np.random.default_rng(0)
print("environment ready — cwd:", os.getcwd())
""")

md(r"""
## Step 0 — The statistical model: a two-state HMM

Two hidden states, **helix** and **other**. Every chain starts in *other*. The state path is a
first-order Markov chain; each state emits an amino acid from its own table. All numbers come
from `simulate.py` (single source of truth).
""")

code(r"""
from simulate import AA, START_PROB, TRANS, EMIT, H, O, N_AA

print("Amino-acid alphabet:", "".join(AA))
print("\nStart distribution [helix, other]:", START_PROB, " -> always starts in 'other'")
print("\nTransition matrix (row = current state):")
print("             ->helix  ->other")
for name, i in [("helix", H), ("other", O)]:
    print(f"  from {name:5s}   {TRANS[i,H]:.2f}     {TRANS[i,O]:.2f}")
print("\nEmission probabilities (%):")
print("  AA    " + " ".join(f"{a:>3}" for a in AA))
for name, row in zip(["helix", "other"], EMIT):
    print(f"  {name:5s} " + " ".join(f"{int(round(v*100)):>3}" for v in row))
""")

md("**Diagram — the two-state HMM and its emission tables:**")
code(r"""
display(Image(filename="../hmm_explainer/1_two_state_hmm.png"))
display(Image(filename="../hmm_explainer/2_emission_probabilities.png"))
""")

md(r"""
## Step 1 — Simulator

Generate amino-acid chains from the HMM. Chain **lengths are sampled from the real PISCES
length distribution**, so the synthetic set mirrors real proteins in length.
""")

code(r"""
from simulate import simulate_chain, load_lengths

# one example chain
st, seq, n_h = simulate_chain(60, rng)
print("hidden states:", st)
print("amino acids  :", seq)

# a demo set (small, for interactivity; the real run uses 100,000)
gt_lengths = load_lengths("../archive/2018-06-06-pdb-intersect-pisces.csv")
N_DEMO = 4000
lengths = rng.choice(gt_lengths, size=N_DEMO, replace=True)
chains = [simulate_chain(int(L), rng) for L in lengths]
hf = sum(c[2] for c in chains) / sum(len(c[0]) for c in chains)
print(f"\nsimulated {N_DEMO} chains; helix fraction {hf:.3f} (stationary theory 0.333)")
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
ax[0].hist(lengths, bins=40, color=TEAL)
ax[0].set(title="simulated chain lengths", xlabel="length", ylabel="chains")
s0 = np.array([1 if c == 'H' else 0 for c in st])
ax[1].step(range(len(s0)), s0, where="mid", color=DEEP)
ax[1].set(title="example hidden-state path", xlabel="residue", yticks=[0, 1])
ax[1].set_yticklabels(["other", "helix"])
plt.tight_layout(); plt.show()
""")

md(r"""
## Step 2 — Forward–Backward (exact per-residue posterior)

Given a sequence, `hmmlearn`'s Forward–Backward gives the exact **P(helix | whole chain)** for
every residue. These are the "answer key" targets that BayesFlow will learn to reproduce.
""")

code(r"""
from forward_backward import build_model, fb_posterior, encode

fb_model = build_model()
p_helix, loglik = fb_posterior(seq, fb_model)
true = np.array([1 if c == 'H' else 0 for c in st])

plt.figure(figsize=(11, 3.2))
plt.step(range(len(true)), true, where="mid", color="0.6", label="true hidden state")
plt.plot(p_helix, color=DEEP, lw=2, label="P(helix) — Forward–Backward")
plt.axhline(0.5, ls=":", color="0.7"); plt.ylim(-.05, 1.05)
plt.xlabel("residue"); plt.ylabel("P(helix)"); plt.legend()
plt.title("Exact posterior for one simulated chain"); plt.show()
print("residue-0 P(helix) =", round(float(p_helix[0]), 3), " (start rule -> exactly 0)")
""")

code(r"""
# validate the exact posterior against the KNOWN hidden states over a subset
pa, ta = [], []
for st_i, seq_i, _ in chains[:1000]:
    p, _ = fb_posterior(seq_i, fb_model); pa.append(p)
    ta.append((np.frombuffer(st_i.encode(), np.uint8) == ord('H')).astype(int))
pa = np.concatenate(pa); ta = np.concatenate(ta)
print(f"mean P(helix) {pa.mean():.3f}  vs  true helix fraction {ta.mean():.3f}  (well calibrated)")
print(f"AUC vs true hidden state {roc_auc_score(ta, pa):.3f}  (Bayes-optimal ceiling ~0.79)")
""")

md(r"""
## Step 3 — BayesFlow amortized posterior

A normalizing flow needs a **fixed-size** input, but chains vary in length. So we predict **one
residue at a time** from a fixed **31-residue window** (±15), slid along the chain. The window is
**one-hot encoded** as 31 × 21 channels (20 amino acids + 1 padding for chain ends). Because the
window is only local, the target stays uncertain → a genuine posterior (mean ± std).
""")

code(r"""
display(Image(filename="bayesflow_window_schematic.png"))

from train_bayesflow import windows_for_sequence, WINDOW, N_CH, predict_helix
Wex = windows_for_sequence(encode(seq))
print(f"one window per residue -> shape (L, WINDOW, channels) = {Wex.shape}")
print(f"WINDOW = {WINDOW} residues,  channels = {N_CH} (20 amino acids + 1 padding)")
""")

md(r"""
Load the trained posterior and compare it to exact FB on the example chain.
*(To retrain from scratch: `python train_bayesflow.py` — 15,000 chains, 300,000 windows, 30 epochs.)*
""")

code(r"""
import bayesflow as bf     # registers serializable classes for loading
import keras
approx = keras.saving.load_model("bayesflow_posterior.keras")

mean, std = predict_helix(approx, encode(seq), num_samples=200)

plt.figure(figsize=(11, 3.2))
plt.step(range(len(true)), true, where="mid", color="0.6", label="true state")
plt.plot(p_helix, "k-", lw=2, label="exact FB")
plt.plot(mean, color=TEAL, lw=2, label="BayesFlow mean")
plt.fill_between(range(len(mean)), np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                 color=TEAL, alpha=0.25, label="±1 std")
plt.axhline(0.5, ls=":", color="0.7"); plt.ylim(-.05, 1.05)
plt.xlabel("residue"); plt.ylabel("P(helix)"); plt.legend()
plt.title("BayesFlow vs exact FB — example chain"); plt.show()
""")

md("**Validation on held-out simulated chains (full 15k-chain run):** correlation 0.999, MAE 0.007.")
code(r"""display(Image(filename="validation_figure.png"))""")

md(r"""
## Step 4 — Insulin & real-protein evaluation

The model trained **only on simulated data**, so real proteins are genuinely held out (no leakage).
We use **wild-type** human insulin (`1MSO`) — an earlier version used `1A7F`, which is a *mutant*
(B16 Tyr→Glu, B24 Phe→Gly, des-B30) whose sparse A-chain annotation inflated its A-chain AUC.
Scored against the strict `sst8` H-only helix labels.
""")

code(r"""
from insulin_eval import fetch_chains

for code_, seqi, sst8 in fetch_chains("1MSO"):
    obs = encode(seqi)
    ti = np.array([1 if c == 'H' else 0 for c in sst8], dtype=int)
    bfp, _ = predict_helix(approx, obs, num_samples=300)
    fbp, _ = fb_posterior(seqi, fb_model)
    print(f"1MSO chain {code_} (len {len(seqi):>2}, helix {int(ti.sum()):>2}): "
          f"BayesFlow AUC={roc_auc_score(ti, bfp):.3f}   FB AUC={roc_auc_score(ti, fbp):.3f}")

# 1MSO is a dimer: chains A,C are A-chains; B,D are B-chains.
display(Image(filename="insulin_1MSO.png"))
""")

md(r"""
The **B-chain** helix is propensity-driven (L, V, E, A) → recovered (**AUC 0.98**). The **A-chain**
helix is **disulfide-stabilised and cysteine-rich**, and C is helix-*disfavouring* in the emission
table → the model predicts "not helix" exactly where the helix is (**AUC 0.52, chance**). A
sequence-only HMM cannot see 3-D disulfide bonds. (Only 2–4 chains — illustrative, not a benchmark.)
""")

md(r"""
**All real PISCES proteins** (full run: `python eval_real.py --limit 0`) — 8,994 chains, per-chain
AUC 0.754, BayesFlow = exact FB.
""")
code(r"""display(Image(filename="real_eval_auc_hist.png"))""")

md(r"""
## Step 5 — Comparison & results

BayesFlow reproduces exact Forward–Backward in **every** setting (faithful amortization). Scored
against a **majority-class baseline**: accuracy@0.5 only beats the baseline on the calibrated
(simulated / pooled real) settings — on insulin it ties or falls below it, so **AUC is the fair
metric**. (`python compare_metrics.py`, `python emission_check.py`.)
""")
code(r"""
display(Image(filename="comparison_table.png"))
display(Image(filename="comparison.png"))
display(Image(filename="emission_check.png"))   # emissions vs real data: r = 0.99
""")

md(r"""
## Step 6 — SBI diagnostics

Predictive metrics say the estimate is *useful*; these say whether the *posterior* is honest.
Convergence, recovery (r = 0.999) and posterior contraction (0.999) are excellent, but **SBC
reveals a small (~0.15 posterior-SD) calibration bias** — the uncertainty band is not perfectly
calibrated. Likely cause: the near-degenerate 3-D target (dim-correlation 0.89–0.96) vs a coupling
flow. Not the logit-clip atom at residue 0 (excluding it doesn't fix SBC). (`python diagnostics.py`.)
""")
code(r"""
display(Image(filename="diag_loss.png"))
display(Image(filename="diag_sbc_ecdf.png"))
display(Image(filename="diag_recovery.png"))
""")

md(r"""
## TL;DR

- α-helix as a **2-state HMM** whose transition **and emission** tables match real proteins (r ≈ 0.99).
- **BayesFlow** amortized posterior trained on exact **Forward–Backward** targets via a sliding **31-residue window**; reproduces exact FB almost perfectly (**r = 0.999**).
- **Diagnostics:** excellent recovery & contraction, but SBC shows a small (~0.15 SD) calibration bias — an owned limitation.
- On unseen real proteins it ranks **propensity-driven** helices well (insulin B-chain 0.98; PISCES 0.75) but **fails where structure is 3-D-stabilised** (insulin A-chain 0.52) — the price of a sequence-only model. Exact FB is a ceiling on *simulated* data only.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
with open("SBI_pipeline.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote SBI_pipeline.ipynb with", len(cells), "cells")
