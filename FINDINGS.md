# Protein secondary-structure HMM — findings & reference

Working notes for the alpha-helix (two-state HMM) project. Last updated **2026-07-04**.
Restart workspace: `D:\SBI\restart\`. Original/prior project: `D:\SBI\` root.

---

## 1. Goal

Predict the **alpha-helix** secondary-structure pattern of proteins with a
**two-state Hidden Markov Model**:

- State **helix** = alpha-helix (the curly spiral).
- State **other** = everything else (beta-sheet, coil, turns, …).

Downstream aim: simulate sequences, get exact posteriors via Forward–Backward,
train a **BayesFlow** neural posterior estimator, and validate against real
annotations (e.g. human insulin).

---

## 2. The generative model (simulator spec)

Amino-acid alphabet, in the exact column order of the emission tables (20 residues):

```
A R N D C E Q G H I L K M F P S T W Y V
```

**Start:** every chain **always starts in the `other` state.**

**Transition probabilities** (first-order Markov chain):

| From \ To | helix | other |
|-----------|-------|-------|
| **helix** | 0.90  | 0.10  |
| **other** | 0.05  | 0.95  |

**Emission probabilities** (chance of each amino acid given the state; each row sums to 100%):

| State | A | R | N | D | C | E | Q | G | H | I | L | K | M | F | P | S | T | W | Y | V |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **helix** | 12 | 6 | 3 | 5 | 1 | 9 | 5 | 4 | 2 | 7 | 12 | 6 | 3 | 4 | 2 | 5 | 4 | 1 | 3 | 6 |
| **other** | 6 | 5 | 5 | 6 | 2 | 5 | 3 | 9 | 3 | 5 | 8 | 6 | 2 | 4 | 6 | 7 | 6 | 1 | 4 | 7 |

Verified: **both emission rows sum to 100%**, both transition rows sum to 1.

### Derived facts
- **Stationary helix fraction** (long-run): `0.05 / (0.05 + 0.10) = 0.333`.
- Helix regions come in **runs** (once started, 90% chance to continue) and are
  **rarely started** (only 5% from `other`). Most residues are `other`.
- **Helix-lovers** (emit far more often in helix than other): **A** (12 vs 6),
  **L** (12 vs 8), **E** (9 vs 5), **I** (7 vs 5).
- **Helix-breakers** (far more common in other): **G** (4 vs 9), **P** (2 vs 6),
  plus S, T, N, D.

---

## 3. Ground-truth dataset (`D:\SBI\archive\`)

Real proteins whose secondary structure was measured experimentally (X-ray / NMR)
and annotated per-residue by **DSSP**. This is the real answer key to verify against.

### DSSP 8-state (Q8) labels
`C` loop/irregular · `E` β-strand · `H` α-helix · `B` β-bridge · `G` 3₁₀-helix ·
`I` π-helix · `T` turn · `S` bend.

### Q8 → Q3 collapse (standard)
- helices `(H, G, I) → H`
- strands `(E, B) → E`
- coil `(C, S, T) → C`

> **NB — typo in the assignment PDF:** it writes "(H, G, I) into E"; that is wrong,
> it should be **into H**. Helices go in the helix bucket.

### Accuracy context
Q3 prediction ~85%; Q8 prediction <70% (depends on test set).

### The two files

| File | Rows (chains) | Notes |
|------|---------------|-------|
| `2018-06-06-ss.cleaned.csv` | **393,732** | full dump, all PDB chains |
| `2018-06-06-pdb-intersect-pisces.csv` | **9,078** | PISCES-culled, high quality, **"ready for training"**; extra columns: `Exptl., resolution, R-factor, FreeRvalue` |

Common columns: `pdb_id, chain_code, seq, sst8, sst3, len, has_nonstd_aa`.
`seq`, `sst8`, `sst3` are all the **same length** and aligned position-by-position.
Nonstandard amino acids (B, O, U, X, Z) are masked with `*`; `has_nonstd_aa` flags them.

---

## 4. Mapping the real data onto our two states — DECISION

We use **`sst8` and keep only `H` as helix; the other 7 states → other.**

- This is the **strict, precise** alpha-helix definition.
- It is **better than using `sst3`** for this project, because the `sst3` "H"
  bucket secretly also includes `G` (3₁₀-helix) and `I` (π-helix), which are *not*
  true alpha-helices.

| Approach | counts as helix | goes to other |
|----------|-----------------|---------------|
| **`sst8`, H-only (chosen)** | H | G, I, E, B, T, S, C |
| `sst3`, H bucket | H, G, I | E, B, T, S, C |

**Rule:** train and verify on the *same* mapping.

**Measured real helix fraction** (DSSP `H` only, over annotated PISCES residues): **≈ 0.316**.

---

## 5. Chain-length findings (answers "what is the length?")

Lengths are **not uniform — they vary per chain.** There is no single length.

| Dataset | chains | min | max | mean | median | 5th pct | 95th pct |
|---------|--------|-----|-----|------|--------|---------|----------|
| PISCES (training-ready) | 9,078 | 20 | 1632 | 243.2 | 208 | 55 | 525 |
| ss.cleaned (full) | 393,732 | 3 | 5037 | 260.2 | 223 | 38 | 580 |

To simulate an arbitrary number of chains matching the real profile, **sample
lengths with replacement from the ground-truth length distribution** (default
source: PISCES).

---

## 6. Simulator — `restart/simulate.py`

Self-contained (transition + emission tables baked in; no external project deps
beyond numpy). What it does:

1. Load chain lengths from a ground-truth CSV (default PISCES).
2. Sample `N` lengths with replacement to match the real length profile.
3. For each length, draw a chain: start in `other`, walk the Markov chain, emit
   amino acids from the state's emission table.
4. Write `simulated_chains.csv` with columns: `chain_id, len, states, seq`
   (`states` uses `H`/`O`).

### Run
```
cd D:\SBI\restart
python simulate.py                    # 100,000 chains, seed 0 -> simulated_chains.csv
python simulate.py --n 5000 --seed 1  # smaller run
python simulate.py --input ..\archive\2018-06-06-ss.cleaned.csv   # lengths from full file
```

### Output produced (this session)
- `restart/simulated_chains.csv` — **100,000 chains**, **24,282,450 residues**, ~48 MB.
- Realized length mean/median: **242.8 / 208** (matches real PISCES profile).
- Runtime ≈ 70 s.

### Correctness checks (passed)
- Simulated helix fraction = **0.325** ≈ theoretical stationary **0.333**. ✔
- `seq` only uses the 20 standard amino acids. ✔
- Emission/transition rows sum to 1 (asserted at import). ✔

> **Important caveat:** the simulated `seq` letters are **not** expected to match
> any real protein — the HMM invents letters from fixed emission tables that do
> not know real 3D structure. What matches by design is the **length profile** and
> the **statistical behaviour** (helix runs, rare starts). Real sequences/labels
> live only in the `archive` CSVs.

---

## 7. Related earlier artifacts (prior project, `D:\SBI\` root)

- `hmm.py` — core model constants + `simulate()` + from-scratch `forward_backward()`.
- `bf_infer.py` — BayesFlow amortized posterior over a sliding local window
  (inference variable = logit P(helix at centre residue); condition = one-hot
  window; `WINDOW = 31`). FB targets from hmmlearn over the whole chain.
- `insulin.py`, `run_all.py` — end-to-end pipeline + figures.
- `simulate_from_csv.py` — earlier simulator that made **one** synthetic chain per
  real chain and carried the real `sst8`→H-only truth alongside for comparison.
- Prior reported results (from project memory): BayesFlow ≈ Forward–Backward
  (corr ≈ 0.999); human insulin B-chain helix detection AUC ≈ 0.99.

---

## 8. Explainer images (`D:\SBI\hmm_explainer\`)

Three saved diagrams (`.svg` + 2× `.png`): (1) the two-state HMM with
transition arrows, (2) emission probabilities helix-vs-other bar chart
(green = alpha-helix/spiral, grey = other), (3) the BayesFlow training pipeline.

---

## 9. Forward–Backward step — `restart/forward_backward.py`

Exact per-residue helix posteriors via **hmmlearn 0.3.3** `CategoricalHMM`.
Reuses the model parameters imported from `simulate.py` (single source of truth).

What it does:
1. `build_model()` wires a `CategoricalHMM` to our fixed `startprob_` (always
   other), `transmat_` (0.90/0.10, 0.05/0.95) and `emissionprob_` (the two tables).
   `init_params="" , params=""` → inference only, never trained/re-estimated.
2. `fb_posterior(seq, model)` runs `score_samples` → returns `gamma[:, helix]`,
   i.e. **P(helix | whole chain)** for every residue, plus the log-likelihood.
3. `main()` streams `simulated_chains.csv`, computes posteriors, saves targets,
   and validates the posterior against the KNOWN simulated hidden states.

### Run
```
cd D:\SBI\restart
python forward_backward.py                 # all 100k chains -> fb_targets.npz
python forward_backward.py --limit 3000    # quick subset
```

### Outputs produced (this session)
- `restart/fb_targets.npz` (~86 MB): `lengths` (100,000), `chain_id`,
  `p_helix` (24,282,450 float32), `true_states` (uint8, 1=helix). Split per chain:
  `np.split(p_helix, np.cumsum(lengths)[:-1])`.
- `restart/fb_sample.csv`: first 5 chains rendered residue-by-residue
  (`chain_id, position, residue, true_state, P_helix`).

### Results / checks (passed)
- `sum(lengths) == len(p_helix)` ✔ (targets align with residues).
- **Well calibrated:** mean posterior P(helix) = **0.324** ≈ true helix frac **0.325**.
- Residue 0 always has **P(helix) = 0.000** — confirms the "start in other" rule
  is honoured by the algorithm. ✔
- **Recovering the hidden truth from letters alone:** accuracy @0.5 = **0.753**,
  **AUC = 0.790**. This is the *inherent ceiling*: emission tables for helix/other
  overlap a lot, so even the exact Bayesian posterior is uncertain — that is
  correct behaviour, not a bug. (It also makes the posterior a genuine, non-trivial
  target for BayesFlow to learn.)
- Runtime ≈ 84 s for all 100k chains.

> This `p_helix` (gamma) is the exact "answer key" BayesFlow will be trained to emulate.

### Extended validation (all passed, 2026-07-04)

| Check | What it proves | Result |
|-------|----------------|--------|
| Alignment | no residue dropped/shifted | `sum(lengths)==len(p_helix)` ✔ |
| Global calibration | no systematic bias | mean pred 0.324 ≈ true 0.325 |
| Start rule | `startprob=[0,1]` applied | residue 0 → P=0.000 exactly |
| Truth recovery | posterior is informative | acc 0.753, AUC 0.790 (below 1.0 by design) |
| **A. Independent FB** | hmmlearn == from-scratch log-space FB | max\|diff\| = **9.6e-14**; gamma rows sum to 1 (1e-13) |
| **B. Simulator recovery** | generator draws from intended matrices | (full 100k) max\|emp−TRANS\|=0.00022, max\|emp−EMIT\|=0.00018; **100000/100000** chains start in "other" |
| **C. Reliability curve** | probabilities trustworthy at every level | predicted ≈ empirical within ~0.002 across all 10 deciles |
| **D. Positional marginal** | start rule + burn-in | helix frac rises 0.000 (pos0) → ~0.333 by pos ~50; explains why global mean 0.324 < stationary 0.333 |
| **E. Numerical sanity** | no under/overflow on long chains | all 24.28M values finite and in [0,1]; max 0.980 |
| **F. Viterbi vs posterior** | decoders consistent | agree 83.7%; posterior acc 0.753 ≥ Viterbi acc 0.710 (expected: marginal decoding maximises per-residue accuracy) |

The ~0.79 AUC is the **Bayes-optimal ceiling — ON SIMULATED DATA ONLY**, where the HMM
*is* the true generating model, so exact FB is by definition optimal and BayesFlow can
approach but not exceed it. **This claim does NOT transfer to real proteins**: there the
HMM is misspecified, so FB is *not* a ceiling and a better model can (and does) beat it.
(Numbers differ slightly by subset/run: FB scored 0.790/0.753 on the 3,000-chain FB check
in §9, and 0.798/0.760 on the 500-chain held-out validation split in §10. Same model,
different n — not a discrepancy.)

### Further checks still available (not yet run)
- Log-likelihood model discrimination: true model should score simulated data higher
  than a wrong model (swapped transitions / shuffled emissions).
- Per-chain aggregate calibration: expected helix count `E[Σγ]` vs actual count.
- Reproducibility: same seed → byte-identical output files.
- Edge cases: length-1 / very short chains.
- Cross-check against the prior project's independent `hmm.py` `forward_backward`.
- Accuracy stratified by helix-run length.
- Downstream (post-training): MAE / correlation of BayesFlow vs these FB targets.

---

## 10. BayesFlow amortized posterior — `restart/train_bayesflow.py`

Neural posterior that emulates the exact Forward–Backward answer instantly.

### Modeling choice (why windows)
A normalizing flow needs a **fixed-size** target, but chains are variable length.
So we amortize the posterior of a **single residue** given a fixed **31-residue
one-hot window** centred on it (slid across the chain at inference). The window is
only local, so window→gamma is genuinely stochastic → a non-degenerate posterior,
which is what BayesFlow is built for.

### Parameters (with reasoning)
| Param | Value | Why |
|-------|-------|-----|
| `WINDOW` | 31 (±15) | spans a helix run (~10) + context |
| channels `N_CH` | 21 = 20 AA + 1 pad | categorical one-hot; pad channel marks chain ends |
| target `theta` | 3-D `logit P(helix)` at prev/centre/next | logit → unbounded space for the flow; ≥2 dims because coupling flows split dims |
| inference net | `CouplingFlow(depth=6, widths=(256,256))` | enough capacity; matches prior corr≈0.999 |
| conditions | flattened 31×21 = 651 window as `inference_conditions` | MLP preserves position, avoids slow summary net |
| approximator | `ContinuousApproximator` | standard amortized posterior |
| optimizer | Adam; loss = flow max-likelihood | standard |
| `batch_size` | 512 | small examples, fast |
| `epochs` | 30 | converges on ~100k windows |
| `max_windows` | **300,000** | memory-bounded (~0.78 GB float32); ~20 positions × 15000 chains |
| train/val | **15000 train (front) / 500 (tail)** held-out chains | disjoint blocks → no leakage |
| `num_samples` | 300 | posterior mean = prediction, std = uncertainty |

### No-leakage design (verified at runtime)
Train = front index block `[0, train_chains)`; validation = **tail block**
`[total − val_chains, total)`. They cannot overlap however large training grows.
Two asserts run + print every time: (a) index sets disjoint, (b) no validation
sequence string appears in the training set. Observed: `disjoint OK | overlap = 0`.

### Run
```
cd D:\SBI\restart
python train_bayesflow.py                            # 15k chains / 300k windows / 30 epochs
python train_bayesflow.py --train-chains 30000 --max-windows 600000 --epochs 40
python train_bayesflow.py --smoke                    # tiny fast end-to-end check
```

### Results (held-out validation, scaled-up run, 2026-07-04)
- **BayesFlow vs exact FB: corr = 0.9990, MAE = 0.0074** (↑ from 0.9983 / 0.0101 at
  100k windows) → reproduces the exact posterior almost perfectly.
- mean posterior std = 0.0088 (tight, confident).
- **AUC vs TRUE state: BayesFlow 0.798 = FB 0.798** — now exactly at the Bayes-optimal
  ceiling. acc@0.5: 0.760 = 0.760.
- Saved model: `restart/bayesflow_posterior.keras` (~33 MB). Training ≈ 73 s/epoch
  (586 steps), ~40 min total (CPU, torch).

> Interpretation: the fast neural posterior is interchangeable with slow HMM
> Forward–Backward. It approaches but never beats FB (the exact Bayesian answer);
> AUC 0.798 = 0.798 confirms no leak/bug. Scaling data 3× tightened corr 0.998→0.999.

---

## 11. Evaluation on held-out & REAL proteins

Three scripts, all run against the model that trained ONLY on simulated data
(so every real protein below is genuinely out-of-distribution — no leakage).

### `make_figures.py` → `validation_figure.png`
Held-out simulated tail block. Scatter of BayesFlow mean vs exact FB is a near-
perfect diagonal (**MAE 0.007, r 0.999**, n=17,950 residues); example chain shows
BayesFlow sitting on exact FB with a thin ±1 std band.

### `insulin_eval.py` → `insulin_prediction.png`  (assignment step 4)
Human insulin **1A7F** pulled straight from the real data (seq + DSSP truth),
scored against strict sst8 H-only labels. Model never saw any real protein.

| chain | len | helix res | BayesFlow AUC | FB AUC | acc@0.5 |
|-------|-----|-----------|---------------|--------|---------|
| A | 21 | 4 | **0.971** | 0.971 | 0.810 |
| B | 29 | 10 | **0.984** | 0.984 | 0.793 |

The P(helix) curve rises *inside* the annotated helix on both chains; BayesFlow
overlaps exact FB to 3 decimals even here. Absolute P(helix) stays modest
(B-chain peaks ~0.56, A-chain ~0.3) → the HMM is uncalibrated to reality, which is
why **AUC (ranking) is the fair metric** and acc@0.5 is lower. Matches the prior
project's finding (B-chain AUC ≈ 0.99).

### `eval_real.py` → `real_eval_per_chain.csv`, `real_eval_auc_hist.png`
**ALL real PISCES chains** (`--limit 0`, 50 samples/residue): 8,994 standard-AA
chains scored (84 skipped for nonstandard AA), 2,195,387 residues, sst8 H-only:
- per-chain AUC (mean): **BayesFlow 0.754 = FB 0.754** (n=8,308 chains w/ both classes)
- pooled AUC (2.2M residues): **BayesFlow 0.771 vs FB 0.772**
- per-chain acc@0.5: 0.743 = 0.743; true helix frac 0.316
- (An earlier first-1,000-chain subset read 0.777/0.815 — the top of the file was
  slightly easier; the full-dataset numbers above are the definitive ones.)

Takeaways: (1) BayesFlow ≈ exact FB on simulated, insulin, AND all real data —
the amortization is faithful out-of-distribution. (2) The simple 2-state HMM ranks
real helices meaningfully (AUC ~0.75 across all proteins; ~0.97–0.98 on insulin's
clean canonical helices). (3) acc@0.5 depressed vs AUC due to calibration gap.

### `compare_metrics.py` → `comparison.png`  (consolidated comparison)
Single table + grouped bar chart (AUC and accuracy panels) of **BayesFlow vs exact
FB vs ground truth** across all four settings:

| setting | AUC BF | AUC FB | acc BF | acc FB |
|---------|--------|--------|--------|--------|
| held-out simulated | 0.798 | 0.798 | 0.760 | 0.760 |
| insulin A | 0.971 | 0.971 | 0.810 | 0.810 |
| insulin B | 0.984 | 0.984 | 0.793 | 0.793 |
| real PISCES (8,994 chains) | 0.754 | 0.754 | 0.743 | 0.743 |

The BayesFlow and FB bars are visually identical in every setting — the headline
result: **the trained neural posterior reproduces exact Forward–Backward
everywhere**, and both track ground truth equally (high on insulin's textbook
helices, ~0.75 across the messy real proteome). The real-PISCES row is recomputed
live from `real_eval_per_chain.csv`; other rows are carried from their runs.

---

## 12. Open next steps

- [x] ~~Run exact Forward–Backward to produce per-residue `P(helix)` targets.~~ → `fb_targets.npz`
- [x] ~~Train / validate BayesFlow against these targets.~~ → corr 0.999, `bayesflow_posterior.keras`
- [x] ~~Evaluate on real proteins using `sst8` H-only labels (AUC / accuracy).~~ → `eval_real.py`
      (definitive, all 8,994 chains: **per-chain AUC 0.754**, pooled 0.771)
- [x] ~~Validation figure (scatter + example chain w/ uncertainty band).~~ → `validation_figure.png`
- [x] ~~Insulin test vs ground truth (assignment step 4).~~ → `insulin_prediction.png` (A 0.97, B 0.98)

The restart pipeline is complete end-to-end: simulate → Forward–Backward → BayesFlow
→ validation → real-protein & insulin evaluation. Possible extensions: empirical
emission-table check vs real data; Q3 (H,G,I) helix definition comparison; larger
training run (`--max-windows 600000`).

---

## 13. SBI DIAGNOSTICS (Block A) — `diagnostics.py`

The metrics above (AUC, accuracy, correlation-with-FB) are **predictive**. An SBI workflow
also requires **inferential** diagnostics: is the *posterior* honest? Run on the held-out
tail block (2,000 windows, 250 draws each), never trained on.

| Diagnostic | Figure | Result |
|---|---|---|
| Convergence (loss curve) | `diag_loss.png` | loss −3.83 → **−7.91** over 30 epochs, plateaued |
| Recovery | `diag_recovery.png` | **r = 0.999** on all 3 target dims |
| Posterior contraction | `diag_contraction.png` | **0.999** (posterior far tighter than prior) |
| **SBC rank ECDF** | `diag_sbc_ecdf.png` | ❌ **FAILS** — ECDF exits the 95% band on all 3 dims |

### The SBC failure — diagnosed
- **Magnitude is small:** posterior *width* is well calibrated (RMSE/sd ≈ 1.0, mean\|z\| =
  0.82–0.88 vs 0.80 expected). The problem is a **location bias of only 0.12–0.20 posterior
  SDs** (≈0.01 logit ≈ 0.002 in probability). Point estimates are unaffected (corr 0.999).
- **The logit-clip atom — see §16 for the controlled test.** γ = 0 *exactly* at residue 0 for
  every chain (start rule) → `logit` clipped to −6.907, a **point mass** a continuous flow cannot
  represent (the ONLY such atom: γ ≥ 0.0043 everywhere else in 24.3M residues).
  ⚠️ An earlier note here claimed "excluding the start residues doesn't fix SBC, so it's not the
  cause". **That test was invalid** — it excluded them at *evaluation* time only, while the model
  had already been *trained* with them. §16 reports the proper test (a retrain).
- **Likely cause: a near-degenerate target.** The 3-D target (γ at prev/centre/next) has
  inter-dim correlations **0.89–0.96** — the true joint is nearly a 1-D curve in 3-D. A
  *coupling* flow splits dimensions asymmetrically, which is consistent with the observed
  opposite-sign bias (prev/centre biased high, next biased low).
- **Honest reading:** the point estimate is excellent, but the uncertainty is *not* perfectly
  calibrated. Candidate fixes (untested): decorrelate/reduce the target, more capacity or
  epochs, or a non-coupling inference net (e.g. FlowMatching).

---

## 14. CORRECTIONS (Block B)

### (a) Insulin: we were reporting a MUTANT — the A-chain result does not survive
`1A7F` is **not wild-type human insulin**. Diffed against the canonical sequence it carries
**B16 Tyr→Glu, B24 Phe→Gly, and des-B30** (hence len 29, not 30). `1MSO` is true wild-type.

| Structure | Chain | helix res | AUC BayesFlow | AUC FB |
|---|---|---|---|---|
| **1MSO (wild-type)** | **B** | 11/30 | **0.981** | 0.986 |
| **1MSO (wild-type)** | **A** | 12/21 | **0.519** | 0.509 |
| 1A7F (mutant) | B | 10/29 | 0.984 | 0.984 |
| 1A7F (mutant) | A | 4/21 | 0.971 | 0.971 |

> The previously headlined **A-chain AUC 0.97 was an artifact of the mutant structure's
> sparse annotation.** On wild-type, the A-chain is at **chance (0.52)**.

**Why (and it's a good story):** the B-chain central helix is propensity-driven (L, V, E, A,
L — classic helix formers) and the model nails it. The A-chain's N-terminal helix is
**stabilized by disulfide bonds** and is cysteine-rich — and in the emission table **C is
helix-*disfavoring*** (1% helix vs 2% other). A sequence-propensity HMM therefore predicts
"not helix" exactly where the real helix is. It cannot see 3-D disulfide stabilization.

Also: insulin conclusions rest on very few positive residues (n = 2–4 chains) — treat as
illustrative, not as a robust benchmark.

### (b) Majority-class baseline — accuracy@0.5 is not impressive
| Setting | helix frac | trivial baseline | our acc@0.5 | verdict |
|---|---|---|---|---|
| Held-out simulated | 0.325 | 0.675 | 0.760 | beats it |
| Real PISCES (8,994) | 0.322 | 0.678 | 0.743 | beats it |
| Insulin B (1MSO) | 0.367 | 0.633 | 0.633 | **ties it** |
| Insulin A (1MSO) | 0.571 | 0.571 | 0.429 | **BELOW it** |

On insulin the model **never crosses P = 0.5** (uncalibrated to real proteins), so it predicts
all-"other" → accuracy collapses to the baseline or below. **AUC is the only fair metric here.**

### (c) Emission tables ARE realistic (the check we previously never did)
`emission_check.py` → `emission_check.png`. Empirical P(aa | state) from real PISCES
(sst8 H-only) vs the given tables:
- **max deviation: 0.9 pp (helix), 0.8 pp (other)**
- **correlation: r = 0.992 (helix), r = 0.982 (other)**

So both the transition *and* emission tables are empirically well-founded. (Note: the earlier
claim "100% of real chains start in other" is **near-tautological** — DSSP cannot assign H to a
terminal residue — so it is not independent confirmation of the start rule.)

---

## 15. Conceptual defences (Q&A prep) — items 2, 3, 4

Three points that don't change any number but close the conceptual gaps most likely to be
probed. All are now on the slides + in speaker notes.

### (2) "Where is your prior?"
The report spec demands explicit proper priors, but **this project infers no parameters** —
start, transitions and emissions are all fixed. The correct answer:

> The **prior** is the induced distribution over hidden state paths **p(z₁:T)**, given
> `startprob = [0, 1]` and the transition matrix. The **likelihood** is the emission table.
> In the amortized setup, the effective prior over θ is the **marginal of logit γ** induced by
> simulating chains and running Forward–Backward — which is exactly what SBC ranks against.

On slide 4 ("Prior & likelihood" card).

### (3) The target/condition mismatch — and why it's harmless ⭐
θ is γ computed by FB from the **whole chain**, but we condition on a **31-residue window**.
So the flow learns **p(γ_full | local window)** — not a posterior over anything generative.
We own this. Why it's harmless (**verified numerically**):

| distance k | influence λ₂^k |
|---|---|
| ±5 | 0.444 |
| ±10 | 0.197 |
| **±15 (our window)** | **0.087** |
| ±20 | 0.039 |

- The transition matrix's eigenvalues are exactly **[1.0, 0.85]** → **λ₂ = 1 − 0.10 − 0.05 = 0.85**.
- Influence decays geometrically; beyond ±15 residues **<10%** of the information remains.
- Empirical check: recomputing γ from the 31-window alone vs the full chain differs by
  median **0.010** (mean 0.0147, 95th 0.045). *Caveat: this is an **upper bound** — standalone-window
  FB wrongly re-imposes the "start in other" rule at the window edge, inflating the gap.*

> **This is why r = 0.999 is EXPECTED, not suspicious.** Without this argument the near-perfect
> agreement looks like leakage. On slide 7 ("Why ±15 is enough" card) + slide 11.

### (4) "Why SBI when the likelihood is tractable?"
The weak answer ("instant inference, no HMM at test time") is indefensible — `hmmlearn` FB
already runs in milliseconds. The strong answer:

> **The tractability is the point.** This is a benchmark setting where the exact posterior is
> *known*, which is the only way to **verify** that an amortized neural posterior is faithful
> (r = 0.999). In a real SBI problem the likelihood is intractable and this check is impossible.

On slide 8 ("Why SBI if the likelihood is tractable?" card) + TL;DR.

### Not done — Block C item 1: summary network
We deliberately have **no summary network**; the 31×21 window is flattened to 651 and fed
straight in as `inference_conditions`. A small 1D-CNN summary net (BayesFlow's
`TimeSeriesNetwork`) would be the methodologically canonical choice and is the obvious
"where is your summary network?" answer. **Deferred** — expected to change nothing numerically,
and it would **not** fix the SBC bias (that lives on the *target* side, not the conditioning side).
The actual candidate fixes for SBC: reduce/decorrelate the 3-D target, or swap the coupling flow
for FlowMatching.

---

## 16. The clip-atom experiment + fixes 5/6/7  (2026-07-17)

### The controlled experiment: does removing the logit-clip atom fix SBC?

**Motivation.** γ₀ = 0 *exactly* for every chain (start rule) — the only atom in 24.3M residues.
`logit` clips it to −6.907. Measured damage on the training targets:

| dim | clip atoms | standardisation sd | inflation |
|-----|-----------|--------------------|-----------|
| prev | 4,015 | 1.4347 (vs 1.2713 clean) | **+12.9%** |
| centre | 2,015 | 1.3540 (vs 1.2687 clean) | **+6.7%** |
| next | **0** | 1.2665 | 0.0% |

The atom lands in `prev`/`centre` but never `next` (a window at position 0 has
prev = centre = γ₀ = atom, while next = γ₁ is ordinary). BayesFlow standardises
`inference_variables` by default, so this **skews the target scaling** — and those are exactly
the dims that showed a *positive* SBC bias while the clean dim showed a *negative* one.

**Fix.** `build_training_data(..., skip_start=SKIP_START=2)` drops the first two residues of
every chain. Principled independent of SBC: γ there is **known a priori**, so there is nothing
to learn. Verified → 0 atoms, and the three target sds become matched (1.259 / 1.246 / 1.239).

**Result — hypothesis REFUTED (but the mechanism was confirmed):**

| model | prev | centre | next | pattern | r vs FB |
|-------|------|--------|------|---------|---------|
| with atom (`bayesflow_posterior_ATOM.keras`) | +0.0067 | +0.0102 | −0.0066 | **mixed signs** | 0.9990 |
| atom-free (current) | −0.0117 | −0.0088 | −0.0048 | **all negative** | 0.9984 |

Removing the atom **eliminated the sign asymmetry exactly as predicted** — so the atom *was*
responsible for the asymmetry — but the **miscalibration persists**, now uniform, at comparable
magnitude (|bias/sd| 0.09–0.23 vs 0.12–0.20). **SBC still fails.**

**Decision:** keep the atom-free model. It is principled, and the small r drop (0.9990 → 0.9984)
is expected: it now *extrapolates* at positions 0–1, where it deliberately never trains.

**Remaining suspect for SBC:** the near-degenerate 3-D target (dim-correlations 0.89–0.96) against
a *coupling* flow, which splits dimensions. A uniform same-sign bias across all dims is consistent
with that. Untested candidate fixes: reduce/decorrelate the target, or swap to FlowMatching.

### Fix 5 — one canonical `num_samples`
`NUM_SAMPLES = 300` now lives in `train_bayesflow.py`; every eval script imports it (was
300/500/50 drift). Justification: posterior sd ≈ 0.01 (probability), so MC error on the mean is
0.01/√300 ≈ **0.0006** — an order of magnitude below the MAE-to-FB (0.007). 50 was never *wrong*
(MC error 0.0014), just inconsistent.

### Fix 6 — `make_figures` now uses the FULL held-out block
500 chains (was 80) → **125,311 residues, r = 0.9984, MAE = 0.0079** — now *identical* to the
training-validation number. The two previously disagreed purely through sampling.

### Fix 7 — bootstrap CIs on insulin (residue resampling, 4,000 draws)

| chain | AUC | 95% CI | reading |
|-------|-----|--------|---------|
| B / D | 0.981 | **[0.93 – 1.00]** | robust — excludes chance |
| A | 0.528 | **[0.25 – 0.79]** | **uninformative** — straddles chance |
| C | 0.593 | [0.31 – 0.84] | uninformative |

> **This changes a conclusion.** The B-chain success is real. The A-chain "failure at chance"
> is **not statistically claimable** from 21 residues — the CI spans 0.25–0.79. The disulfide/
> cysteine mechanism remains the right *explanation*, but it must be presented as a hypothesis
> consistent with the data, **not** as a demonstrated result. Say "consistent with", not "shows".

### Other diagnostics bug fixed
`diagnostics.py` hard-coded `train_bf2.log`; it now auto-selects the newest `train_bf*.log`, so
the convergence curve always belongs to the model being diagnosed (it would otherwise have paired
the new model's SBC with the old run's loss).

---

## 17. Code review — defects found and fixed (2026-07-17)

### 🚩 REAL BUG: the majority-class baseline was computed wrongly (biased LOW)
`compare_metrics.py` computed the PISCES baseline as `max(mean(hf), 1-mean(hf))`. For a
**per-chain** comparison the correct quantity is `mean_i(max(hf_i, 1-hf_i))`. `max()` is convex,
so by **Jensen's inequality E[max] >= max(E)** — the old formula is guaranteed to understate.

| baseline | value |
|----------|-------|
| old (wrong) | 0.678 |
| **correct (per-chain mean)** | **0.730** |
| our per-chain accuracy | 0.743 |

We still beat it — but by **+0.013, not +0.065**. The bug made our accuracy look ~5x better than
it is. Fixed in `real_row()`; baselines are now stored directly in `ROWS` rather than re-derived
(the caller no longer re-applies `max()`).

### Other issues fixed
- **Stale docstring** in `insulin_eval.py` still advertised "default PDB 1A7F" and
  `insulin_prediction.png` after the wild-type switch → corrected, and now documents the mutant
  caveat + the CI/baseline reporting.
- **`diagnostics.py` hard-coded `train_bf2.log`** → auto-selects the newest `train_bf*.log`, so
  the convergence curve always belongs to the model being diagnosed. (Would otherwise have paired
  the new model's SBC with the old run's loss.)
- **Dead code**: unused `build_approximator` import (diagnostics), unused `n_skip_degenerate`
  (eval_real), unused `OUT_FIG` and `INSULIN_CI` constants → removed.
- **Duplication**: `eval_real.py` re-implemented `encode()` inline as `[AA.index(c) for c in seq]`
  (O(20) per residue) → now reuses `encode()`.

### Known remaining issues (not fixed)
- **`compare_metrics.ROWS` is hardcoded.** The simulated/insulin rows are hand-copied from other
  runs; only the PISCES row is computed from `real_eval_per_chain.csv`. If a model is retrained
  and ROWS is not updated, the table silently reports stale numbers. **This has already bitten
  twice.** Proper fix: have each eval script emit a small JSON/CSV that `compare_metrics` reads.
- **`load_approx()` is duplicated** in three scripts — harmless but should live in one module.
- **Boundary approximation**: in `build_training_data`, `g_next` at the last residue duplicates
  itself (no real "next"), as `g_prev` did at residue 0 before `SKIP_START`. Affects ~1 window
  per chain; undocumented until now.

---

## 18. FULL AUDIT — all known issues & red flags (2026-07-17)

Severity: 🔴 correctness / misleading result · 🟠 methodological · 🟡 reproducibility/quality.
Items marked ✅ are already fixed; others are open.

### 🔴 Correctness / potentially misleading
1. ✅ **Majority baseline computed wrong** (`max(mean)` vs `mean(max)`, Jensen). Was 0.678, truth
   0.730; overstated our margin 5×. Fixed §17.
2. 🔴 **Two AUCs on different sets, reported side by side.** `eval_real` per-chain AUC uses only
   the 8,308 chains with both classes; pooled AUC uses all 8,994 chains' residues. They are not
   the same population and the output prints them adjacently with no note.
3. 🔴 **Mean accuracy includes 686 single-class chains (7.6%).** An all-"other" chain scores
   accuracy = fraction-predicted-other (often ~1.0 since the model rarely fires), mechanically
   inflating the 0.743 mean. AUC excludes them; accuracy does not — inconsistent.
4. ✅ **NON-ISSUE (I was wrong).** I originally flagged that forcing P(helix)=0 at residue 0
   (startprob=[0,1]) would be wrong whenever a real chain's first residue is truly helix. But
   **0 of 393,732 real chains start with H** — exactly zero, in both PISCES and ss.cleaned. This
   is a structural fact: DSSP only calls a residue `H` if it is part of a run of consecutive
   H-bonded residues (the i→i+4 pattern), and the N-terminal residue has no predecessor to
   complete it, so the first residue is *never* `H`. The "always start in other" assumption is
   therefore empirically exact, and the forced zero is **correct**, not an error. Downgraded from
   🔴 to non-issue.

### 🟠 Methodological
5. 🟠 **`compare_metrics.ROWS` is hardcoded** (only PISCES row is computed). Silent staleness after
   any retrain — has already bitten twice this session. Highest-value open fix. (§17)
6. 🟠 **Pooled AUC conflates within-chain and between-chain ranking.** A per-chain baseline offset
   can inflate it; this is why pooled 0.77 > per-chain 0.75. Reported without this caveat.
7. 🟠 **Domain shift on real data.** The model trains on HMM-*emitted* windows (residues i.i.d.
   given state); real protein windows have motifs and long-range correlations the HMM never
   produces. So at real-eval time the 651-dim conditioning windows are OUT OF DISTRIBUTION. That
   ~0.75 AUC transfers at all is due to marginal AA propensities; the joint window statistics
   differ. The real-data numbers are cross-distribution, not held-out-in-distribution.
8. 🟠 **SBC fails** (~0.2 posterior-SD bias). Clip-atom hypothesis refuted by retrain. Uncertainty
   is not fully honest. (§13, §16)
9. 🟠 **Near-degenerate 3-D target** (dim-corr 0.89–0.96) vs a coupling flow — suspected SBC cause.
10. 🟠 **`SKIP_START=2` side effect.** The model now never trains on positions 0–1, so it
    *extrapolates* there at inference (real proteins included). Trade-off of the atom fix.
11. 🟠 **No summary network** (deferred; canonical BayesFlow component missing). (§15)
12. 🟠 **Insulin rests on 2 chains**; A-chain CI [0.25–0.79] straddles chance — the "A-chain fails"
    story is a *hypothesis consistent with* the data, not a demonstrated result. (§14)
13. 🟠 **"r = 0.999" measures fidelity to FB, not correctness.** Easy to misread as "99.9%
    accurate." FB is truth only under the (misspecified-on-real-data) model.

### 🟡 Reproducibility / code quality
14. 🟡 **CSV↔NPZ alignment relies on row order + equal count, no checksum.** Currently verified
    aligned (diff 3e-8 = float32), but regenerating one without the other would silently corrupt
    targets. Add a hash/length-vector check in `load_chains`.
15. 🟡 **Targets stored float32** in the npz while FB recomputes float64 — ~3e-8 lossy. Harmless
    for training, but means `fb_targets.npz` is not bit-exact to `forward_backward.py`.
16. 🟡 **`eval_real` default `--limit 1000`, not all.** Top-of-file chains are easier (gave 0.777
    vs true 0.754). Easy to accidentally report a non-representative subset.
17. 🟡 **Numbers hand-propagated** into deck/README/notebook/ROWS — same drift class as #5.
18. 🟡 ✅ `load_approx` duplicated in 3 files; dead code; `encode` re-implemented — partly cleaned
    (§17); `load_approx` dedup still open.
19. 🟡 **Undocumented boundary approximation**: `g_next` self-duplicates at the last residue
    (mirror of the `g_prev` issue `SKIP_START` fixed at the front). ~1 window/chain.

### What is actually SOLID (for balance)
- Train/val split is genuinely leak-free (disjoint chains + sequence-overlap assert; window-level
  overlap negligible at 20^31).
- CSV↔NPZ targets verified aligned to float32.
- Simulator recovers TRANS/EMIT to <0.001 and the given tables match real data (r≈0.99).
- BayesFlow faithfully reproduces FB in-distribution (r=0.998), confirmed by recovery + contraction.
- num_samples sensitivity is negligible (AUC stable to ~0.001 across 10×).

---

## 19. Code-review fixes applied (2026-07-17, round 2)

All items from the §18 audit that were fixable in-code are now done:

- **#2/#3 accuracy consistency** — `eval_real` now averages per-chain AUC *and* accuracy over the
  SAME both-class set (8,308 chains); single-class chains are counted and excluded from both. The
  PISCES row is now **acc 0.728 vs baseline 0.708** (was an inconsistent 0.743 vs 0.730 across
  different populations).
- **#5/#17 hardcoded ROWS eliminated** — `compare_metrics` is now fully data-driven: it reads
  `metrics_sim.json` (make_figures), `metrics_insulin_1MSO.json` (insulin_eval) and
  `metrics_real.json` / `real_eval_per_chain.csv` (eval_real). No number is hand-copied; a retrain
  cannot silently produce a stale table. Verified: every printed value traces to a metrics file.
- **#6 pooled-AUC caveat** — printed and JSON-noted that pooled AUC mixes within/between-chain
  ranking and isn't comparable to the per-chain mean.
- **#12 CSV<->NPZ alignment assert** — `load_chains` now raises on any length mismatch between the
  CSV seq, its `len` column, and the NPZ `lengths[i]`.
- **#13 subset footgun** — `eval_real` default is now `--limit 0` (all chains); a subset prints a
  loud WARNING.
- **#14 load_approx dedup** — single `load_model()` in `train_bayesflow`; the three eval scripts
  import it. (diagnostics keeps its own top-level load; harmless.)
- **#4 residue-0 / #19 boundary** — documented in code (eval_real docstring; build_training_data
  comment). Not "fixed" (they are model properties), but no longer silent.
- **num_samples** unified at 50 (matches the sensitivity analysis and every regenerated output).

Still open (require scope/architecture changes, documented not fixed): #4 model-forced residue-0,
#7 train/test domain shift, #8/#9 SBC + degenerate target, #10 SKIP_START extrapolation,
#11 no summary network. See §18.

---

## 20. Improvements for the talk (2026-07-17, round 3)

### (2) Explicit framing of the estimand  [WRITE THIS ON THE SLIDE]
> **What posterior are we estimating, and why a window?** Given the *full* amino-acid
> sequence, the Forward-Backward marginal gamma_t is a *deterministic* function of the data
> (the HMM parameters are fixed), so p(gamma_t | full sequence) is a point mass -- there is
> no density for a posterior estimator to learn. We therefore condition on a *local* 31-residue
> window: residues outside it are integrated out, which makes gamma_t genuinely uncertain and
> yields a non-degenerate estimand p(gamma_t | window) that BayesFlow can model. The window is
> justified by fast mixing (second eigenvalue 0.85 -> 0.85^15 < 0.10 influence beyond +/-15).
> Trade-off: a larger, more informative window sharpens the point estimate (r=0.998) but pushes
> the estimand toward the deterministic limit, which is exactly why calibration (SBC) is hard.

### (4) Real-data evidence: PISCES is the headline, insulin is the illustration
The brief names insulin only as an *example* ("e.g. ... human insulin"). Insulin alone is weak
evidence (2 usable chains; A-chain AUC CI [0.25, 0.79] straddles chance). The honest real-data
result is the **full PISCES set**:

| Real-data evidence | n | result |
|--------------------|---|--------|
| **PISCES (headline)** | **8,308 chains** | per-chain AUC **0.753, 95% CI [0.750, 0.756]**; 96.7% of chains beat chance; 73% beat 0.7; median 0.764 |
| Insulin B (illustration) | 1 chain | AUC 0.98 [0.93, 1.00] -- a clean textbook helix |
| Insulin A (illustration) | 1 chain | AUC 0.52 [0.25, 0.79] -- uninformative alone; motivates the disulfide story |

Presentation order: lead with the PISCES distribution (tight CI, thousands of chains) as the
quantitative claim; show insulin as the single-protein visual the brief asks for, explicitly
flagged as illustrative, not the evidence base.

---

## 21. CALIBRATION RESOLVED — explained AND demonstrated (audit #1, round 3)

We ran the professor's suggested experiments and one decisive follow-up. Verdict: SBC fails
on the reported (large-window) models, and we now know exactly why — and can toggle it.

### Experiments
| model | window | flow | target | post.sd | point r | SBC |
|-------|--------|------|--------|---------|---------|-----|
| main | 31 | Coupling | 3-D | 0.05 | 0.998 | fail (over-confident, +bias) |
| calibrated | 31 | **FlowMatching** + CNN summary | **1-D** | 0.06 | 0.998 | fail (under-confident, −bias) |
| **small-window** | **7** | FlowMatching + CNN summary | 1-D | **0.53** | 0.895 | **PASS (rank ECDF inside the band)** |

### What this proves
1. **The miscalibration is architecture-invariant.** Coupling+3-D and FlowMatching+summary+1-D
   — completely different — both fail at the **same ~0.15-posterior-SD magnitude** (opposite
   signs). So the cause is the *estimand*, not the network.
2. **The cause is a near-deterministic estimand.** At W=31 the window carries >90% of the
   information (λ₂¹⁵≈0.09), so γ|window is almost a point mass (post.sd≈0.06). A flow cannot
   perfectly hit a near-delta, and SBC — hypersensitive when the true spread is tiny — flags the
   residual.
3. **It is controllable (the decisive demo).** Shrink the window to W=7 → the window is no longer
   near-sufficient → γ|window has genuine spread (post.sd 0.06→0.53) → **SBC passes**, at the cost
   of point accuracy (logit r 0.998→0.895). Figure: `calibration_tradeoff.png` (same FlowMatching
   model, only the window differs).

### The honest conclusion for the talk
> Our headline model is optimised for the point estimate (near-perfect r), which makes the
> estimand near-deterministic and therefore hard to calibrate — SBC catches a ~0.002 (probability)
> residual. This is a **point-accuracy vs calibration trade-off tunable by window size**: a smaller
> window yields an honest, SBC-passing posterior with a weaker point estimate. We demonstrated both
> ends. This is a property of the *estimand*, not a bug.

Delivered en route: **audit #3 (summary network)** — the `TimeSeriesNetwork` (Conv+GRU over the
window) is now a working, canonical BayesFlow component (`train_calibrated.py`).

---

## 22. NEW FINDING — the accuracy ↔ calibration Pareto frontier (experiment_pareto.py)

Generalises the single W=31-vs-W=7 calibration comparison (§21) into a full window-size
sweep, measuring BOTH point accuracy and calibration on held-out chains. Model at each W:
FlowMatching + Conv/GRU summary net + 1-D centre target, 15 epochs, 8k chains.

| W | r (prob) | MAE | SBC-KS (↓ better) | post.sd |
|---|----------|-----|-------------------|---------|
| 7  | 0.873 | 0.084 | 0.063 | 0.585 |
| 11 | 0.946 | 0.053 | 0.052 | 0.365 |
| **15** | **0.976** | 0.036 | **0.047 (best)** | 0.229 |
| 21 | 0.992 | 0.021 | 0.118 | 0.141 |
| 31 | 0.998 | 0.011 | 0.146 | 0.072 |

(SBC-KS = max |rank-ECDF − uniform|, a scalar calibration metric; 0 = perfectly calibrated.)

### The insight
- **Point accuracy rises monotonically** with window size (r 0.87 → 0.998): a bigger window
  sees more of the chain, so γ|window approaches the deterministic full-chain γ.
- **Calibration is U-shaped**: best at **W ≈ 11–15**, then degrades **2–3×** at W = 21, 31. As the
  window becomes near-sufficient the estimand becomes near-deterministic (post.sd 0.585 → 0.072),
  and a flow cannot calibrate a near-point-mass (§21).
- **There is a knee at W ≈ 15**: near-top accuracy (r = 0.976) AND the best calibration. The
  headline **W = 31 model is over-tuned for point accuracy** — it buys +0.02 r for +0.10 KS, a
  bad trade. W ≈ 15 is the near-Pareto-optimal operating point.

### Why this matters
This turns "SBC fails" from an apology into a *controlled, quantified* result: the trade-off is
a **design knob**, and we can pick the operating point. Reproduce: `python experiment_pareto.py`
→ `outputs/metrics_pareto.json`, figure `figures/pareto_frontier.png` (deck slide 13).
