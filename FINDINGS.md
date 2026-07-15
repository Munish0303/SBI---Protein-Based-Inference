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
- **Not the logit-clip atom.** γ = 0 *exactly* at residue 0 for every chain (start rule) →
  `logit` clipped to −6.907, a **point mass** a continuous flow cannot represent. It is the
  ONLY such atom (γ ≥ 0.0043 everywhere else in 24.3M residues). But excluding the start
  residues does **not** fix SBC (`diag_sbc_ecdf_nostart.png`) — so this is not the cause.
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
