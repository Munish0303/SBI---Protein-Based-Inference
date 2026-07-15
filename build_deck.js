const pptxgen = require("pptxgenjs");
const P = new pptxgen();
P.layout = "LAYOUT_WIDE";           // 13.33 x 7.5 in
P.author = "SBI project";
P.title = "Predicting alpha-helix secondary structure with a two-state HMM and BayesFlow";

const IMG = "D:/SBI";
const DARK="0C3B2E", TEAL="1D9E75", DEEP="0F6E56", MINT="8FE3C6",
      INK="1A2E29", MUTE="5B6B66", TINT="EAF5F0", WHITE="FFFFFF", WM="13463A";
const HEAD="Cambria", BODY="Calibri";

function bullets(items, size, color){
  return items.map((t,i)=>({text:t, options:{bullet:{indent:14}, breakLine:true,
    fontSize:size, color:color||INK, paraSpaceAfter:8}}));
}
function badge(s, num, name){
  s.addShape(P.shapes.OVAL,{x:0.5,y:0.34,w:0.52,h:0.52,fill:{color:TEAL}});
  s.addText(String(num),{x:0.5,y:0.34,w:0.52,h:0.52,align:"center",valign:"middle",
    fontFace:HEAD,fontSize:20,bold:true,color:WHITE,margin:0});
  s.addText(name.toUpperCase(),{x:1.15,y:0.36,w:9,h:0.48,valign:"middle",
    fontFace:BODY,fontSize:13,bold:true,color:TEAL,charSpacing:2,margin:0});
}
function title(s, t){
  s.addText(t,{x:0.5,y:0.98,w:12.3,h:0.8,fontFace:HEAD,fontSize:27,bold:true,color:DEEP,margin:0});
}
function caption(s, t, x, y, w){
  s.addText(t,{x:x,y:y,w:w,h:0.5,fontFace:BODY,fontSize:11,italic:true,color:MUTE,align:"center",margin:0});
}
function content(num,name,t){
  const s=P.addSlide(); s.background={color:WHITE}; badge(s,num,name); title(s,t); return s;
}

/* ---------- 1. TITLE ---------- */
let s=P.addSlide(); s.background={color:DARK};
s.addText("α",{x:9.2,y:0.2,w:4,h:7,fontFace:HEAD,fontSize:340,bold:true,color:WM,align:"center",valign:"middle",margin:0});
s.addText("SIMULATION-BASED INFERENCE  ·  TU DORTMUND",{x:0.7,y:1.15,w:11,h:0.4,fontFace:BODY,fontSize:14,bold:true,color:MINT,charSpacing:3,margin:0});
s.addText("Predicting α-helix secondary structure",{x:0.7,y:1.9,w:11.5,h:1.0,fontFace:HEAD,fontSize:40,bold:true,color:WHITE,margin:0});
s.addText("A two-state Hidden Markov Model fit with amortized BayesFlow inference",
  {x:0.7,y:3.0,w:11,h:0.7,fontFace:HEAD,fontSize:22,italic:true,color:MINT,margin:0});
s.addShape(P.shapes.LINE,{x:0.75,y:4.15,w:3.2,h:0,line:{color:TEAL,width:2}});
s.addText([
  {text:"Group [ number ]",options:{breakLine:true,fontSize:18,bold:true,color:WHITE,paraSpaceAfter:6}},
  {text:"[ Member 1 ]   ·   [ Member 2 ]   ·   [ Member 3 ]",options:{fontSize:15,color:MINT}}
],{x:0.75,y:4.4,w:11,h:1.2,fontFace:BODY,margin:0});
s.addNotes("Title slide. State the goal in one line: predict per-residue alpha-helix probability from an amino-acid sequence, using a fixed two-state HMM and a BayesFlow amortized posterior. Replace the group number and member names.");

/* ---------- 2. ROADMAP ---------- */
s=P.addSlide(); s.background={color:WHITE};
s.addText("Roadmap",{x:0.5,y:0.55,w:8,h:0.8,fontFace:HEAD,fontSize:30,bold:true,color:DEEP,margin:0});
const road=[
  ["1","The question","Predict α-helix vs other from sequence"],
  ["2","Forward–Backward","Exact per-residue posterior as training targets"],
  ["3","BayesFlow","Amortized neural posterior via a sliding window"],
  ["4","Insulin & real evaluation","Held-out real proteins vs ground truth"],
  ["5","Comparison & results","BayesFlow vs exact Forward–Backward"],
];
road.forEach((r,i)=>{
  const y=1.7+i*1.02;
  s.addShape(P.shapes.OVAL,{x:0.9,y:y,w:0.62,h:0.62,fill:{color:TEAL}});
  s.addText(r[0],{x:0.9,y:y,w:0.62,h:0.62,align:"center",valign:"middle",fontFace:HEAD,fontSize:22,bold:true,color:WHITE,margin:0});
  s.addText([{text:r[1]+"   ",options:{fontSize:19,bold:true,color:INK}},
             {text:r[2],options:{fontSize:14,color:MUTE}}],
    {x:1.8,y:y,w:11,h:0.62,valign:"middle",fontFace:BODY,margin:0});
});
s.addNotes("Five sections. Keep this to ~15 seconds — it just sets expectations.");

/* ---------- 3. [1] The question ---------- */
s=content(1,"The question","From amino-acid sequence to α-helix");
s.addText(bullets([
  "Proteins fold; the local shape is the secondary structure (helix / sheet / coil).",
  "We predict one thing: is each residue an α-helix, or 'other'?",
  "Solving 3-D structure needs X-ray / NMR — expensive; we want it from sequence.",
  "Our statistical model: a two-state Hidden Markov Model."
],15),{x:0.55,y:2.0,w:6.7,h:4.2,fontFace:BODY,valign:"top"});
// right visual: state pills + mapping note
s.addShape(P.shapes.ROUNDED_RECTANGLE,{x:7.7,y:2.15,w:5.0,h:0.95,fill:{color:TEAL},rectRadius:0.12});
s.addText("α-helix  =  state “H”",{x:7.7,y:2.15,w:5.0,h:0.95,align:"center",valign:"middle",fontFace:HEAD,fontSize:20,bold:true,color:WHITE,margin:0});
s.addShape(P.shapes.ROUNDED_RECTANGLE,{x:7.7,y:3.3,w:5.0,h:0.95,fill:{color:"D3D1C7"},rectRadius:0.12});
s.addText("“other”  =  E B G I T S C",{x:7.7,y:3.3,w:5.0,h:0.95,align:"center",valign:"middle",fontFace:HEAD,fontSize:18,bold:true,color:"2B2A26",margin:0});
s.addShape(P.shapes.ROUNDED_RECTANGLE,{x:7.7,y:4.5,w:5.0,h:1.4,fill:{color:TINT},rectRadius:0.1});
s.addText([{text:"Mapping decision\n",options:{fontSize:13,bold:true,color:DEEP,breakLine:true}},
  {text:"DSSP labels each residue with 8 states. We keep only H as helix (strict α-helix); the other 7 → other.",options:{fontSize:12.5,color:INK}}],
  {x:7.95,y:4.6,w:4.5,h:1.2,valign:"middle",fontFace:BODY,margin:0});
s.addNotes("Introduce the topic and the exact prediction target. Emphasize the modeling choice: strict H-only helix definition (cleaner than merging G and I).");

/* ---------- 4. [1] HMM model ---------- */
s=content(1,"The question","The generative model: a two-state HMM");
s.addImage({path:IMG+"/hmm_explainer/1_two_state_hmm.png",x:5.85,y:1.95,w:6.8,h:4.0});
caption(s,"Fig 1. Two-state HMM: start rule and transition probabilities.",5.85,6.0,6.8);
s.addText(bullets([
  "The chain always starts in “other”.",
  "Transitions: helix→helix 0.90, other→helix 0.05.",
  "So helices form ~10-residue runs, and are started rarely.",
  "Realistic: real data gives 0.912 and 0.041 (see next slide)."
],14.5),{x:0.55,y:2.05,w:5.0,h:4.2,fontFace:BODY,valign:"top"});
s.addNotes("Explain the statistical model. Point at the self-loops (0.90 / 0.95) and the cross arrows (0.05 / 0.10). The 0.90 stay-probability is what makes helices come in runs.");

/* ---------- 5. [1] Emissions + realism ---------- */
s=content(1,"The question","Emissions — and are they realistic?");
s.addImage({path:IMG+"/hmm_explainer/2_emission_probabilities.png",x:2.95,y:1.85,w:7.45,h:3.46});
caption(s,"Fig 2. Per-state amino-acid emission probabilities (α-helix vs other).",2.95,5.35,7.45);
s.addShape(P.shapes.ROUNDED_RECTANGLE,{x:0.55,y:5.9,w:12.2,h:1.15,fill:{color:TINT},rectRadius:0.1});
s.addText([{text:"Yes — checked against real PISCES data:   ",options:{bold:true,color:DEEP,fontSize:13.5}},
  {text:"emissions match at r = 0.99 (max 0.9 pp off); transitions match too (P(h→h) 0.912 vs 0.90). Both tables are empirically well-founded.",options:{fontSize:13,color:INK}}],
  {x:0.85,y:5.9,w:11.6,h:1.15,valign:"middle",fontFace:BODY,margin:0});
s.addNotes("Each state emits amino acids with its own propensities (helix loves A,L,E; breakers G,P favor other). We measured the empirical P(amino acid | DSSP H vs non-H) from real PISCES and overlaid it on the given tables: correlation 0.992 (helix) / 0.982 (other), max deviation 0.9 pp. Transitions match too. So both halves of the model are grounded in data. (The old 'transitions-only' evidence was a mismatch with this slide's title — now fixed.)");

/* ---------- 6. [2] Simulator + FB ---------- */
s=content(2,"Forward–Backward","Notebooks 1–2: simulate, then get exact targets");
s.addImage({path:IMG+"/hmm_explainer/3_bayesflow_pipeline.png",x:6.35,y:1.85,w:6.25,h:4.0});
caption(s,"Fig 3. Full pipeline: simulator → Forward–Backward → BayesFlow → evaluation.",6.35,5.9,6.25);
s.addText(bullets([
  "Simulate 100,000 chains; lengths sampled from real PISCES (median 208).",
  "Run Forward–Backward (hmmlearn) → exact P(helix) per residue = training targets.",
  "24.3M residues — the exact Bayesian “answer key”.",
  "Checks: mean P 0.324; residue-0 = 0; recovers hidden states at the Bayes ceiling (AUC 0.79)."
],14),{x:0.55,y:2.0,w:5.6,h:4.3,fontFace:BODY,valign:"top"});
s.addNotes("Forward-Backward gives the exact posterior P(helix) for each residue given the whole chain. hmmlearn does this. These become the targets BayesFlow learns to emulate. The 0.79 AUC ceiling comes from overlapping emission tables — even exact inference can't separate helix/other perfectly.");

/* ---------- 7. [3] BayesFlow windowing ---------- */
s=content(3,"BayesFlow","Turning a variable-length chain into a fixed input");
s.addImage({path:IMG+"/restart/bayesflow_window_schematic.png",x:1.85,y:1.8,w:9.6,h:3.8});
caption(s,"Fig 4. Sliding 31-residue window → one-hot encoding → CouplingFlow posterior.",1.85,5.6,9.6);
s.addText(bullets([
  "A flow needs a fixed-size target → predict one residue from a fixed 31-residue window (±15), slid along the chain.",
  "Encode the window as one-hot: 31 × 21 channels = 20 amino acids + 1 padding (for chain ends).",
  "The window is only local → the answer stays uncertain → a genuine posterior (mean ± std), not a point estimate."
],12.5),{x:0.55,y:6.0,w:12.2,h:1.4,fontFace:BODY,valign:"top"});
s.addNotes("This is the key modeling idea. Explain: normalizing flows need fixed dimensions, sequences vary, so we amortize one residue at a time through a fixed window and slide it. One-hot (not integer) because amino acids are categorical; padding channel marks chain edges. Local view => stochastic map => non-degenerate posterior, which is exactly what BayesFlow captures.");

/* ---------- 8. [3] Training setup ---------- */
s=content(3,"BayesFlow","How we trained it");
const tbl=[
  [{text:"Component",options:{bold:true,color:WHITE,fill:{color:DEEP},fontSize:13}},
   {text:"Choice  ·  why",options:{bold:true,color:WHITE,fill:{color:DEEP},fontSize:13}}],
  ["Inference network","CouplingFlow, depth 6, 256×256 subnets"],
  ["Target θ (3-D)","logit P(helix) at [prev, centre, next]"],
  ["Conditions","flattened 31×21 = 651 one-hot window"],
  ["Training data","300,000 windows from 15,000 chains"],
  ["Optimizer / epochs","Adam · 30 epochs · batch 512"],
];
s.addTable(tbl,{x:0.55,y:2.05,w:7.5,rowH:0.62,fontFace:BODY,fontSize:13,color:INK,
  valign:"middle",border:{pt:0.5,color:"D8E4DF"},fill:{color:WHITE},
  colW:[2.5,5.0]});
s.addShape(P.shapes.ROUNDED_RECTANGLE,{x:8.4,y:2.05,w:4.35,h:1.75,fill:{color:TINT},rectRadius:0.1});
s.addText([{text:"No leakage (verified)\n",options:{bold:true,color:DEEP,fontSize:14,breakLine:true}},
  {text:"Train = front block [0, 15000); validation = tail block [99500, 100000). Disjoint indices AND 0 shared sequences — asserted at runtime.",options:{fontSize:12,color:INK}}],
  {x:8.65,y:2.2,w:3.9,h:1.5,valign:"top",fontFace:BODY,margin:0});
s.addShape(P.shapes.ROUNDED_RECTANGLE,{x:8.4,y:4.0,w:4.35,h:1.65,fill:{color:DEEP},rectRadius:0.1});
s.addText([{text:"Why a window?\n",options:{bold:true,color:MINT,fontSize:14,breakLine:true}},
  {text:"One trained model does inference for a chain of ANY length — just slide the window. Amortized: no re-fitting per protein.",options:{fontSize:12,color:WHITE}}],
  {x:8.65,y:4.15,w:3.9,h:1.4,valign:"top",fontFace:BODY,margin:0});
s.addNotes("Walk through the table. Stress the disjoint train/val split — a natural exam question is 'how do you know there's no leakage?' Answer: front vs tail blocks, plus a runtime assert that no validation sequence appears in training.");

/* ---------- 9. [4] Insulin (wild-type 1MSO) ---------- */
s=content(4,"Insulin & real evaluation","Held-out real protein: wild-type insulin (1MSO)");
s.addImage({path:IMG+"/restart/insulin_1MSO_slide.png",x:5.35,y:1.95,w:7.2,h:4.19});
caption(s,"Fig 5. Insulin P(helix): BayesFlow (green) vs exact FB (black); gold = true helix.",5.35,6.25,7.2);
s.addText(bullets([
  "Insulin + true DSSP labels from the dataset; model trained ONLY on simulated data (unseen).",
  "B-chain helix is propensity-driven (L,V,E,A) → nailed, AUC 0.98.",
  "A-chain helix is disulfide-stabilised & Cys-rich; C disfavours helix in our table → model predicts “not helix” → AUC 0.52 (chance).",
  "A sequence-only HMM cannot see 3-D disulfide bonds."
],13.5),{x:0.5,y:2.0,w:4.7,h:4.5,fontFace:BODY,valign:"top"});
s.addNotes("CORRECTION from an earlier version: 1A7F is a MUTANT insulin (B16E, B24G, des-B30) whose sparse A-chain annotation inflated the A-chain AUC to 0.97. On wild-type 1MSO the A-chain is at chance (0.52). The B-chain (0.98) is genuine. The A-chain failure is scientifically informative: its N-terminal helix is stabilised by disulfide bonds, and cysteine is helix-disfavouring in the emission table, so a propensity HMM predicts non-helix exactly where the helix is. Insulin rests on only 2-4 chains — illustrative, not a robust benchmark.");

/* ---------- 10. [4] Real proteome ---------- */
s=content(4,"Insulin & real evaluation","All real proteins (PISCES)");
s.addImage({path:IMG+"/restart/real_eval_auc_hist.png",x:6.3,y:2.0,w:6.2,h:3.99});
caption(s,"Fig 6. Per-chain AUC over 8,308 real chains (those with both classes).",6.3,6.05,6.2);
s.addText(bullets([
  "Scored all 8,994 standard-AA chains (2.2M residues).",
  "Per-chain AUC 0.754; BayesFlow = exact FB.",
  "A simple HMM ranks helices meaningfully across the messy real proteome.",
  "AUC > accuracy: the HMM isn’t calibrated to reality → use ranking."
],14),{x:0.5,y:2.05,w:5.5,h:4.3,fontFace:BODY,valign:"top"});
s.addNotes("Generalization test on the whole culled real dataset. The point: BayesFlow tracks exact FB even far out of distribution, and the honest real-world number is ~0.75 AUC for this deliberately simple model.");

/* ---------- 11. [5] BayesFlow ~ FB ---------- */
s=content(5,"Comparison & results","Result 1: the neural posterior reproduces exact inference");
s.addImage({path:IMG+"/restart/validation_figure.png",x:1.35,y:1.8,w:10.6,h:4.08});
caption(s,"Fig 7. Left: BayesFlow mean vs exact FB (r=0.999). Right: an example chain with the ±1 std band.",1.35,5.9,10.6);
s.addText([{text:"Held-out simulated chains: correlation 0.999, MAE 0.007.  ",options:{bold:true,color:DEEP,fontSize:14}},
  {text:"The fast neural posterior is interchangeable with slow HMM Forward–Backward.",options:{fontSize:14,color:INK}}],
  {x:0.7,y:6.5,w:12,h:0.7,fontFace:BODY,align:"center",margin:0});
s.addNotes("Headline result 1: BayesFlow reproduces the exact posterior almost perfectly. The scatter hugging the diagonal is the money plot.");

/* ---------- 12. [5] SBI diagnostics ---------- */
s=content(5,"Comparison & results","SBI diagnostics: convergence, calibration, contraction");
s.addImage({path:IMG+"/restart/diag_loss.png",x:0.55,y:2.0,w:4.35,h:2.61});
caption(s,"Fig 8a. Training loss.",0.55,4.6,4.35);
s.addImage({path:IMG+"/restart/diag_sbc_ecdf.png",x:5.25,y:2.05,w:7.45,h:2.44});
caption(s,"Fig 8b. Simulation-based calibration (SBC) rank ECDF, per target dim.",5.25,4.55,7.45);
s.addText(bullets([
  "Convergence: loss −3.8 → −7.9, plateaued.  Recovery r = 0.999 and posterior contraction 0.999 — the estimate is excellent and tight.",
  "SBC (right): the rank ECDF slightly exits the 95% band → a small location bias (~0.15 posterior SD ≈ 0.002 in probability). The uncertainty band is not perfectly calibrated.",
  "Honest limitation: point estimates are unaffected; likely cause is the near-degenerate 3-D target (dim-correlations 0.89–0.96) vs a coupling flow. Candidate fix left as future work."
],12.5),{x:0.55,y:5.05,w:12.2,h:2.2,fontFace:BODY,valign:"top"});
s.addNotes("This is the SBI-specific evidence the course asks for (convergence + calibration + contraction), and it's the most likely place questions land. Own the SBC result: the posterior is a superb point estimate (r=0.999) but its uncertainty is slightly mis-located — a ~0.15-SD bias that only a sensitive test like SBC catches. We diagnosed it: NOT the logit-clip atom at residue 0 (excluding it doesn't fix SBC); the more likely cause is that the 3 target dims are 0.89-0.96 correlated, which coupling flows handle poorly. Showing you found AND interpreted a calibration failure is worth more than hiding it.");

/* ---------- 13. [5] results table ---------- */
s=content(5,"Comparison & results","Results at a glance");
s.addImage({path:IMG+"/restart/comparison_table.png",x:1.05,y:1.95,w:11.2,h:2.80});
caption(s,"Table 1. AUC & accuracy@0.5 vs the majority-class baseline — BayesFlow vs exact FB.",1.05,4.85,11.2);
s.addText(bullets([
  "BayesFlow ≈ exact FB in every setting — faithful amortization.",
  "Accuracy@0.5 only beats the trivial baseline on the calibrated (simulated / pooled real) settings; on insulin it ties or falls below it → AUC is the fair metric.",
  "FB is a ceiling on SIMULATED data only; on real proteins the HMM is misspecified, so a better model can beat it."
],13),{x:1.1,y:5.3,w:11.1,h:1.9,fontFace:BODY,valign:"top"});
s.addNotes("The two method columns are equal everywhere (faithful amortization). The added baseline column is the honest reading: on insulin the model never crosses P=0.5, so accuracy collapses to the baseline. Do NOT claim FB is a ceiling on real data — it isn't, because the HMM is misspecified there.");

/* ---------- 14. TL;DR ---------- */
s=P.addSlide(); s.background={color:DARK};
s.addText("α",{x:9.4,y:0.2,w:4,h:7,fontFace:HEAD,fontSize:340,bold:true,color:WM,align:"center",valign:"middle",margin:0});
s.addText("TL;DR — take-home message",{x:0.7,y:0.7,w:12,h:0.8,fontFace:HEAD,fontSize:30,bold:true,color:WHITE,margin:0});
s.addText([
  {text:"α-helix as a 2-state HMM whose transition AND emission tables match real proteins (r ≈ 0.99).",options:{bullet:{indent:14},breakLine:true,fontSize:16,color:WHITE,paraSpaceAfter:11}},
  {text:"BayesFlow amortized posterior trained on exact Forward–Backward targets via a sliding 31-residue window; it reproduces exact FB almost perfectly (r = 0.999).",options:{bullet:{indent:14},breakLine:true,fontSize:16,color:WHITE,paraSpaceAfter:11}},
  {text:"Diagnostics: excellent recovery & contraction, but SBC reveals a small (~0.15 SD) calibration bias — a real, owned limitation.",options:{bullet:{indent:14},breakLine:true,fontSize:16,color:WHITE,paraSpaceAfter:11}},
  {text:"On unseen real proteins it ranks propensity-driven helices well (insulin B-chain 0.98; PISCES 0.75) but fails where structure is 3-D-stabilised (insulin A-chain 0.52) — the price of a sequence-only model.",options:{bullet:{indent:14},fontSize:16,color:WHITE}}
],{x:0.75,y:1.7,w:11.6,h:3.7,fontFace:BODY,valign:"top"});
s.addShape(P.shapes.LINE,{x:0.8,y:5.55,w:3.2,h:0,line:{color:TEAL,width:2}});
s.addText([{text:"Group [ number ]  ·  [ Member names ]\n",options:{fontSize:15,bold:true,color:MINT,breakLine:true}},
  {text:"[ email ]@tu-dortmund.de   ·   Simulation-Based Inference, TU Dortmund",options:{fontSize:13,color:WHITE}}],
  {x:0.8,y:5.75,w:11.5,h:1.0,fontFace:BODY,margin:0});
s.addNotes("TL;DR. Land these four sentences. Be ready for questions on: what amortization buys you, why a normalizing flow, why AUC not accuracy, and the difference between Forward-Backward and Viterbi.");

P.writeFile({fileName:"D:/SBI/restart/SBI_presentation.pptx"}).then(f=>console.log("saved",f));
