# Splatter Image under a Gaussian budget — plan v2 (16 → 24 November)

*Same project as `project_overview.md` (pruning → learned importance → budget-aware fine-tuning), made tighter and stronger. What changed and why is in §1; the rest is the working plan.*

---

## Status — 16 November, evening

**Done**
- Level 1 on all 704 test cars, 12 criteria × 6 budgets + merging (`results/analysis704/`). Key results: 25 % is free (−0.004 ± 0.001 dB); at 10 % the free **hybrid** rank is best (21.32 dB, LPIPS 0.106, best worst-case angle −5.0 dB); opacity×area best PSNR at 5 %; iterative + balanced contribution oracle best at 1 % and +1.0 dB over plain contribution at 5 %; PhGS-style edge prior ≈ random; merge best PSNR at 6.25 % but blurry (LPIPS 0.176).
- Level 2 heads trained (lr 1e-3 chosen on val). Val top-10 % overlap with the oracle: head 0.58 vs opacity 0.49. Ablations: Gaussians-only input ≈ all inputs (0.589); RGB-only 0.409 (< opacity); ranking ≈ MSE; 20k ≈ 8k steps.
- Paper (CVPR 2026 template, `cvpr_paper/`): Abstract, Intro, Related Work, Preliminaries, Method, Setup written; Results for Level 1 written with real numbers, table and two figures.

**Running / next**
- ✅ Test evaluation of the heads done; Level 2 results written in the paper (head beats its teacher by +0.75 dB at 10 %, not the free hybrid rank).
- ⏳ Cost: storage + selection time done; rendering time / 360° video still needs `benchmark_cost.py`.
- **Level 3 (18 Nov): `finetune_budget.py` + `finetune.sbatch` written and CPU-tested.** Next: smoke test + lr check, then 4 runs on Taya's account (control, FT-10 learned, FT-10 hybrid, Group DRO) — see §3.
- Demos (§8), SRN Chairs (evaluation only), cost table.
- **Level 3b (new):** viewpoint-robust fine-tuning with Group DRO over angle bins (§9) — same patch and control run as Level 3.

---

## 0. What we already have (verified from the `taya/dev` branch)

| Result | Number | Source |
|---|---|---|
| Baseline, 704 test cars | PSNR **23.999**, SSIM 0.923, LPIPS 0.077 (paper: 24.00 / 0.92 / 0.078) | `baseline_cars_full_scores.txt` |
| Opacity sparsity | **74.4 %** of Gaussians have α < 0.001 | 20 cars |
| Opacity pruning, keep 25 % (4 096) | ΔPSNR **−0.004 dB** | 20 cars |
| Opacity pruning, keep 10 % (1 638) | ΔPSNR **−4.63 dB** (worst object −8.2 dB) | 20 cars |
| Render time, keep 25 % / 10 % | 1.03× / 1.18× faster (1.98 → 1.92 / 1.68 ms) | 5 cars |
| Quality vs viewpoint, 176 000 views | **U-shaped**, see below | `viewpoint_gap_full704.csv` |

Viewpoint gap on all 704 cars (computed from your CSV):

| Angle input→target | Views | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| 30–60° | 11 968 | 26.75 | 0.956 | 0.040 |
| 60–90° | 24 640 | 23.32 | 0.925 | 0.076 |
| **90–120°** | 59 136 | **23.17** | **0.913** | **0.089** |
| 120–150° | 76 032 | 24.37 | 0.924 | 0.075 |
| 150–180° | 4 224 | 25.14 | 0.936 | 0.064 |

→ The hardest views are the **side views (~90–120°)**, not the opposite view. Opposite views are easier, probably because cars are left/right and roughly front/back symmetric, so the model can "reuse" what it sees. This is a nice finding for the report, and it tells us *which* views to look at when pruning hurts.

---

## 1. What changes vs `project_overview.md`, and why

1. **The headline is a curve, not a single number.** "75 % pruning is free" is trivial: 74 % of Gaussians are already transparent. The real result is the **quality–budget curve** from 25 % down to 1 %: opacity loses 4.6 dB at 10 %. That 4.6 dB is the headroom that a better selector or a fine-tuned model can win back. Every method is judged by how far it pushes this curve up.
2. **Level 3 is no longer a stretch goal: it is the main trained contribution.** Budget-aware fine-tuning (decoder + importance head trained together through a hard top-k mask) is the experiment most likely to give a big positive result. Post-hoc pruning cannot change the remaining Gaussians, while a fine-tuned model can learn to *compensate* (bigger, more opaque, better placed Gaussians). This also satisfies "we must train something substantial".
3. **Level 2 stays, but only as distillation into a ranking (variant A; B if cheap).** Variant C (end-to-end mask) *is* Level 3, so it isn't a separate thing anymore.
4. **Cut or shrink:** separate profiling at 3 resolutions → one table (128² vs 512², single frame vs 100-frame 360° video, memory). Merging stays because it costs no training and is already implemented.
5. **The evaluation is 30× cheaper.** One U-Net forward per object; all criteria × budgets reuse it; 25 evaluation views (odd indices) and 25 disjoint scoring views (even indices) per object. The old way (one 45-min job per ratio) would need >30 GPU-hours for the Level 1 table.
6. **OOM fixed.** `SRNDataset` caches all images of every object it loads and never frees them (~49 MB/object → ~34 GB for 704 cars). That is why `eval.py` jobs died at 16/32 GB. The new scripts free the cache after each object.

### Honest motivation (use this wording)

> Pixel-aligned feed-forward models allocate Gaussians by **image resolution, not by object complexity**: every object gets 16 384 Gaussians (1.5 MB in fp32), and 74 % of them are invisible. We ask how small the representation can get, which Gaussians actually matter (especially for the hallucinated, unseen side), and whether a model can be **trained to reconstruct under a budget**. Benefits: storage and transmission (4–20× smaller assets), memory when many objects or views are merged (the multi-view variant unions N × 16 384), and rendering cost at high resolution or for many frames. **We do not claim a speed-up at 128²**: there the rasteriser cost is dominated by fixed overhead (1.03× measured).

---

## 2. Research questions (final)

- **RQ1. Where is the redundancy?** Opacity distribution; importance maps; how many Gaussians carry the unseen side.
- **RQ2. Which criterion is best at a fixed budget?** random, opacity, scale, opacity×area, edge+opacity (PhGS-style), multi-view **contribution**, **error sensitivity**, 2×2/4×4 **merging**. Broken down by viewpoint angle.
- **RQ3. Can importance be predicted from the image?** A tiny head distilled from the contribution oracle, compared with opacity (cheap) and the oracle (upper bound).
- **RQ4. Does training for the budget beat pruning after the fact?** Budget-aware fine-tuning at 10 % and 5 % vs post-hoc pruning with the same Gaussian count, plus a **control fine-tune without the mask** (separates "more training" from "the budget objective").
- **RQ5. What does it cost and save?** Storage, memory, render time at 128²/512²/360°-video, selection time.

---

## 3. Method

Notation: object → N = 16 384 Gaussians with opacity αᵢ. Rendering view v gives blending weights w_{i,p,v} = αᵢ Gᵢ(p) T_{i,p,v}.

### Level 1: training-free selection (`importance.py`, `eval_budget.py`)

| Criterion | Score | Cost |
|---|---|---|
| random / opacity / scale / opacity×area | from predicted parameters | free |
| edge + opacity (PhGS-style) | ½·sobel(input) + ½·α | free |
| **contribution** | Σ_v Σ_p w_{i,p,v} over 25 scoring views | 25 renders + 25 backward passes |
| **error sensitivity** (SNIP/OBD) | Σ_v \|αᵢ ∂L_v/∂αᵢ\| | 25 renders + backward passes + GT |
| merging (ToMe-style) | b×b blocks → 1 Gaussian, moment-matched covariance | free |

**Implementation trick (one sentence for the report):** rendering every Gaussian with colour 1 gives C_p = Σᵢ w_{i,p}, so ∂(Σ_p C_p)/∂cᵢ = Σ_p w_{i,p}. One backward pass through the *unmodified* rasteriser returns the exact per-Gaussian contribution for a view.

No leakage: oracle scores use even-indexed test views, metrics use odd-indexed ones.

### Level 2: learned importance head (`train_head.py`)

- Frozen Splatter Image. The head is 4 conv layers, dilated, ~80 k parameters. Input: RGB + predicted opacity, log-scale, distance, colour maps (11 channels, 128×128). Output: one logit per Gaussian.
- Target: log contribution over 16 random training views of the same object, computed on the fly (no stored targets; random views act as augmentation).
- Loss: **pairwise ranking loss** (only the order matters for top-k).
- Monitored: top-10 % / top-5 % overlap with the oracle on 20 val objects, vs opacity's overlap.
- Test-time cost: one forward of the head (<1 ms), no renders, no GT.

### Level 3: budget-aware fine-tuning (main trained contribution) — updated 18 Nov

**Idea.** Post-hoc selection (Levels 1–2) cannot change the Gaussians that are kept: they were predicted for the full set. Level 3 fine-tunes Splatter Image so that it reconstructs well *given* that only k % of its Gaussians will be kept. This is the only part of the project that changes the model itself; if it beats the best Level-1 result at the same number of Gaussians (hybrid, 21.32 dB at 10 %), it is the main result of the paper.

**How (script `finetune_budget.py`, not a patch of `train_network.py`).** Every step: the model predicts 16 384 Gaussians → a top-k mask keeps k % (forward: exact top-k, as at test time; backward: straight-through through a sigmoid for the learned mask) → render the target views from the kept Gaussians only → original Splatter Image loss (L2 + 0.01·LPIPS) → update the decoder (and the head). Starts from the released cars checkpoint; lr 5e-6 (stage-2 lr) unless the lr check says otherwise.
Why a separate script: `train_network.py` validates on all 352 val objects (~1 h), logs to wandb and keeps an EMA with β = 0.9999 (updated every 10 steps), which barely moves during a short fine-tune, so the fine-tuning would be invisible. `finetune_budget.py`: no EMA, cached validation on 20 val objects every 1000 it (full PSNR, masked PSNR, PSNR per angle bin), resume after a time limit, `FreeCacheDataset` + parallel loading, outputs readable by `eval_budget.py --experiment_path ... --ckpt model_best.pth`.
Head initialisation: `runs/abl_reference/head.pth` (already on Taya's account; same configuration as `head_A`, 8k instead of 20k it., test 20.98 vs 21.11 dB — irrelevant as the head is fine-tuned anyway).

**Runs — Taya's account only (2 running + 2 queued), 10 h each, two waves ≈ 20–22 h in total.**

| # | Run | Command flags | What we want to see |
|---|---|---|---|
| 1 | **Control** | `--mask none` | how much "just more training" gives → the **zero line** for runs 2–4 |
| 2 | **FT-10, learned head** | `--mask learned --keep 0.10` | **main result**: at 1 638 Gaussians, clearly above the control pruned with hybrid and above 21.32 dB → training for the budget beats post-hoc pruning |
| 3 | **FT-10, hybrid mask** | `--mask hybrid --keep 0.10` | vs run 2: if 2 ≈ 3, the gain comes from the decoder adapting and a simple rule is enough; if 2 > 3, learned selection matters |
| 4 | **Group DRO over angle bins** (Level 3b) | `--mask none --view_weighting dro` | per angle bin vs the control at 100 %: the worst bin (90–120°) goes up while the average barely drops |

Wave 1 = runs 1 + 2, wave 2 = runs 3 + 4 (submitted at once; they start automatically). Second wave if time is left: FT-5, fixed angle weights, random budget per step.

**Before launching: smoke test + lr check (≈ 1 h, mandatory).**
1. `--mask learned --keep 0.10 --max_iters 200 --val_every 100` — runs end to end, prints it/s and the validation line.
2. lr check: 1000 it with `--lr 5e-6` and with `--lr 2e-5` (mask 10 %); compare the masked val PSNR. If it barely moves with 5e-6, use 2e-5 for all runs (the same lr for all runs, including the control).

**Evaluation (704 test cars, same protocol as Level 1).**
- FT-10 models at **5 %, 10 %, 20 %** (does the model only fit one budget, or does it become more compact in general?) — `eval_budget.py --experiment_path runs/<run> --ckpt model_best.pth --criteria opacity hybrid learned --head_ckpt runs/<run>/head_best.pth --budgets 0.2 0.1 0.05`.
- Control and DRO at 100 % and with hybrid at 10 % / 5 %.
- Per-angle tables for every model (`analyze_level1.py`), so Level 3 connects to the viewpoint findings and to Level 3b.

**Strengthening (no extra training runs, only evaluation/analysis).**
1. lr check (above) — the main protection against "nothing changed".
2. Evaluate FT-10 at 5 / 10 / 20 %.
3. **How did the model adapt?** Compare opacity / scale distributions of the kept Gaussians (and of the discarded ones) between FT-10 and the original model: e.g. kept Gaussians became larger and more opaque, the others even more transparent. Gives the paper an explanation, not only a number.
4. Angle breakdown for all fine-tuned models.
5. Second wave, optional: one model with a **random budget per step** (k ∈ {5, 10, 20} %) — one model for any budget.

**Known limitation of the straight-through mask.** Masked Gaussians have opacity 0, so the rasteriser gives them no gradient: the head can learn to *demote* kept Gaussians but gets only the sigmoid signal to promote discarded ones. Mention in the paper; the hybrid-mask run (3) is the check.

### Second dataset: SRN Chairs (evaluation only)

Why: cars are left/right **and** roughly front/back symmetric, which probably explains why opposite views (150–180°) are easier than side views. Chairs are left/right symmetric but **not** front/back (backrest vs seat), and they have thin structures (legs, bars).
- **Hypothesis H-sym:** chairs show no U-shape; quality keeps dropping towards the opposite view. Then the viewpoint finding becomes "difficulty depends on object symmetry, not on camera distance".
- **Hypothesis H-thin:** at small budgets opacity pruning hurts chairs more than cars (thin parts), so the contribution/opacity gap is larger.

What we run: `eval_budget.py chairs` (Level 1 + per-view angles), with the released `model_chairs.pth`. If time allows, the cars-trained head on chairs as a transfer test. No Level 2/3 training on chairs.
Cost: download + sbatch unzip of `srn_chairs`, ~1 night of evaluation. Not NMR (64×64, needs code changes), CO3D or GSO (data too heavy for the deadline).

---

## 4. Evaluation protocol

- 704 test cars, 25 evaluation views per object (odd indices, spread over the whole orbit); final best configurations re-run on all 250 views.
- PSNR, **masked PSNR** (object pixels only: the white background inflates PSNR), SSIM, LPIPS.
- Paired differences vs the full model per object, mean ± 95 % CI.
- Everything also broken down by angle bin (30° bins), using `*_views.csv`.
- Cost table: Gaussians, bytes (23 floats/Gaussian: 1.51 MB full, 0.38 MB at 25 %, 0.15 MB at 10 %), render ms at 128² and 512², 100-frame 360° video time, selection ms.

### Main figures (decide now, fill in later)
1. Quality–budget curves (PSNR, LPIPS) for all criteria + learned head + FT-10/FT-5 points. **This is the money plot.**
2. ΔPSNR vs angle bin at 10 % for opacity / contribution / learned / FT.
3. Importance maps over the input image: opacity vs contribution vs learned head.
4. Qualitative strip: side view (~100°) at 5 % for each method.
5. Cost table.

---

## 5. Day-by-day plan

| Day | You | Partner | GPU jobs |
|---|---|---|---|
| **Mon 16** | copy scripts to cluster; smoke test `eval_budget.py --max_objects 3`; fix whatever breaks | Overleaf (CVPR template), skeleton, Preliminaries (3DGS, Splatter Image) | evening: Level 1 full run (704 cars, split into 4 jobs of 176 objects) |
| **Tue 17** | smoke test `train_head.py` (200 steps); launch Level 2 (A, and B if fast); analyse Level 1 CSVs → figures 1–2; **download + unzip SRN Chairs** (sbatch) | Related work (§3 of overview) | Level 2 training (~3–6 h); night: `eval_budget.py chairs` |
| **Wed 18** | `finetune_budget.py` smoke test + lr check (5e-6 vs 2e-5); **launch wave 1 (control, FT-10 learned) + queue wave 2 (FT-10 hybrid, DRO)** | — (Taya's account only) | 4 fine-tunes, 10 h each, 2 at a time |
| **Thu 19** | write intro / related work while wave 1 runs; eval wave 1 in the evening; `benchmark_cost.py` when a slot is free | slides draft | wave 2 running |
| **Fri 20** | eval wave 2; Level 3 analysis (budgets 5/10/20 %, angles, how the model adapted); all final figures; **360° comparison video (§8.1)** | Experiments section | eval jobs |
| **Sat 21** | Results + Limitations; numbers into slides; *optional:* own-photo demo (§8.2) | Intro, Abstract, polish | full-250-view re-runs of best configs |
| **Sun 22–Mon 23** | rehearsal, likely questions, final PDF | same | — |
| **Tue 24** | submission + defense | | |

**Fallback:** if Level 3 doesn't converge by Fri, we report Levels 1 + 2 + cost (still a complete project with a trained component) plus the control-vs-masked training curves as a limitation.

---

## 6. What counts as success

| | Strong | Still solid |
|---|---|---|
| L1 | contribution/error-sens beat opacity by ≥1 dB at 5–10 %, gain concentrated at 90–120° | opacity is sufficient: a clean negative result |
| L2 | head recovers ≥50 % of the oracle–opacity gap at ~0 selection cost | head ≈ opacity; we show what it misses |
| L3 | FT-10 at 1 638 Gaussians close to the full model (e.g. within 0.5 dB), far above post-hoc pruning | gain over post-hoc but not over the control → "more training" explains it (honest) |

---

## 7. Likely committee questions (prepare answers)

- *Why prune if quality is fine and speed barely changes?* → storage/memory/scalability argument; we measured and do not claim speed at 128².
- *Isn't 75 % pruning trivial?* → yes, and that's why the study is about 1–25 %.
- *How is this different from PhGS / AdaptiveSplat / BudSplat?* → they are scene-level and multi-view with hand-designed scores; we work on single-view objects, learn the selection, and train the reconstructor for the budget.
- *Is the oracle leaking test information?* → disjoint view sets (even vs odd).
- *Does the fine-tune gain just come from extra training?* → FT-ctrl.

---

## 8. Demos and creative part (for the defense and the poster, 1 December)

The project direction stays the same (how many / which Gaussians are needed). These demos *illustrate* the results; they do not change the study.

**8.1 360° comparison video — do this (≈1–2 h, cheap, very clear).**
- One car (and later one chair) rotating, 4 panels side by side: full model (16 384 Gaussians) | opacity 10 % | hybrid 10 % | merge 6.25 %, with the number of Gaussians and PSNR written under each panel.
- Shows in one second what the table says in numbers: opacity breaks the silhouette, merge is blurry, hybrid holds up.
- Doubles as the cost measurement planned in §4: time to render a 100-frame 360° video, at 128² and 512², full vs pruned.
- Output: `.mp4` for the slides, a 4-frame strip for the paper / poster.
- How: reuse the orbit rendering from the baseline visualisation (`baseline_renders`, ffmpeg), apply `importance.subset(...)` / `merge_blocks(...)` before rendering.

**8.2 "Our own photo" — optional, only if time is left after Level 3 (≈ half a day, risky).**
- Photograph a real object (toy car, mug, shoe) → Splatter Image → prune to 10 % with the hybrid rank → rotating video.
- Use the **Objaverse** model (released weights), not the cars model (trained on synthetic white-background renders; will likely fail on real photos). Needs background removal (`rembg`, not installed on the cluster yet); the repo's `gradio_app.py` shows the preprocessing.
- Qualitative only (no metrics): one figure in the appendix + a demo at the defense. If the reconstruction is poor, that is an honest result too ("the model struggles on real photos; pruning preserves what is there").

**8.3 Not doing:** video as input (Splatter Image reconstructs from a single frame; per-frame reconstructions flicker and are unrelated to the budget question).

---

## 9. Level 3b — viewpoint-robust fine-tuning (new idea, 16 Nov)

**Motivation (from our own results).** Quality of the full model depends strongly on the angle between the input and target camera: side views (90–120°) are the worst (23.17 dB on 176 000 views) and nearly opposite views recover (25.14 dB). Pruning also hurts different angles very differently (§5.3 of the paper). Question: *can we fine-tune Splatter Image so that the hardest viewpoints improve, without hurting the average?*

**Method.** Fine-tune from the released cars checkpoint (same set-up as Level 3) and change only the weight of each rendered target view in the loss. The angle θ between the input and target camera is computed in the training loop for every rendered view; views are grouped into 30° bins.

| Run | View weights | Purpose |
|---|---|---|
| **Control** | uniform | separates "more training" from the weighting; **the same run is the control of Level 3** |
| **Fixed** | weight λ (e.g. 2–3) for 90–120°, 1 otherwise | the simple version of the idea |
| **Group DRO over angle bins** | weight of bin g ∝ exp(η · running mean loss of bin g), updated online (Sagawa et al., ICLR 2020) | principled version: the model itself finds and up-weights the hardest viewpoints; a method borrowed from robust ML |

**Implementation (≈30 lines, shared with Level 3).** One patch of `train_network.py` with flags `--budget_keep` (Level 3) and `--view_weighting {none,fixed,dro}`:
1. compute the per-image L2 (and LPIPS) loss instead of the batch mean;
2. compute θ for each target view from `camera_centers` (input view = index 0);
3. weight each image's loss by its bin weight (normalise weights to mean 1, so the learning rate stays comparable);
4. for DRO keep an exponential moving average of the loss per bin; Group DRO needs some regularisation / early stopping (Sagawa et al.).

**Evaluation.** `eval_budget.py` (criterion `full` only, or with `--criteria opacity hybrid` to combine with pruning) + `analyze_level1.py` angle table. Main metric: **PSNR of the worst angle bin**; also average PSNR / LPIPS and the full per-bin curve vs. the control.

**Runs to launch together (4 parallel fine-tunes, ~20 h each, 2 slots per account):** control · budget 10 % (Level 3) · Group DRO angles (Level 3b) · budget 5 % *or* fixed angle weights.

**Risks (be honest in the paper).**
- The gain may be small or a redistribution (e.g. +0.2 dB at the side, −0.1 dB near the input): side views are hard largely because they show what the input view cannot see; re-weighting moves capacity, it does not create information. A trade-off is still a valid result.
- Short fine-tuning of a large pretrained model often changes metrics by only tenths of a dB — the control run is mandatory.
- Time: shares the infrastructure with Level 3, so the extra cost is one fine-tuning run, not a new project.

**Where it goes in the paper.** Results §5.3 ("importance and quality depend on the viewpoint") → new subsection "Can training target hard viewpoints?". Related Work: CAT3D / G3DR (view schedules), ERGO (per-view weights in single-image 3DGS optimisation), hard-sample mining in NeRF/3DGS, Group DRO (robust optimisation over groups).

### Resources (saved for Level 3b; BibTeX entries added to `cvpr_paper/main.bib`)

| Paper | What it does | Relation to us |
|---|---|---|
| [Group DRO — Sagawa, Koh, Hashimoto, Liang, ICLR 2020](https://arxiv.org/abs/1911.08731) | minimise the loss of the worst group, online group weights; needs strong regularisation / early stopping | **our method** with groups = angle bins |
| [ERGO — Ma et al., arXiv 2602.10278 (2026)](https://arxiv.org/abs/2602.10278) | per-view adaptive loss weights ("excess risk") in single-image 3DGS optimisation with generated views | closest prior work; differs: per-object optimisation on noisy generated views vs. our feed-forward model on real views and worst-viewpoint objective |
| [CAT3D — Gao et al., NeurIPS 2024](https://arxiv.org/abs/2405.10314) | weights views by distance to the captured views (Gaussian kernel, annealed) | view schedule by proximity, not by difficulty |
| [G3DR — Reddy et al., CVPR 2024](https://arxiv.org/abs/2403.00939) | samples novel views with increasing probability over training | curriculum over view types |
| [Uncertainty-weighted few-shot NeRF, arXiv 2410.17839](https://arxiv.org/abs/2410.17839) | learned uncertainty → higher loss weight for hard examples | per-sample, sparse-view NeRF |
| [Korhonen et al., "Not All Samples Remain Equally Hard", arXiv 2408.03193](https://arxiv.org/abs/2408.03193) | hard-ray mining in NeRF | hard samples, not hard viewpoints |
| [Hard Gaussian Splatting — Xu et al., AAAI 2026, arXiv 2412.04826](https://arxiv.org/abs/2412.04826) | grows Gaussians using multi-view gradients / errors of hard views | hard Gaussians in per-scene 3DGS |
| [TokenGS, arXiv 2604.15239](https://arxiv.org/abs/2604.15239); [Image2GS, arXiv 2507.00916](https://arxiv.org/abs/2507.00916) | visibility losses / masks in feed-forward 3DGS | related: what the input cannot see |

*Check every arXiv ID / venue on the abstract page before citing (the 2026 ones in particular).*
