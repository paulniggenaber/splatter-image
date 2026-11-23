# How Many Gaussians Are Really Needed?
### Opacity sparsity, viewpoint robustness and learned Gaussian selection in single-view 3D reconstruction

*Project overview: motivation, related work, method, plan and what counts as a good result. Base paper: Splatter Image (Szymanowicz, Rupprecht, Vedaldi, CVPR 2024). Defense: 24 November.*

---

## 1. One-paragraph summary

Splatter Image reconstructs a 3D object from one image by predicting **one 3D Gaussian per input pixel** (128 × 128 = 16,384 Gaussians per object). This fixed, dense allocation is convenient but almost certainly wasteful. Our preliminary experiments on ShapeNet-SRN Cars confirm it: **74% of the predicted Gaussians are nearly transparent**, and removing 75% of them (keeping 4,096 out of 16,384) changes PSNR by only −0.004 dB. We turn this observation into a study of *which Gaussians actually matter for novel-view quality*. We compare simple pruning criteria with a multi-view contribution score and an error-sensitivity score, show where each one breaks down (very small budgets, difficult viewpoints), and train a lightweight network that **predicts Gaussian importance directly from the input image**, so a compact representation can be produced in a single forward pass. As a stretch goal, we fine-tune Splatter Image with a sparsity objective and test whether *training for compactness* beats *pruning after the fact*.

---

## 2. Motivation: why this matters

1. **Pixel-aligned Gaussians are redundant by design.** Every pixel gets a Gaussian, including background pixels and pixels whose Gaussians end up invisible. The representation size is fixed by the image resolution, not by how complex the object is.
2. **Compactness has real costs.** Fewer Gaussians mean less memory and storage per object, cheaper transmission (e.g. streaming 3D assets) and potentially faster rendering when many views are rendered (360° videos, VR) or the output resolution is high. These costs grow with the multi-view variant, where the Gaussians of N views are merged.
3. **Opacity is the obvious criterion, but is it the right one?** A Gaussian can be very opaque yet hidden behind others, or semi-transparent yet the only primitive representing the unseen back of the object. Single-view reconstruction must *hallucinate* the back side, so viewpoint-dependent effects are likely. This gives a concrete scientific question beyond engineering.
4. **The interesting part is the hard regime.** Our first result shows that 75% pruning is essentially free. The question becomes: what happens at 10%, 5%, 2% of the Gaussians? That is where a better importance criterion should make a measurable difference.

### Research questions

- **RQ1 (characterisation).** How sparse is Splatter Image's opacity distribution, and where in the image are the near-transparent Gaussians (background vs object pixels)?
- **RQ2 (criteria).** At a fixed Gaussian budget, which selection criterion best preserves novel-view quality: random, opacity, scale, edge + opacity, multi-view contribution, or error sensitivity? Does opacity-based pruning disproportionately hurt *difficult* target viewpoints?
- **RQ3 (learned selection).** Can a lightweight head on frozen Splatter Image features predict Gaussian importance well enough to match the expensive multi-view scores, without any extra renders at test time?
- **RQ4 (cost).** How do quality, rendering cost and selection cost trade off? Why does a 4× reduction in Gaussians give only a 1.03× speed-up at 128 × 128?
- **RQ5 (stretch).** Does fine-tuning Splatter Image with a budget-aware sparsity objective beat post-hoc pruning at the same Gaussian count?

---

## 3. Related work, and how we differ

### 3.1 Feed-forward Gaussian reconstruction (our setting)
- **Splatter Image** (Szymanowicz et al., CVPR 2024): our base model; one Gaussian per pixel from a U-Net, object-level (ShapeNet, CO3D, Objaverse).
- **pixelSplat** (Charatan et al., CVPR 2024), **MVSplat** (Chen et al., ECCV 2024), **Flash3D** (Szymanowicz et al., 2024), **LRM / GS-LRM**: other feed-forward models; all predict dense, mostly pixel-aligned Gaussians.

### 3.2 Reducing redundancy in feed-forward 3DGS (closest to us, cite carefully)
- **PhGS: Post-Hoc Pruning and Refinement of Single-View Feed-Forward 3D Gaussian Reconstructions** (arXiv 2609.20623, 2026). **Closest prior work.** Frozen backbone, a *heuristic* importance mask (edge prior + predicted opacity, fixed-count weighted sampling) and a trainable recurrent module that *refines* the surviving Gaussians. It is evaluated on scene-level backbones (Flash3D, Niagara, SHARP).
  → **Difference:** our selection score is *learned* (distilled from multi-view contribution / error-sensitivity oracles) instead of hand-designed; we work on object-level Splatter Image on ShapeNet; we analyse the effect by *viewpoint* and profile rendering cost. We also **use their edge + opacity score as a baseline**.
- **AdaptiveSplat** (ECCV 2026, arXiv 2607.04256): texture-aware (wavelet) pruning under a user budget plus a head that re-predicts the kept Gaussians' attributes; scene-level, multi-view.
- **BudSplat** (arXiv 2610.38278): feed-forward 3DGS under a rendering budget; detail-aware importance plus rendering contribution; scene-level, multi-view.
- **Long-LRM** (low-opacity pruning), **GGN** (Zhang et al., NeurIPS 2024; deduplicates pixel-aligned Gaussians across views with a graph network), **FreeSplat** (fuses overlapping per-view Gaussians), **SparseSplat / ZipSplat** (abandon the one-Gaussian-per-pixel grid).
  → These target *multi-view scene* redundancy (overlap between views). **Ours targets single-view object reconstruction, where the redundancy is different**: background pixels and the hallucinated back side.

### 3.3 Pruning per-scene optimised 3DGS (importance scores we borrow)
- **3D Gaussian Splatting** (Kerbl et al., SIGGRAPH 2023): the renderer; alpha compositing, opacity, transmittance.
- **LightGaussian** (Fan et al., NeurIPS 2024): "global significance" = how often and how strongly a Gaussian contributes across training views. Our *multi-view contribution* score is the feed-forward analogue.
- **PUP 3D-GS** (Hanson et al., CVPR 2025): pruning score from a second-order (Fisher) approximation of the reconstruction error w.r.t. each Gaussian's mean and scale; prunes 90% with a 3.56× speed-up on optimised scenes. Our *error-sensitivity* score is a cheaper first-order cousin, computed for predicted (not optimised) Gaussians.
- **Compact 3DGS** (Lee et al., CVPR 2024) and **LP-3DGS** (Zhang et al., NeurIPS 2024): *learned* masks, but optimised per scene. Ours predicts the mask feed-forward and generalises to new objects.
- **Mini-Splatting** (Fang & Wang, ECCV 2024): reorganises Gaussians for compactness, per scene.
  → All of these need per-scene optimisation and many posed images. **We have one image and no test-time optimisation.**

### 3.4 Motivation from other domains: learned selection works elsewhere
- **Learning to Sample** (Dovrat, Lang, Avidan, CVPR 2019) and **SampleNet** (Lang, Manor, Avidan, CVPR 2020): learning *task-aware* point-cloud sampling beats farthest-point sampling and keeps accuracy with a small fraction of points. → Our Level 2 is "task-aware sampling" for Gaussians, where the task is novel-view rendering.
- **DynamicViT** (Rao et al., NeurIPS 2021) and **Token Merging, ToMe** (Bolya et al., ICLR 2023): a light prediction module decides which vision-transformer tokens to keep, cutting compute with little accuracy loss. → Same idea: a cheap head predicts which primitives are worth keeping.
- **Optimal Brain Damage** (LeCun et al., 1990) and **SNIP** (Lee, Ajanthan, Torr, ICLR 2019): estimate how much the loss changes when a weight is removed, using gradients (|w · ∂L/∂w|). → Our error-sensitivity score applies exactly this to Gaussians (|α · ∂L/∂α|).
- **Knowledge distillation** (Hinton et al., 2015): train a cheap model to imitate an expensive one. → Our importance head imitates expensive multi-view oracles.

**How we use these ideas concretely:**

| Idea from another domain | Where it appears in our project |
|---|---|
| SNIP / Optimal Brain Damage (gradient saliency of weights) | the **error-sensitivity** score \|α·∂L/∂α\| (Level 1, target B for Level 2) |
| DynamicViT (light module that predicts which tokens to keep) | the **Level 2 importance head**; its differentiable masking becomes **variant C** |
| Learning to Sample / SampleNet (task-aware learned sampling) | Level 2 overall, and especially variant C, trained through the task loss (rendering) |
| Token Merging, ToMe (merge similar tokens instead of dropping them) | **merging** 2×2 blocks of pixel-aligned Gaussians as a training-free Level 1 criterion |
| Knowledge distillation | variants A and B imitate expensive multi-view oracles |

> **Novelty in one sentence:** prior work prunes either *per-scene optimised* Gaussians, or *multi-view scene-level* feed-forward Gaussians with *hand-designed* scores; we study *single-view object-level* pixel-aligned Gaussians, compare contribution- and error-based importance by viewpoint, and *learn* to predict importance from the image.

---

## 4. Method

### Notation
For an object, Splatter Image outputs Gaussians *i = 1…N* (N = 16,384) with opacity α_i. Rendering a view *v* gives, for every pixel *p*, the blending weight *w_{i,p,v} = α_i · G_i(p) · T_{i,p,v}*, where T is the transmittance (how much light is still unblocked in front of Gaussian *i*).

### Level 1: post-hoc pruning (training-free)
Importance scores for each Gaussian:

| Criterion | Score | Needs |
|---|---|---|
| Random | — | nothing (control) |
| Opacity | α_i | nothing |
| Scale | volume of Σ_i | nothing |
| Edge + opacity (PhGS-style) | λ_e · edge(p_i) + λ_o · α_i | input image |
| **Multi-view contribution** | Σ_v Σ_p w_{i,p,v} over V virtual orbit cameras | V renders, **no ground truth** |
| **Error sensitivity (new)** | Σ_v \|α_i · ∂L_v/∂α_i\| | renders + ground-truth views, one backward pass |
| **Merging (ToMe-style)** | no score: merge each 2×2 block of pixel-aligned Gaussians into one | nothing (training-free) |

- **Contribution** asks: *how much does this Gaussian appear in the images?*
- **Error sensitivity** asks: *how much worse would the images get if this Gaussian were weakened?* It is the first-order Taylor estimate of the loss change from removing the Gaussian (SNIP / Optimal Brain Damage). It needs ground-truth images, so it is computed on training objects (as a target for Level 2) and, as an oracle upper bound on test objects, **only from views disjoint from the evaluation views** (no test leakage).

**Merging instead of dropping (inspired by Token Merging).** Neighbouring pixels usually describe neighbouring surface points, so each 2×2 (or 4×4) block of Gaussians is replaced by one: opacity-weighted mean position and colour, covariance by moment matching (the covariance of the mixture: weighted Σ_j plus the spread of the means), and opacity 1 − Π_j(1 − α_j). This gives exactly 25% (or 6.25%) of the Gaussians without discarding any information. Hypothesis: merging degrades more gracefully than dropping at very small budgets.

Keep the top-k Gaussians for budgets **k ∈ {25, 20, 10, 5, 2, 1}%** (25% is already "free", so the interesting range is below it). Report PSNR / SSIM / LPIPS, rendering time and memory, and everything broken down by the angle between the input and target cameras.

### Level 2: learned importance prediction (our trainable component)
A small head (a few 1×1 / 3×3 conv layers) on the **frozen** U-Net decoder features plus the predicted Gaussian parameters outputs one score per pixel (that is, per Gaussian). It is trained on the training split to imitate a target score:
- **Variant A:** multi-view contribution target;
- **Variant B:** error-sensitivity target.

- **Variant C (end-to-end, inspired by DynamicViT / SampleNet):** instead of imitating a hand-made target, the head outputs a soft keep-probability m_i ∈ (0,1) per Gaussian (sigmoid, or Gumbel-softmax for near-binary masks). Opacities are multiplied by m_i, the masked Gaussians are rendered from training views, and the head is trained directly with L2 + LPIPS + a budget penalty (mean(m) − k)². At test time we keep the top-k by m_i. This lets the network discover what matters for the image itself. It is riskier to train, so it starts only once A and B are running.

Loss for A and B: regression on log-importance, or a top-k ranking loss (only the order matters for pruning). At test time, selection costs **one extra forward pass of a tiny head**, with no renders and no ground truth.

### Level 3 (stretch): budget-aware fine-tuning
Fine-tune the decoder of Splatter Image with an added sparsity term (e.g. an L1 penalty on opacity, or a penalty on the number of Gaussians above a threshold), at two strengths λ. Compare with post-hoc pruning **at the same number of Gaussians**, and with plain fine-tuning without the sparsity term (to separate "more training" from "the sparsity objective").

### Cost analysis (RQ4)
Time each stage separately (U-Net forward, Gaussian preprocessing/sorting, rasterisation) with CUDA events, warm-up and 100+ repeats, on a full GPU (not a MIG slice). Repeat at 128², 256², 512² output resolution and for a 100-frame 360° render. **Hypothesis:** at 128², rendering is a small fraction of total time and near-transparent Gaussians are already cheap for the rasteriser, so pruning saves memory but not time; savings should grow with resolution and the number of rendered views.

---

## 5. Evaluation protocol

- **Data:** ShapeNet-SRN Cars test split (all objects, not 20), standard Splatter Image evaluation views.
- **Metrics:** PSNR, SSIM, LPIPS; also **masked PSNR** (object pixels only), because the white background inflates PSNR and differs between viewpoints.
- **Statistics:** paired differences per object against the unpruned model, with mean and 95% confidence interval.
- **Viewpoint analysis:** θ = arccos(ĉ_input · ĉ_target) in degrees, in bins of 30°.
- **Cost:** number of Gaussians, memory and storage (bytes), render time per stage, selection time.

---

## 6. What counts as a good result

Each level produces a result we can present, whether the answer is "yes" or "no".

| Level | Strong result | Still a solid, honest result |
|---|---|---|
| **1** | At 5–10% budgets, contribution or error sensitivity clearly beats opacity (e.g. +0.5–1 dB PSNR, lower LPIPS), and the gain is concentrated on difficult viewpoints | All criteria are equivalent down to ~10%, so opacity is a sufficient criterion for Splatter Image. That is a useful negative result: cheap pruning is enough |
| **2** | The learned head closes most of the gap between opacity and the oracle (e.g. recovers >50% of the oracle's improvement) at almost zero selection cost | The head matches opacity but not the oracle; we analyse what it fails to predict (e.g. the back side) |
| **3** | The fine-tuned model reaches the same quality with far fewer Gaussians than post-hoc pruning at small budgets | Fine-tuning is unstable or gives no gain over pruning; we report it as a limitation |
| **Cost** | Profiling explains the 1.03× speed-up and shows larger speed-ups at higher resolution or for many rendered views | Rendering is negligible at this scale; the benefit is memory and storage (4× smaller) |

Our first result already holds: **a 4× reduction in Gaussians with no measurable loss in quality** (to be confirmed on the full test set with confidence intervals). **We must not claim a 4× speed-up.** We reduced the Gaussian count 4×; the speed changed only 1.03×.

---

## 7. Plan until the defense

| Date | Work |
|---|---|
| **Mon 16.11** | Export Gaussians; multi-view contribution + **error-sensitivity** scores; pruning function for all criteria; start the report (template, skeleton, Preliminaries). Tonight: launch the job computing both importance targets on the train split |
| **Tue 17.11** | Level 1 on the full test set (all criteria × budgets 25–1%), per-view CSV; stage-wise profiling + 512² / 360° timing; implement ToMe-style merging while the job runs. Evening: check targets, write the predictor's dataloader |
| **Wed 18.11** | Level 2: train the head (variants A and B; then the end-to-end variant C). Level 3: add the sparsity loss, 1k-iteration test, launch 2 fine-tuning jobs. Report: Method |
| **Thu 19.11** | Slides draft; check Level 3 jobs |
| **Fri 20.11** | Evaluate Levels 2 and 3; final plots (quality vs budget, quality vs angle per criterion, renders of the back view, cost table). Report: Intro + Related Work |
| **Sat 21.11** | Report: Experiments, Results, Limitations; real numbers into the slides |
| **Sun 22 – Mon 23.11** | Run-throughs, likely committee questions, final PDF |
| **Tue 24.11** | Report submission + defense |

**Fallback:** if Level 3 does not finish, Levels 1 + 2 + the cost analysis form a complete project, with a trained component of our own.

---

## 8. Why it is useful

- **For Splatter Image users:** a concrete recipe for 4× smaller (or smaller still) reconstructions with no quality loss, and an honest account of when it does or does not speed things up.
- **For the field:** evidence on whether opacity is a sufficient importance criterion for pixel-aligned feed-forward Gaussians, and whether importance can be *predicted* from the image. This matters for any one-Gaussian-per-pixel model (pixelSplat, Flash3D, LRM-style models).
- **For understanding single-view reconstruction:** the viewpoint breakdown shows which Gaussians carry the hallucinated, unseen geometry, and whether compression hurts exactly the hardest part of the problem.

---

## 9. References (check every one before citing; arXiv numbers from a web search, Nov 2026)

- Szymanowicz, Rupprecht, Vedaldi. *Splatter Image: Ultra-Fast Single-View 3D Reconstruction.* CVPR 2024. arXiv 2312.13150
- Kerbl et al. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* SIGGRAPH 2023
- *PhGS: Post-Hoc Pruning and Refinement of Single-View Feed-Forward 3D Gaussian Reconstructions.* arXiv 2609.20623 (2026)
- Singhal et al. *AdaptiveSplat: Texture Aware Controllable 3D Gaussian Allocation for Feed-Forward Reconstruction.* ECCV 2026, arXiv 2607.04256
- Xu et al. *BudSplat: Feed-forward 3D Gaussian Splatting under a Rendering Budget.* arXiv 2610.38278
- Hanson et al. *PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting.* CVPR 2025, arXiv 2406.10219
- Fan et al. *LightGaussian.* NeurIPS 2024
- Lee et al. *Compact 3D Gaussian Representation for Radiance Field.* CVPR 2024
- Zhang et al. *LP-3DGS: Learning to Prune 3D Gaussian Splatting.* NeurIPS 2024, arXiv 2405.18784
- Fang, Wang. *Mini-Splatting.* ECCV 2024
- Zhang et al. *GGN: Gaussian Graph Network.* NeurIPS 2024
- Charatan et al. *pixelSplat.* CVPR 2024; Chen et al. *MVSplat.* ECCV 2024
- Dovrat, Lang, Avidan. *Learning to Sample.* CVPR 2019; Lang, Manor, Avidan. *SampleNet.* CVPR 2020, arXiv 1912.03663
- Rao et al. *DynamicViT.* NeurIPS 2021; Bolya et al. *Token Merging (ToMe).* ICLR 2023
- LeCun, Denker, Solla. *Optimal Brain Damage.* NeurIPS 1990; Lee, Ajanthan, Torr. *SNIP.* ICLR 2019
- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* 2015
