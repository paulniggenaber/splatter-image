# Splatter Image 

## So far

I started by looking at how well original Splatter Image model from the paper reconstructs cars from different viewpoints.

I implemented `eval_viewpoint_gap.py`, which measures the angle between the input camera and each target camera and records PSNR, SSIM and LPIPS and ran this on the full SRN Cars test set (704 cars, 176,000 views), so we now have the data to see how reconstruction quality changes as we move further away from the original viewpoint.

(Results are in `viewpoint_gap_full704.csv`)

## Gaussian pruning

I wanted to see if we reallt need all the Gaussians that Splatter Image predicts.

`eval_pruning.py` removes Gaussians with the lowest opacity after the model predicts them. No retraining is needed for this.

Each car originally has 16,384 Gaussians.

I tested pruning on 20 cars and got these results:

| Pruning | Gaussians left | Mean PSNR |
|---------|----------------|-----------|
| 0%      | 16,384         | 24.8592   |
| 50%     | 8,192          | 24.8592   |
| 75%     | 4,096          | 24.8548   |
| 90%     | 1,638          | 20.2292   |

The we can remove 75% of the Gaussians with no quality loss (only -0.0043 dB PSNR on average).

I also checked the opacity distribution and found that around 74% of Gaussians have opacity below 0.001, so most of them are almost completely transparent.

I measured rendering time (on 5 cars only). At 75% pruning, rendering was only around 3% faster. These are preliminary measurements, so we shouldn't claim a significant speedup yet.

## Where I think we could take the project

My idea is to build the project in a few levels:

### Level 1 — Multi-view importance

Instead of deciding which Gaussians to keep based only on opacity, we could measure how important each Gaussian actually is for reconstruction across different viewpoints.

For example, we could estimate how much reconstruction quality changes when a Gaussian (or a group of Gaussians) is removed.

This would give us better importance scores and help us understand whether low-opacity Gaussians are always unnecessary, especially for difficult viewpoints.

We could compare this against our current opacity-based pruning baseline.

### Level 2 — Learned importance prediction (main idea)

We could freeze the original Splatter Image model and train a small additional head that predicts how important each Gaussian is directly from the input image features. For training targets, we would use the multi-view importance scores from Level 1, computed on the training split.

The goal is that, after training, the head can select the most useful Gaussians in a single forward pass, without needing extra renders to calculate their importance.

Then we could compare:
- Opacity pruning (simple baseline)
- Learned importance prediction (our trained method)
- Multi-view importance (oracle/reference method)

We would compare reconstruction quality at different Gaussian budgets, as well as the time needed to select the Gaussians.

I think this could be our main contribution because we would actually train our own component and investigate whether the model can learn which Gaussians matter.

### Level 3 — Fine-tuning Splatter Image

If we have enough time, we could also try fine-tuning part of the original Splatter Image model.

For example, we could add a sparsity penalty or a budget-aware loss to encourage the model to produce fewer useful Gaussians while maintaining reconstruction quality.

## What I think we should do next

I'd suggest keeping the opacity pruning results as our baseline and focusing on Level 1 and Level 2 next.

We already have the viewpoint gap analysis, which could also help us investigate whether different Gaussian selection methods perform worse on unseen or difficult viewpoints.

Ideally, the final project would show not just that many Gaussians are redundant, but whether we can **learn to select a smaller, more useful set of Gaussians** while preserving reconstruction quality.

Level 3 would be nice I'm afraid we wouldn't have enough time though but hopefully
