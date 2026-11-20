# Symmetry extension: results (Paul), as of 20 Nov 2026

All symmetry code is on this branch (`paul/dev`). All runs on the Bocconi cluster. One seed per run. "Hidden side" = the views the input photo cannot see.

**Footnote for all cars tables:** SRN test car 375 (`876d92ce6a0e4bf399588eee976baae`) contains no car. All 251 of its images are blank white (at most 2 non-white pixels of 16,384). Every model renders a white image too and scores 60 to 78 dB on it (pure mirroring even an infinite PSNR). That is a data error, not a result, so this car is left out of all cars tables: they use the **703** other test cars. Leaving it in would raise every model's mean by about 0.06 dB. The differences between models barely change. How: each model was also run on car 375 alone, and that score was subtracted from its 704-car mean, i.e. mean_703 = (704 × mean_704 − car_375) / 703. Pure mirroring with the predicted plane (infinite PSNR on car 375) was re-run directly on the 703 cars (job `car375`, results in `results/car375/` and `results/sym_purem_40k_n703/`). The first-100-car tables do not contain car 375 and are unchanged.

## 1. Result in one table (cars, 703 test cars, see footnote)

Both models start from the released cars model and get the same fine-tuning: 40,000 steps, learning rate 2e-5, LPIPS loss, same data. The only difference is the symmetry extension.

| | PSNR | SSIM | LPIPS | Hidden-side PSNR | Hidden-side LPIPS |
|---|---|---|---|---|---|
| Released model (Taya's 704-car score, car 375 subtracted*) | 23.93 | 0.923 | 0.077 | n/a | n/a |
| **Control**: fine-tuned, no symmetry | 23.93 | 0.9230 | 0.0758 | 23.19 | 0.0858 |
| **Symmetry**, plane predicted from the photo | **24.21** | **0.9263** | **0.0711** | **23.64** | **0.0770** |
| Symmetry with the true plane (upper bound) | 24.35 | 0.9282 | 0.0703 | 23.86 | 0.0758 |

\* Taya's 704-car numbers are rounded and her per-car scores are not available. Car 375 was subtracted from them using our own run of the released model on that car (70.7 dB), so this row is accurate to about ±0.01.

- Gain over the fair control: **+0.28 dB PSNR, +0.0033 SSIM, -0.0047 LPIPS** overall; on the hidden side **+0.45 dB and -0.0088 LPIPS**.
- The plane being predicted instead of known costs about 0.14 dB.
- The control did not improve with longer training (23.91 at 10,000 steps, 23.93 at 40,000). So the gain comes from the symmetry, not from the extra steps.
- Still one photo in, one forward pass out. The only additions at test time are a small ResNet-18 for the plane and the mirror step.

## 2. What was built (short)

1. **Plane predictor** (`scene/plane_predictor.py`, `train_plane.py`): ResNet-18 on the photo, outputs the mirror plane. Trained alone (so it cannot disturb the reconstruction). Test error: 1.5 to 1.7 degrees on cars (mean), median about 1 degree.
2. **Twins**: every Gaussian on the camera's side gets a mirrored copy across the plane, with a per-Gaussian on/off switch and a small colour correction (`scene/symmetry.py`, `sym_head` in `scene/gaussian_predictor.py`).
3. **Training**: the released model is fine-tuned with the frozen plane predictor in the loop, so the network learns to live with its small plane errors.

Bug found on the way: training the plane inside the U-Net damaged the reconstruction (twins off: 22.46 against the control 24.22). The fix was to decouple the plane from the network.

## 3. Ablations (703 test cars, `sym_long`, same cars for every row)

All rows use the trained `sym_long` model; only one thing is changed at test time. (Earlier version of this table: first 100 cars, same conclusions.)

| | PSNR | SSIM | LPIPS | Hidden PSNR | Hidden LPIPS |
|---|---|---|---|---|---|
| Control (separate model, no symmetry) | 23.93 | 0.9230 | 0.0758 | 23.19 | 0.0858 |
| Symmetry, predicted plane | 24.21 | 0.9263 | 0.0711 | 23.64 | 0.0770 |
| Symmetry, true plane | **24.35** | **0.9282** | **0.0703** | **23.86** | 0.0758 |
| ...true plane, colour correction off | 24.31 | 0.9277 | 0.0713 | 23.78 | 0.0777 |
| ...twins switched off | 23.78 | 0.9218 | 0.0770 | 22.96 | 0.0878 |
| ...true plane, all switches forced on | 23.92 | 0.9247 | 0.0727 | 23.42 | 0.0765 |
| ...true plane, "replace far side" (training-free) | 23.68 | 0.9261 | 0.0718 | 23.08 | **0.0733** |
| ...true plane, "twins for everything" (training-free) | 23.21 | 0.9182 | 0.0765 | 23.10 | 0.0770 |

What this says:
- **The twins carry the gain**: switching them off drops 0.43 dB, even 0.15 dB below the control (the network got used to having them).
- **The learned switch matters**: forcing every twin on loses 0.43 dB against the learned switch (both with the true plane). The model switches off about 80% of the twins (mean switch value 0.19).
- **The colour correction is worth little**: -0.04 dB PSNR without it, but hidden-side LPIPS gets worse (0.0758 → 0.0777). Keep it.
- **Replacing the far side by mirror copies** gives the best hidden-side LPIPS (0.0733) but loses 0.67 dB PSNR. The trained version of this idea (section 4c) also lost.

## 4. Chairs (SRN chairs, all 1,317 test chairs)

Fine-tuned 10,000 steps on **all 4,612** training chairs, released chairs model as start, learning rate 2e-5. Control and symmetry runs have identical settings. (This replaces the earlier runs on 2,000 chairs.)

| | PSNR | SSIM | LPIPS | Hidden PSNR | Hidden LPIPS |
|---|---|---|---|---|---|
| Released chairs model | 24.43 | 0.9321 | 0.0675 | 24.01 | 0.0745 |
| Control (fine-tuned, no symmetry) | 24.37 | 0.9321 | 0.0664 | 23.95 | 0.0733 |
| Symmetry, predicted plane | 24.45 | 0.9323 | 0.0658 | 24.05 | 0.0722 |
| Symmetry, true plane (upper bound) | 24.50 | 0.9330 | 0.0652 | 24.12 | 0.0714 |

- Symmetry against its control: +0.08 dB PSNR, -0.0006 LPIPS; hidden side +0.10 dB, -0.0011 LPIPS. Every metric moves the right way, but the PSNR gain is under our 0.1 dB noise line (one seed). **Report as "small, consistent, not significant"**, not as a gain.
- Against the released model: +0.02 dB PSNR, -0.0017 LPIPS.
- Fine-tuning alone still lowers PSNR (24.37 against 24.43) but improves LPIPS: the LPIPS loss trades a little PSNR for sharper images. So not overfitting; the 2,000-chair suspicion is gone.
- Why chairs gain less than cars: many chairs are already well covered from the near side (thin legs, open backs), and the plane predictor is less exact on chairs (4.4 degrees mean, 1.5 median; cars 1.5). The true plane adds only +0.05 dB more, so plane error is not the main limit.

## 4b. Tried, no gain: far-side keep gate (cars, 703 test cars)

Idea: let the network also switch off its own original Gaussians on the hidden side, so the twins can take over there. Same 40,000-step fine-tune as `sym_long`.

| | PSNR | SSIM | LPIPS | Hidden PSNR | Hidden LPIPS |
|---|---|---|---|---|---|
| Symmetry (`sym_long_40k`) | 24.21 | 0.9263 | 0.0711 | 23.64 | 0.0770 |
| Symmetry + keep gate (`sym_fargate_40k`) | 24.21 | 0.9262 | 0.0711 | 23.64 | 0.0770 |
| ...with the true plane | 24.35 | 0.9282 | 0.0704 | 23.87 | 0.0759 |

- Identical within 0.01 dB. The gate learned to keep 72% of the far-side originals, so it removes some but the result does not change. **Dropped; `sym_long_40k` stays the main model.**
- Worth one line in the report: the learned twin switch already does the job, a second gate adds nothing.

## 4c. Tried, worse: pure mirroring (cars, 703 test cars)

Idea: remove the network's own Gaussians on the hidden side, so that side is built only from mirrored twins (the trained version of "replace far side"). Same 40,000-step fine-tune as `sym_long`, run `sym_purem`.

| | PSNR | SSIM | LPIPS | Hidden PSNR | Hidden LPIPS |
|---|---|---|---|---|---|
| Control (`control_long_40k`) | 23.93 | 0.9230 | 0.0758 | 23.19 | 0.0858 |
| Symmetry (`sym_long_40k`) | **24.21** | **0.9263** | **0.0711** | **23.64** | 0.0770 |
| Pure mirroring, predicted plane | 23.87 | 0.9237 | 0.0745 | 23.45 | **0.0765** |
| Pure mirroring, true plane | 24.22 | 0.9278 | 0.0734 | 24.00 | 0.0748 |

- Pure mirroring is **0.34 dB worse** than the symmetry model and even below the control. **Dropped.**
- It depends completely on the plane: with the true plane it gains 0.35 dB, against 0.14 dB for `sym_long`. A small plane error now misplaces the whole hidden side.
- Lesson for the report: keeping the network's own hidden-side guess and adding twins with a learned switch is better than replacing the hidden side by a mirror copy.

## 5. Limits to state in the report

- Helps for roughly symmetric objects only; the hidden side beyond the mirror (the rear) is not covered. On chairs the gain is small (+0.08 dB on all 1,317 test chairs), below our noise line.
- Per-object score lists (`per_object_*.txt`) of evaluations that ran at the same time got mixed (shared file, fixed in commit edffc8d). The mean scores in this report are computed in memory and are correct. Only a paired per-object test would need a re-run.
- The true plane is known in SRN because all objects are aligned. For unaligned data (CO3D) the plane predictor would have to learn much more.
- One seed per run. The 703-car gain (+0.28 dB) is solid for that seed; the 100-car ablations are smaller samples.
- No speed claim: the twins double the Gaussians that are rendered, though at 128 x 128 render time barely changes.

## 6. Still open

- Done: trained pure-mirror variant (4c, worse); far-side keep gate (4b, no gain); ablations on 703 cars (3).
- Not run: higher learning rate; training without the colour correction (the ablation already shows it is worth little).
- Combination with Taya's fewer-Gaussians (store only the camera-side half plus the plane).

## 7. Where things are

- Code: this branch, `paul/dev` (plane predictor `scene/plane_predictor.py` + `train_plane.py`, mirroring `scene/symmetry.py`, model changes in `scene/gaussian_predictor.py`, evaluation `eval.py`, cluster scripts `slurm/`, tests `tests/`). A full copy of results, checkpoints and logs is on Paul's Mac. Cluster results: `~/splatter-image-night/results/` (`sym_long_40k`, `control_long_40k`, `control_long_10k`, `sym_fargate_40k`, `sym_purem_40k`, `sym_purem_40k_n703`, `sym_long_abl_n704`, `car375`, `sym_long_abl`, `sym_long_abl2`, `chairs_released_n1317`, `chairs_sym_10k_n1317`, `chairs_control_10k_n1317`), checkpoints in `experiments_out/` (`sym_long`, `control_long`, `sym_fargate`, `chairs_sym_full`, `chairs_control_full`, `plane2`, `plane_cc`).
