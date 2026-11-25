# Reproduction of Splatter Image (official eval.py, all 250 views)

| objects | psnr | ssim | lpips |
|---|---|---|---|
| all 704 | 23.9991 | 0.9228 | 0.0770 |
| 703 (without blank car 375) | 23.9326 | 0.9227 | 0.0771 |

Paper (Table, SRN Cars): PSNR 24.00, SSIM 0.92, LPIPS 0.078.

**Sources:** `results/raw/raw_official/baseline_cars_full_scores.txt`
