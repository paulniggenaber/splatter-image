# Quality of the full model vs input-target angle

## a) Official protocol, all 250 views (703 objects)

| bin | views | psnr | ssim | lpips |
|---|---|---|---|---|
| 30-60 | 11951 | 26.684 | 0.956 | 0.040 |
| 60-90 | 24605 | 23.249 | 0.925 | 0.076 |
| 90-120 | 59052 | 23.100 | 0.913 | 0.089 |
| 120-150 | 75924 | 24.305 | 0.924 | 0.075 |
| 150-180 | 4218 | 25.071 | 0.936 | 0.064 |

## b) Our protocol, 25 views (703 objects)

| bin | views | psnr | lpips |
|---|---|---|---|
| 30-60 | 1406 | 26.634 | 0.039 |
| 60-90 | 2109 | 23.439 | 0.072 |
| 90-120 | 6327 | 23.175 | 0.087 |
| 120-150 | 7030 | 24.009 | 0.076 |
| 150-180 | 703 | 25.200 | 0.064 |

**Sources:** `results/raw/raw_official/viewpoint_gap_full704.csv.gz`, `budget_full_part0_views.csv.gz`, `budget_full_part1_views.csv.gz`, `budget_full_part2_views.csv.gz`, `budget_full_part3_views.csv.gz`, `budget_full_part_new0_views.csv.gz`, `budget_full_part_new1_views.csv.gz`, `budget_full_part_new2_views.csv.gz`, `budget_full_part_new3_views.csv.gz`
