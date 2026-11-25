# Level 2 - learned importance head

| model | criterion | budget | n_objects | psnr | ssim | lpips | dPSNR_vs_original_full | dPSNR_vs_original:contribution@0.1 | dPSNR_vs_original:contribution@0.05 | dPSNR_vs_original:contribution@0.01 |
|---|---|---|---|---|---|---|---|---|---|---|
| original | opacity | 0.100 | 703 | 19.843 | 0.885 | 0.113 | -4.055 ± 0.146 | — | — | — |
| original | opacity_area | 0.100 | 703 | 21.197 | 0.894 | 0.113 | -2.701 ± 0.094 | — | — | — |
| original | hybrid | 0.100 | 703 | 21.253 | 0.896 | 0.106 | -2.644 ± 0.104 | — | — | — |
| original | contribution | 0.100 | 703 | 20.287 | 0.895 | 0.108 | -3.611 ± 0.120 | — | — | — |
| head_A | learned | 0.100 | 703 | 21.038 | 0.895 | 0.108 | -2.859 ± 0.104 | +0.752 ± 0.064 | — | — |
| head_A | learned | 0.050 | 703 | 18.707 | 0.868 | 0.142 | -5.191 ± 0.152 | — | +0.573 ± 0.088 | — |
| head_A | learned | 0.010 | 703 | 15.891 | 0.829 | 0.211 | -8.007 ± 0.192 | — | — | +0.988 ± 0.100 |
| head:abl_reference | learned | 0.100 | 703 | 20.907 | 0.894 | 0.110 | -2.991 ± 0.108 | — | — | — |
| head:abl_inputs_gauss | learned | 0.100 | 703 | 20.901 | 0.894 | 0.111 | -2.996 ± 0.108 | — | — | — |
| head:abl_inputs_rgb | learned | 0.100 | 703 | 19.150 | 0.877 | 0.129 | -4.747 ± 0.144 | — | — | — |
| head:abl_loss_mse | learned | 0.100 | 703 | 20.744 | 0.893 | 0.113 | -3.153 ± 0.114 | — | — | — |
| head:abl_target_errsens | learned | 0.100 | 703 | 17.948 | 0.874 | 0.126 | -5.949 ± 0.177 | — | — | — |

(Validation overlaps: copy `runs/<run>/log.csv` from the cluster to `results/head_logs/<run>/log.csv`.)

**Sources:** `test_head_A_views.csv.gz`, `test_abl_reference_views.csv.gz`, `test_abl_inputs_gauss_views.csv.gz`, `test_abl_inputs_rgb_views.csv.gz`, `test_abl_loss_mse_views.csv.gz`, `test_abl_target_errsens_views.csv.gz`
