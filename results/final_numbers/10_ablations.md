# Ablations: final model (any budget + symmetry) and training mask

| ablation | criterion | budget | n_objects | psnr | lpips | dPSNR_vs_original_full | dPSNR_vs_reference | reference |
|---|---|---|---|---|---|---|---|---|
| test-time rule: other criteria | full | 1.000 | 703 | 23.824 | 0.077 | -0.074 ± 0.032 | +0.000 ± 0.000 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | hybrid | 0.020 | 703 | 19.744 | 0.148 | -4.154 ± 0.136 | +0.000 ± 0.000 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | hybrid | 0.050 | 703 | 22.211 | 0.115 | -1.687 ± 0.078 | +0.000 ± 0.000 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | hybrid | 0.100 | 703 | 23.280 | 0.093 | -0.618 ± 0.044 | +0.000 ± 0.000 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | hybrid | 0.200 | 703 | 23.789 | 0.078 | -0.109 ± 0.033 | +0.000 ± 0.000 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | hybrid | 0.250 | 703 | 23.823 | 0.077 | -0.075 ± 0.032 | +0.000 ± 0.000 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | hybrid | 0.500 | 703 | 23.824 | 0.077 | -0.074 ± 0.032 | — | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | opacity | 0.020 | 703 | 17.486 | 0.156 | -6.411 ± 0.185 | -2.257 ± 0.086 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | opacity | 0.050 | 703 | 20.744 | 0.117 | -3.153 ± 0.128 | -1.466 ± 0.075 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | opacity | 0.100 | 703 | 23.106 | 0.091 | -0.791 ± 0.065 | -0.173 ± 0.044 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | opacity | 0.200 | 703 | 23.810 | 0.077 | -0.088 ± 0.033 | +0.021 ± 0.003 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | opacity | 0.250 | 703 | 23.823 | 0.077 | -0.074 ± 0.032 | +0.001 ± 0.000 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | opacity | 0.500 | 703 | 23.824 | 0.077 | -0.074 ± 0.032 | — | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | opacity_area | 0.020 | 703 | 19.288 | 0.170 | -4.609 ± 0.149 | -0.456 ± 0.061 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | opacity_area | 0.050 | 703 | 21.451 | 0.131 | -2.446 ± 0.098 | -0.759 ± 0.049 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | opacity_area | 0.100 | 703 | 22.932 | 0.104 | -0.966 ± 0.055 | -0.348 ± 0.024 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | opacity_area | 0.200 | 703 | 23.748 | 0.081 | -0.150 ± 0.033 | -0.041 ± 0.005 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | opacity_area | 0.250 | 703 | 23.821 | 0.077 | -0.077 ± 0.032 | -0.002 ± 0.001 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | opacity_area | 0.500 | 703 | 23.824 | 0.077 | -0.074 ± 0.032 | — | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | random | 0.020 | 703 | 14.704 | 0.224 | -9.194 ± 0.230 | -5.040 ± 0.154 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | random | 0.050 | 703 | 16.314 | 0.192 | -7.584 ± 0.204 | -5.897 ± 0.170 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | random | 0.100 | 703 | 18.363 | 0.167 | -5.534 ± 0.164 | -4.916 ± 0.147 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | random | 0.200 | 703 | 20.941 | 0.136 | -2.956 ± 0.109 | -2.847 ± 0.106 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | random | 0.250 | 703 | 21.741 | 0.125 | -2.157 ± 0.090 | -2.082 ± 0.087 | Sym+AnyBudget (hybrid, own plane) |
| test-time rule: other criteria | random | 0.500 | 703 | 23.486 | 0.093 | -0.411 ± 0.041 | — | Sym+AnyBudget (hybrid, own plane) |
| TRUE mirror plane | full | 1.000 | 703 | 23.983 | 0.076 | +0.086 ± 0.035 | +0.160 ± 0.025 | Sym+AnyBudget (hybrid, own plane) |
| TRUE mirror plane | hybrid | 0.050 | 703 | 22.290 | 0.114 | -1.607 ± 0.079 | +0.079 ± 0.016 | Sym+AnyBudget (hybrid, own plane) |
| TRUE mirror plane | hybrid | 0.100 | 703 | 23.407 | 0.093 | -0.490 ± 0.045 | +0.128 ± 0.022 | Sym+AnyBudget (hybrid, own plane) |
| TRUE mirror plane | hybrid | 0.200 | 703 | 23.951 | 0.078 | +0.053 ± 0.035 | +0.162 ± 0.025 | Sym+AnyBudget (hybrid, own plane) |
| twins switched off | full | 1.000 | 703 | 23.407 | 0.084 | -0.490 ± 0.030 | -0.416 ± 0.031 | Sym+AnyBudget (hybrid, own plane) |
| twins switched off | hybrid | 0.050 | 703 | 21.201 | 0.126 | -2.697 ± 0.097 | -1.010 ± 0.045 | Sym+AnyBudget (hybrid, own plane) |
| twins switched off | hybrid | 0.100 | 703 | 22.561 | 0.103 | -1.337 ± 0.059 | -0.718 ± 0.039 | Sym+AnyBudget (hybrid, own plane) |
| twins switched off | hybrid | 0.200 | 703 | 23.351 | 0.086 | -0.547 ± 0.032 | -0.438 ± 0.033 | Sym+AnyBudget (hybrid, own plane) |
| float16 storage | full | 1.000 | 703 | 23.823 | 0.077 | -0.074 ± 0.032 | -0.000 ± 0.000 | Sym+AnyBudget (hybrid, own plane) |
| float16 storage | hybrid | 0.050 | 703 | 22.211 | 0.115 | -1.687 ± 0.078 | +0.000 ± 0.000 | Sym+AnyBudget (hybrid, own plane) |
| float16 storage | hybrid | 0.100 | 703 | 23.279 | 0.093 | -0.618 ± 0.044 | -0.000 ± 0.000 | Sym+AnyBudget (hybrid, own plane) |
| float16 storage | hybrid | 0.200 | 703 | 23.789 | 0.078 | -0.109 ± 0.033 | -0.000 ± 0.000 | Sym+AnyBudget (hybrid, own plane) |
| FT-10 trained with a RANDOM mask | full | 1.000 | 703 | 19.423 | 0.142 | -4.475 ± 0.133 | -3.291 ± 0.088 | FT10-hybrid (hybrid) |
| FT-10 trained with a RANDOM mask | hybrid | 0.050 | 703 | 19.868 | 0.174 | -4.030 ± 0.134 | -1.438 ± 0.073 | FT10-hybrid (hybrid) |
| FT-10 trained with a RANDOM mask | hybrid | 0.100 | 703 | 20.183 | 0.158 | -3.715 ± 0.122 | -2.975 ± 0.098 | FT10-hybrid (hybrid) |
| FT-10 trained with a RANDOM mask | random | 0.050 | 703 | 20.256 | 0.165 | -3.641 ± 0.129 | -1.049 ± 0.073 | FT10-hybrid (hybrid) |
| FT-10 trained with a RANDOM mask | random | 0.100 | 703 | 22.017 | 0.145 | -1.880 ± 0.089 | -1.141 ± 0.065 | FT10-hybrid (hybrid) |

## How the symmetric models use their twins (fraction of kept originals on the camera side, twin switch)

| model | budget | near_side_kept | near_side_all | switch_kept | twins_on_kept |
|---|---|---|---|---|---|
| Sym+AnyBudget | 0.050 | 0.656 | 0.622 | 0.252 | 0.207 |
| Sym+AnyBudget | 0.100 | 0.652 | 0.622 | 0.230 | 0.176 |
| Sym+AnyBudget | 0.200 | 0.628 | 0.622 | 0.216 | 0.163 |
| Sym+AnyBudget | 1.000 | 0.622 | 0.622 | 0.213 | 0.162 |
| Sym+FT10 | 0.050 | 0.649 | 0.604 | 0.208 | 0.149 |
| Sym+FT10 | 0.100 | 0.663 | 0.604 | 0.205 | 0.145 |
| Sym+FT10 | 0.200 | 0.639 | 0.604 | 0.223 | 0.172 |
| Sym+FT10 | 1.000 | 0.604 | 0.604 | 0.220 | 0.177 |
| sym_long (no budget training) | 0.050 | 0.634 | 0.584 | 0.147 | 0.068 |
| sym_long (no budget training) | 0.100 | 0.629 | 0.584 | 0.141 | 0.069 |
| sym_long (no budget training) | 0.200 | 0.612 | 0.584 | 0.141 | 0.076 |
| sym_long (no budget training) | 1.000 | 0.584 | 0.584 | 0.137 | 0.073 |

**Sources:** `test_symany_criteria_views.csv.gz`, `test_symany_oracle_views.csv.gz`, `test_symany_twinsoff_views.csv.gz`, `test_symany_fp16_views.csv.gz`, `test_ft10_random_views.csv.gz`, `sym_usage_summary.csv`
