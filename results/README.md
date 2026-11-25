# Results

| Folder | Contents |
|---|---|
| `raw/` | raw evaluation outputs (`*_objects.csv`, `*_views.csv.gz`): one row per test object or per object and view; every table and figure is computed from these |
| `final_numbers/` | all result tables (`.md` and `.csv`, each with its sources), and the compact LaTeX table |
| `final_figures/` | the figures of the report |
| `examples/` | qualitative examples: hidden-side comparison, internet photos, 360-degree videos |

Rebuild every table and figure from `raw/` (a laptop is enough, about two minutes):

```bash
python -m src.analysis.make_all_tables
python -m src.analysis.make_final_figures
python -m src.analysis.make_level1_compact
```

`raw/` is produced on a GPU by `python -m src.reduction.eval_budget` (and `src.analysis.sym_usage`);
the `--out` name of each run is the file prefix, e.g. `test_sym_anybudget`.
