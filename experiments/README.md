# Experiments

Study orchestration (phase 1 balance comparison, phase 2 deep FT grid).

See **`.docs/AGENDA.md`** for local workflow notes (gitignored).

```bash
python runners/phase1_balance.py
python experiments/phase1_balance/run_all.py --dry-run
python3 experiments/<phase>/aggregate_results.py
```

Outputs: `experiments/results/<phase>/<run_id>/` (gitignored) and `reports/experiments/<phase>/comparison_latest.csv` (gitignored; VM only).
