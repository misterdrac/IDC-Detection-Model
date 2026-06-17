# Phase 2 — deep fine-tuning (planned)

**Prerequisite:** Phase 1 complete; identify best model per data regime from
`reports/experiments/phase1_balance/comparison_latest.csv`.

## Goal

Go deeper into hyperparameter blocks for the top 1–2 setups (natural train vs balanced train), not just light head+last-stage FT.

## Suggested search blocks (ConvNeXt-Tiny FT)

| Block | Knobs |
|-------|--------|
| Optimizer | `lr_head`, `lr_backbone`, `weight_decay` |
| Schedule | `head_epochs`, `ft_epochs`, `finetune_last_stage` |
| Imbalance | `use_pos_weight`, `train_balance` (balanced sampler) |
| Resolution / batch | `image_size`, `batch_size`, `grad_accum_steps` |

## Implementation status

- [ ] Add CLI to `src/cnn/convnext_5fold_ft.py` (mirror linear script flags)
- [ ] `experiments/phase2_deep_ft/run_grid.py` — grid over blocks, same logging as phase 1
- [ ] Auto-select finalists from phase 1 comparison CSV

Until implemented, run ConvNeXt manually and copy metrics into `reports/experiments/phase2_deep_ft/`.
