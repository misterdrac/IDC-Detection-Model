# Legacy scripts (archive)

Reference copies of older runners. **Do not run from here.**

| File | Was at |
|------|--------|
| `model_gpu_embbedding.py` | repo root |
| `run_all_vm.sh` | `runners/linear_vm/` |
| `run_pipeline.py` | `runners/` |

**Current replacements:**

- Phase 1: `python runners/phase1_balance.py`
- Single backbone: `python runners/linear_vm/<backbone>.py`
- ConvNeXt FT: `python src/cnn/convnext_5fold_ft.py`
- Data split: `python src/data/split.py`
