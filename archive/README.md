# Archive

Notebooks kept for reference but not part of the active pipeline. Nothing in the training or
evaluation path imports them.

| Notebook | Why archived |
|---|---|
| `temp_exp.ipynb` | Scratch experiments |
| `sanity_check.ipynb` | Ad-hoc checks; superseded by proper smoke tests |
| `image_editing.ipynb` | Upstream CIFAR10 model-editing demo (uses `iceberg_seed10/`), unrelated to BioCLIP/birds525 |
| `create_dataset.ipynb` | Dataset prep, superseded by `download_assets.ipynb` |

**To run one of these:** they use repo-root-relative imports (`import utils`, `import cbm`) and
paths (`data/...`), so start Jupyter from the repo root and add this as the first cell:

```python
import sys, os
sys.path.insert(0, "..")
os.chdir("..")
```

`iceberg_seed10/` stays at the repo root — `concept_ablation.ipynb`, which is still active, uses it.
