# BioCLIP + Label-free CBM

A fork of [Label-free Concept Bottleneck Models](https://github.com/Trustworthy-ML-Lab/Label-free-CBM)
(Oikarinen et al., ICLR 2023) that uses **[BioCLIP](https://huggingface.co/imageomics/bioclip)**
as the backbone, to produce an interpretable concept-bottleneck version of BioCLIP for
fine-grained biological classification.

LF-CBM turns a frozen backbone into a CBM without labeled concept data: it learns a linear
projection from backbone features into a concept space defined by CLIP text embeddings, prunes
concepts that are weakly activated or poorly aligned, then fits a sparse linear layer from
concepts to classes.

**What this fork adds over upstream**
- BioCLIP as a backbone (`--backbone bioclip`) and as a concept encoder (`--clip_name bioclip`)
- `vit_in21k` and `dino_vitb16` backbones (the BioCLIP paper's ViT-B/16 baselines, via `timm`)
- Two new datasets: `birds525` (Kaggle BIRDS 525 SPECIES) and `treeoflife` (5 coarse taxa)
- `download_assets.ipynb` — one-shot fetch and layout of the checkpoint and datasets
- `parquet_to_imagefolder.py` — converts the HuggingFace parquet build of BIRDS 525 to ImageFolder

---

## Setup

### Environment

```bash
conda create -p /workspace/envs/lfcbm python=3.9 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate /workspace/envs/lfcbm

pip install -r requirements.txt
pip install open_clip_torch huggingface_hub kaggle
pip install numpy==1.26.4          # several deps still break on numpy 2.x
```

Register a Jupyter kernel (optional, for the notebooks):

```bash
pip install ipykernel
python -m ipykernel install --user --name lfcbm --display-name "Python (lfcbm)"
```

### Assets

Run **`download_assets.ipynb`** from the repo root. It downloads the BioCLIP checkpoint and the
BIRDS 525 dataset, reorganizes the dataset into the ImageFolder layout the code expects, and
regenerates `data/birds525.txt`. Every cell is idempotent.

It needs a Kaggle API token at `~/.kaggle/kaggle.json` (Kaggle → Settings → Create New Token),
with `chmod 600`.

Resulting layout:

```
/workspace/models/bioclip/open_clip_pytorch_model.bin
data/birds525/{train,val,test}/<CLASS NAME>/*.jpg
data/birds525.txt                       # 525 class names, ImageFolder order, no trailing newline
data/concept_sets/birds525_filtered.txt # 2011 concepts
```

Upstream datasets are still available via `download_cub.sh`, `download_models.sh`, and
`download_rn18_places.sh`.

### Adding a dataset

Register it in `data_utils.py`:

```python
DATASET_ROOTS["mydata_train"] = "data/mydata/train"
DATASET_ROOTS["mydata_val"]   = "data/mydata/val"
LABEL_FILES["mydata"]         = "data/mydata.txt"
```

The class-name file must have one line per class in `sorted()` directory order and **no trailing
newline** — `train_cbm.py` uses `len(classes)` to size the final layer, so a trailing newline
silently creates a phantom class.

---

## Training

Runs are defined by config files rather than long command lines:

```bash
python train_cbm.py --config bioclip_birds525
```

Every flag is still available and **overrides the config file**, which makes one-off variations
cheap without editing anything:

```bash
python train_cbm.py --config bioclip_birds525 --lam 0.002 --run_name quick_test
```

The equivalent fully-explicit invocation, if you prefer no config at all:

```bash
python train_cbm.py \
  --backbone bioclip \
  --clip_name bioclip \
  --feature_layer visual.ln_post \
  --dataset birds525 \
  --concept_set data/concept_sets/birds525_filtered.txt \
  --batch_size 256 \
  --print
```

Key arguments:

| Flag | Meaning |
|---|---|
| `--config` | Experiment config in `configs/experiments/`, by bare name or path |
| `--run_name` | Output directory under `--save_dir`. Defaults to a timestamp. |
| `--backbone` | Model made interpretable. `bioclip`, `clip_RN50`, `resnet50`, `resnet18_places`, … |
| `--clip_name` | CLIP model that defines the concept space. `bioclip`, `ViT-B/16`, … |
| `--feature_layer` | Backbone layer to tap. Used for all non-`clip_`-prefixed backbones, including `bioclip`. |
| `--concept_set` | Concept list. Defaults to `data/concept_sets/<dataset>_filtered.txt`. |
| `--clip_cutoff` | Drop concepts whose mean top-5 CLIP activation is below this (default `0.25`). |
| `--interpretability_cutoff` | Drop concepts whose projection similarity is below this (default `0.45`). |
| `--lam` | Sparsity on the final layer; higher is sparser (default `0.0007`). |
| `--n_iters` | GLM-SAGA iterations for the final layer (default `1000`). |
| `--activation_dir` | Cache for backbone/CLIP activations (default `saved_activations`). |
| `--print` | Log every concept as it is pruned. |

For fine-grained biological data the default cutoffs prune too aggressively; `--clip_cutoff 0.2
--interpretability_cutoff 0.3` is a reasonable starting point (see `commands.txt`).

**Activation caching.** The first run writes backbone and CLIP features to `--activation_dir`
and later runs reuse them, keyed by `(dataset, backbone, clip_name, concept_set, layer)`. It is
*not* concurrency-safe — two runs sharing a key will race on the same files, so do not run them
in parallel against one cache.

**Where the time actually goes.** Measured on birds525 (84k train images, 2011 concepts, 525
classes) on an RTX 2080 Ti:

| Stage | Time |
|---|---|
| Feature extraction, all encoders, train + val | **~25 min total** |
| GLM-SAGA final layer, `n_iters: 1000` | **~4.5 h per run** |

The solver dominates by roughly 12×, so caching activations saves far less than it appears to —
budget cluster time per *run*, not per dataset. `--n_iters` is the knob that controls cost.

### Outputs

Each run writes `saved_models/<dataset>_cbm_<timestamp>/`:

| File | Contents |
|---|---|
| `W_c.pt` | Concept projection weights (backbone dim → concepts) |
| `W_g.pt`, `b_g.pt` | Sparse final layer (concepts → classes) |
| `proj_mean.pt`, `proj_std.pt` | Concept-activation normalization statistics |
| `concepts.txt` | Surviving concepts after both pruning stages |
| `classes.txt` | Class names in final-layer output order |
| `args.txt` | Full argument dump for the run |
| `metrics.txt` | Accuracy, loss, and final-layer sparsity |
| `eval_metrics.json` | Written by `evaluate_cbm.py` / `run_suite.py`, not by training |

## Configs and suites

```
configs/base.yaml                 defaults + the authoritative list of legal keys
configs/experiments/*.yaml        one file per run, overriding base
configs/suites/*.yaml             an ordered list of runs -> one job
```

A config only needs the keys it changes; everything else falls through to `base.yaml`. Keys not
declared in `base.yaml` are **rejected**, so `lamda: 0.002` fails loudly instead of silently
doing nothing.

Shipped experiments:

| Config | What it is |
|---|---|
| `bioclip_birds525` | Primary run — BioCLIP backbone at `visual.ln_post`, BioCLIP concepts |
| `bioclip_birds525_vitb_concepts` | Ablation — same backbone, generic CLIP concept space |
| `rn50_birds525` | Baseline — upstream setup (ImageNet ResNet-50 + generic CLIP) |
| `bioclip_treeoflife` | Small/fast real-data check before spending GPU time on birds525 |
| `clip_vitb16_birds525` | Standard LF-CBM recipe — CLIP ViT-B/16 backbone + CLIP concepts |
| `in21k_birds525` | BioCLIP-paper baseline — ViT-B/16 supervised on ImageNet-21k |
| `dino_birds525` | BioCLIP-paper baseline — ViT-B/16 with DINO self-supervision |

### The backbone comparison

Five configs share a **ViT-B/16 concept space** and the default `lam`, so the only
variable is the backbone:

| Config | Backbone | Pretraining |
|---|---|---|
| `bioclip_birds525_vitb_concepts` | BioCLIP ViT-B/16 | TreeOfLife-10M, contrastive |
| `clip_vitb16_birds525` | CLIP ViT-B/16 | WIT-400M, contrastive |
| `in21k_birds525` | ViT-B/16 | ImageNet-21k, supervised |
| `dino_birds525` | ViT-B/16 | ImageNet, self-supervised |
| `rn50_birds525` | ResNet-50 | ImageNet, supervised |

The first four are all ViT-B/16, so those comparisons isolate pretraining from
architecture; `rn50_birds525` also changes architecture and is the upstream reference.

`bioclip_birds525` is **not** in this set — it uses a BioCLIP concept space, so it
differs on two axes at once. Compare it against `bioclip_birds525_vitb_concepts` to
isolate the effect of the concept encoder, and against the `lam` grid for sparsity.

A **suite** is what a cluster job runs: many trainings and evaluations in sequence on one GPU
allocation, one log, one summary. A `grid` sweeps a parameter without needing a file per point.

```yaml
# configs/suites/bioclip_sweep.yaml
runs:
  - experiment: bioclip_treeoflife
  - experiment: rn50_birds525
  - experiment: bioclip_birds525
    grid:
      lam: [0.0007, 0.002, 0.005]
  - experiment: bioclip_birds525_vitb_concepts
```

```bash
python run_suite.py --suite bioclip_sweep --dry_run   # list the 6 runs, execute nothing
python run_suite.py --suite bioclip_sweep
```

Each run trains, then evaluates, then records its result. A failing run is captured with its
traceback and the suite **continues** — one bad config doesn't cost you the rest of the sweep.
Pass `--stop_on_error` for the opposite behavior. Results land in
`suite_results/<suite>_<timestamp>/summary.json` alongside a printed table, and the process exits
non-zero if anything failed.

Because runs execute sequentially they share the activation cache safely. Group same-backbone
runs together in a suite so the expensive feature-extraction pass happens once.

## Evaluation

```bash
python evaluate_cbm.py --load_dir saved_models/bioclip_birds525
```

Reports validation accuracy, surviving concept count, and final-layer sparsity, and writes
`eval_metrics.json` into the run directory. `run_suite.py` calls this automatically after each
training run.

For anything visual, use **`evaluate_my_cbm.ipynb`**. It is a thin driver over
`cbm_analysis.py`, so switching models means editing one line:

```python
import cbm_analysis as ca

run = ca.load_run("saved_models/bioclip_birds525__lam0p0007")   # the only line to change
res = ca.evaluate(run)

ca.compare(ca.find_runs())                    # accuracy + sparsity table across all runs
ca.plot_wrong_predictions(run, res, n=5)      # misclassifications + what drove them
ca.explain_example(run, idx=20)               # the paper's per-decision bar plot
ca.sankey(run, "Felidae", "Canidae")          # concept -> class flows (needs plotly)
ca.concept_heatmap(run, res)                  # dataset-level concept activity
```

`load_run` validates that the directory is complete and recovers class names from
`classes.txt`, falling back to the dataset label file for runs trained before that file
existed. Image de-normalization is read out of each backbone's own preprocessing
pipeline, so displayed images are correct for CLIP, BioCLIP, `augreg_in21k`, and DINO
alike rather than assuming ImageNet constants.

`evaluate_cbm.ipynb` (upstream), `concept_ablation.ipynb`, and the notebooks under
`experiments/` cover concept ablation, manual weight editing, and intervention.

Notebooks that are no longer part of the active pipeline live in [`archive/`](archive/README.md).

## Concept sets

Pre-built sets for all datasets live in `data/concept_sets/`. To build a new one:

1. `GPT_initial_concepts.ipynb` — generate candidates with GPT for all three prompt types.
   Requires your own OpenAI API key and costs money.
2. `GPT_conceptset_processor.ipynb` — filter and deduplicate.

`ConceptNet_conceptset.ipynb` is a no-cost alternative that pulls concepts from ConceptNet.

---

## Smoke tests

Run before pushing or submitting a job:

```bash
python smoke_test.py
```

Three tiers, cheapest first. A tier whose dependencies are missing is reported as **SKIP**, never
as a pass:

| Tier | Needs | Checks |
|---|---|---|
| `config` | pyyaml only | Config merge, override precedence, typo rejection, suite expansion |
| `backbone` | torch, torchvision, open_clip, BioCLIP checkpoint | Feature dims (768-d at `ln_post`), activation cache keys don't collide |
| `e2e` | above + ~350 MB of first-run downloads | Full train → save → evaluate on a stratified CIFAR10 subset, CPU only |

```bash
python smoke_test.py --tier config    # seconds, no ML dependencies
python smoke_test.py --keep           # leave smoke_out/ for inspection
```

The `e2e` tier uses `configs/experiments/smoke_cifar10.yaml` — 200 train / 100 val images, 20
projection steps, 5 GLM-SAGA iterations, negative pruning cutoffs so no concept is dropped. Its
accuracy number is meaningless; what it proves is that the pipeline runs end to end and writes
every expected artifact. It also forces `num_workers=0`, since `utils.py` hardcodes 8.

Exit code is non-zero if anything failed.

## Running on Nautilus

The manifests live one directory up, in the `nrp/` parent folder (not currently version
controlled): `pvc.yaml`, `jupyter-gpu-pod.yaml`, `jupyter-pod.yaml`.

Interactive GPU session:

```bash
kubectl apply -f jupyter-gpu-pod.yaml
kubectl exec -it jupyter-gpu-pod -- bash
```

Inside the pod:

```bash
cd /workspace/Label-free-CBM
source /opt/conda/etc/profile.d/conda.sh && conda activate /workspace/envs/lfcbm
```

Port-forward Jupyter from a second terminal:

```bash
kubectl port-forward jupyter-gpu-pod 8888:8888
```

Long runs in an interactive pod should be detached so they survive a dropped shell:

```bash
nohup python run_suite.py --suite bioclip_sweep > /workspace/suite.log 2>&1 &
tail -f /workspace/suite.log
```

### Batch Jobs

`k8s/job-trial-treeoflife.yaml` runs a suite as a batch `Job` — the sanctioned way to do
non-interactive work on Nautilus (bare pods are capped at 6 hours).

```bash
kubectl apply -f k8s/job-trial-treeoflife.yaml
kubectl get job lfcbm-trial-treeoflife
kubectl logs -f job/lfcbm-trial-treeoflife
```

The job activates the conda env on the PVC, logs the git commit it is running, preflights the
BioCLIP checkpoint and dataset, runs the `config` smoke tier as a gate, then runs the suite. It
reads whatever code is in `/workspace/Label-free-CBM`, so `git pull` there first.

To run a different suite, change the `SUITE` env var in the manifest — `bioclip_sweep` for the
full set of experiments. Nothing else needs to change.

Policy compliance ([cluster policies](https://nrp.ai/documentation/userdocs/start/policies/)):
limits are within 20% of requests on cpu/memory/ephemeral-storage, GPU request equals limit,
`restartPolicy: Never` with a real terminating command, `activeDeadlineSeconds` bounds the run,
and `ttlSecondsAfterFinished` cleans up the finished Job. The affinity list excludes A100s, which
need prior access approval.

Tear down when finished — idle GPU pods are reclaimed:

```bash
kubectl delete pod jupyter-gpu-pod
```

The PVC `ayush-workspace` mounts at `/workspace` and persists across pods; everything outside it
is lost on delete.

---

## Known issues

**1. Hardcoded checkpoint path in the concept encoder.** The backbone now reads
`LFCBM_BIOCLIP_CKPT` (defaulting to `/workspace/models/bioclip/open_clip_pytorch_model.bin`), but
`utils.save_activations` still hardcodes that path for the BioCLIP *concept* encoder.
`DATASET_ROOTS` also uses paths relative to the repo root.

**2. `num_workers=8` is hardcoded** in `utils.py` DataLoaders. The GPU pod requests 6 Gi of
`/dev/shm`; lower this if you hit shared-memory errors.

**3. Concept encoder and backbone are the same model** under the default config
(`--backbone bioclip --clip_name bioclip`). This is intended — the point is to make BioCLIP
itself interpretable — but it means `W_c` can partly recover the concept text embeddings
directly. A `--clip_name ViT-B/16` ablation quantifies how much the domain-tuned concept
grounding is contributing.

---

## Sources

BioCLIP: https://huggingface.co/imageomics/bioclip ·
BIRDS 525: https://www.kaggle.com/datasets/gpiosenka/100-bird-species ·
CUB: https://www.vision.caltech.edu/datasets/cub_200_2011/ ·
Sparse final layer: https://github.com/MadryLab/glm_saga ·
CLIP: https://github.com/openai/CLIP ·
Barplots adapted from https://github.com/slundberg/shap

## Cite the original work

T. Oikarinen, S. Das, L. Nguyen and T.-W. Weng,
[*Label-free Concept Bottleneck Models*](https://openreview.net/pdf?id=FlCg47MNvBA), ICLR 2023.

```
@inproceedings{oikarinenlabel,
  title={Label-free Concept Bottleneck Models},
  author={Oikarinen, Tuomas and Das, Subhro and Nguyen, Lam M and Weng, Tsui-Wei},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
