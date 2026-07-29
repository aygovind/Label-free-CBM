"""Reusable analysis helpers for trained CBM run directories.

Everything is keyed off a single `Run` returned by `load_run(dir)`, so swapping models
means changing one path. The plotting functions previously lived inline in
evaluate_my_cbm.ipynb; they are here so the notebook stays a thin driver and so
multiple runs can be compared in one place.

    import cbm_analysis as ca
    run = ca.load_run("saved_models/bioclip_birds525__lam0p002")
    res = ca.evaluate(run)
    ca.sankey(run, "Felidae", "Canidae")
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import cbm
import data_utils

ARTIFACTS = ("W_c.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt",
             "concepts.txt", "args.txt")


class Run:
    """A loaded CBM run: model plus the labels needed to interpret its outputs."""

    def __init__(self, load_dir, model, concepts, classes, train_args, preprocess, device):
        self.load_dir = load_dir
        self.model = model
        self.concepts = concepts
        self.classes = classes
        self.train_args = train_args
        self.preprocess = preprocess
        self.device = device

    @property
    def name(self):
        return os.path.basename(self.load_dir.rstrip("/"))

    @property
    def dataset(self):
        return self.train_args["dataset"]

    @property
    def backbone(self):
        return self.train_args["backbone"]

    def __repr__(self):
        return "<Run {} backbone={} concepts={} classes={}>".format(
            self.name, self.backbone, len(self.concepts), len(self.classes))


def _read_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.read().split("\n") if line.strip()]


def _load_classes(load_dir, dataset, n_expected):
    """Prefer the run's own classes.txt; fall back to the dataset label file.

    Runs trained before classes.txt was written have no copy, so the label file is
    the only source. Warn on a length mismatch rather than silently mislabelling
    every plot -- a trailing newline in a label file inflates the class count.
    """
    path = os.path.join(load_dir, "classes.txt")
    if os.path.exists(path):
        classes = _read_lines(path)
    else:
        classes = _read_lines(data_utils.LABEL_FILES[dataset])
        print("note: no classes.txt in {}, using {}".format(
            load_dir, data_utils.LABEL_FILES[dataset]))
    if len(classes) != n_expected:
        print("WARNING: {} class names but the final layer has {} outputs -- "
              "labels may be misaligned".format(len(classes), n_expected))
    return classes


def load_run(load_dir, device=None):
    """Load a run directory produced by train_cbm.py."""
    missing = [f for f in ARTIFACTS if not os.path.exists(os.path.join(load_dir, f))]
    if missing:
        raise FileNotFoundError("{} is not a complete run dir; missing {}".format(
            load_dir, missing))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        train_args = json.load(f)

    model = cbm.load_cbm(load_dir, device)
    model.eval()

    concepts = _read_lines(os.path.join(load_dir, "concepts.txt"))
    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location="cpu")
    classes = _load_classes(load_dir, train_args["dataset"], W_g.shape[0])

    _, preprocess = data_utils.get_target_model(train_args["backbone"], device)
    return Run(load_dir, model, concepts, classes, train_args, preprocess, device)


def get_data(run, split="val", raw=False):
    """Eval split for this run. raw=True returns undecoded PIL images for display."""
    name = "{}_{}".format(run.dataset, split)
    if raw:
        import torchvision.transforms as T
        return data_utils.get_data(name, preprocess=T.Lambda(lambda x: x))
    return data_utils.get_data(name, run.preprocess)


class Results:
    """Per-example predictions and concept activations, in dataset order."""

    def __init__(self, run, split, preds, labels, concept_acts):
        self.run = run
        self.split = split
        self.preds = preds
        self.labels = labels
        self.concept_acts = concept_acts

    @property
    def accuracy(self):
        return (self.preds == self.labels).float().mean().item()

    @property
    def wrong_indices(self):
        return (self.preds != self.labels).nonzero(as_tuple=True)[0]

    def __repr__(self):
        return "<Results {} {} acc={:.4f} n={}>".format(
            self.run.name, self.split, self.accuracy, len(self.labels))


def evaluate(run, split="val", batch_size=256, num_workers=4):
    """Run the model over a split, keeping predictions and concept activations."""
    data = get_data(run, split)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    preds, labels, acts = [], [], []
    with torch.no_grad():
        for images, y in loader:
            logits, concept_act = run.model(images.to(run.device))
            preds.append(logits.argmax(dim=1).cpu())
            labels.append(y)
            acts.append(concept_act.cpu())

    return Results(run, split, torch.cat(preds), torch.cat(labels), torch.cat(acts))


def summarize(run, results=None, split="val"):
    """One-row summary dict; use with compare() across runs."""
    results = results or evaluate(run, split)
    W_g = run.model.final.weight.detach().cpu()
    nnz = (W_g.abs() > 1e-5).sum().item()
    return {
        "run": run.name,
        "backbone": run.backbone,
        "clip_name": run.train_args["clip_name"],
        "lam": run.train_args["lam"],
        "accuracy": round(results.accuracy, 4),
        "n_concepts": len(run.concepts),
        "frac_non_zero": round(nnz / W_g.numel(), 4),
        "concepts_per_class": round(nnz / W_g.shape[0], 1),
    }


def compare(load_dirs, split="val", device=None):
    """Summary table across several runs. Returns a DataFrame if pandas is available."""
    rows = []
    for d in load_dirs:
        try:
            rows.append(summarize(load_run(d, device), split=split))
        except Exception as exc:
            print("skipping {}: {}".format(d, exc))
    try:
        import pandas as pd
        return pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    except ImportError:
        return rows


def find_runs(save_dir="saved_models"):
    """Every complete run directory under save_dir."""
    if not os.path.isdir(save_dir):
        return []
    return sorted(os.path.join(save_dir, d) for d in os.listdir(save_dir)
                  if all(os.path.exists(os.path.join(save_dir, d, f)) for f in ARTIFACTS))


# --------------------------------------------------------------------- plotting

def _normalization(preprocess):
    """Recover (mean, std) from a preprocessing pipeline so images can be un-normalised.

    Each backbone family normalises differently -- CLIP, augreg_in21k and DINO all
    use different constants -- so hardcoding ImageNet values would tint the displayed
    images for most models.
    """
    from torchvision import transforms as T
    stack, seen = [preprocess], 0
    while stack and seen < 100:
        seen += 1
        t = stack.pop()
        if isinstance(t, T.Normalize):
            return np.array(t.mean), np.array(t.std)
        for attr in ("transforms", "transform"):
            sub = getattr(t, attr, None)
            if isinstance(sub, (list, tuple)):
                stack.extend(sub)
            elif sub is not None:
                stack.append(sub)
    print("note: no Normalize found in preprocess; showing images un-scaled")
    return np.zeros(3), np.ones(3)


def _to_displayable(img, preprocess):
    mean, std = _normalization(preprocess)
    arr = img.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(arr * std + mean, 0, 1)


def _top_contributions(run, concept_act, class_idx, top_k=8):
    """Concept contributions to one class logit, largest magnitude first."""
    weights = run.model.final.weight[class_idx].detach().cpu()
    contrib = (concept_act.detach().cpu() * weights).numpy()
    order = np.argsort(np.abs(contrib))[::-1][:top_k]
    labels = [("NOT " if concept_act[i] < 0 else "") + run.concepts[i] for i in order]
    return labels, contrib[order]


def plot_wrong_predictions(run, results, n=5, top_k=8, seed=None):
    """Misclassified examples: image alongside the concepts that drove the wrong call."""
    import matplotlib.pyplot as plt

    wrong = results.wrong_indices
    if len(wrong) == 0:
        print("no wrong predictions")
        return
    if seed is not None:
        torch.manual_seed(seed)
    chosen = wrong[torch.randperm(len(wrong))[:n]].tolist()

    data = get_data(run, results.split)
    fig, axes = plt.subplots(len(chosen), 2, figsize=(13, 3.2 * len(chosen)))
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(chosen):
        img, _ = data[idx]
        pred_i, true_i = int(results.preds[idx]), int(results.labels[idx])

        axes[row, 0].imshow(_to_displayable(img, run.preprocess))
        axes[row, 0].axis("off")
        axes[row, 0].set_title("#{}  true: {}\npredicted: {}".format(
            idx, run.classes[true_i], run.classes[pred_i]), fontsize=9)

        labels, values = _top_contributions(run, results.concept_acts[idx], pred_i, top_k)
        colors = ["tab:red" if v > 0 else "tab:blue" for v in values]
        axes[row, 1].barh(range(len(values))[::-1], values, color=colors)
        axes[row, 1].set_yticks(range(len(values))[::-1])
        axes[row, 1].set_yticklabels(labels, fontsize=8)
        axes[row, 1].axvline(0, color="k", lw=0.8)
        axes[row, 1].set_xlabel("contribution to predicted class", fontsize=8)

    plt.tight_layout()
    plt.show()
    print("{} wrong of {} ({:.2f}% accuracy)".format(
        len(wrong), len(results.labels), results.accuracy * 100))


def explain_example(run, idx, split="val", top_k=8):
    """Single-example view: image, top-2 predictions, concept contribution bars."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    data = get_data(run, split)
    raw = get_data(run, split, raw=True)

    x, true_i = data[idx]
    with torch.no_grad():
        logits, concept_act = run.model(x.unsqueeze(0).to(run.device))

    probs = torch.nn.functional.softmax(logits[0], dim=0)
    top_vals, top_classes = torch.topk(logits[0], k=min(2, len(run.classes)))

    try:
        display(raw[idx][0].resize([320, 320]))
    except Exception:
        plt.imshow(_to_displayable(x, run.preprocess)); plt.axis("off"); plt.show()

    print("#{}  true: {}".format(idx, run.classes[int(true_i)]))
    for rank, c in enumerate(top_classes.tolist()):
        print("  {}. {}  logit {:.3f}  p={:.3f}".format(
            rank + 1, run.classes[c], top_vals[rank], probs[c]))

    labels, values = _top_contributions(run, concept_act[0], int(top_classes[0]), top_k)
    colors = ["tab:red" if v > 0 else "tab:blue" for v in values]
    plt.figure(figsize=(7, 0.4 * len(values) + 1))
    plt.barh(range(len(values))[::-1], values, color=colors)
    plt.yticks(range(len(values))[::-1], labels, fontsize=9)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("contribution to {}".format(run.classes[int(top_classes[0])]))
    plt.tight_layout()
    plt.show()


def sankey(run, class_a, class_b, weight_cutoff=0.05, max_per_class=None):
    """Interactive concept -> class flow diagram from final-layer weights.

    Reproduces the paper's figure programmatically instead of pasting weights into
    sankeymatic.com. Negative weights render as "NOT <concept>", matching the paper.
    """
    import plotly.graph_objects as go

    for name in (class_a, class_b):
        if name not in run.classes:
            raise ValueError("{!r} not in classes; e.g. {}".format(
                name, run.classes[:5]))

    final_weight = run.model.final.weight.detach().cpu()
    edges = []
    for target_pos, class_name in enumerate([class_a, class_b]):
        weights = final_weight[run.classes.index(class_name)]
        keep = (weights.abs() > weight_cutoff).nonzero(as_tuple=True)[0]
        keep = keep[torch.argsort(weights[keep].abs(), descending=True)]
        if max_per_class:
            keep = keep[:max_per_class]
        for ci in keep.tolist():
            w = weights[ci].item()
            label = run.concepts[ci] if w > 0 else "NOT {}".format(run.concepts[ci])
            edges.append((label, target_pos, abs(w), w > 0))

    if not edges:
        print("no concepts above |weight| > {}; lower weight_cutoff".format(weight_cutoff))
        return

    concept_labels = list(dict.fromkeys(e[0] for e in edges))
    nodes = concept_labels + [class_a, class_b]
    index = {label: i for i, label in enumerate(nodes)}

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, pad=12, thickness=14),
        link=dict(
            source=[index[e[0]] for e in edges],
            target=[len(concept_labels) + e[1] for e in edges],
            value=[e[2] for e in edges],
            color=["rgba(214,39,40,0.4)" if e[3] else "rgba(31,119,180,0.4)" for e in edges],
        ),
    ))
    fig.update_layout(
        title="{}: concept -> class (|w| > {})".format(run.name, weight_cutoff),
        font_size=10, height=max(400, 18 * len(concept_labels)))
    fig.show()
    return fig


def concept_heatmap(run, results, n_examples=30, n_concepts=20):
    """Which concepts fire across many examples -- dataset-level companion to the bars."""
    import matplotlib.pyplot as plt

    acts = results.concept_acts
    top = torch.argsort(acts.abs().mean(dim=0), descending=True)[:n_concepts]
    subset = acts[:n_examples][:, top].numpy()
    scale = np.abs(subset).max() or 1.0

    plt.figure(figsize=(10, 8))
    plt.imshow(subset, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
    plt.colorbar(label="concept activation")
    plt.yticks(range(min(n_examples, len(subset))),
               ["#{} ({})".format(i, run.classes[int(results.labels[i])])
                for i in range(min(n_examples, len(subset)))], fontsize=7)
    plt.xticks(range(len(top)), [run.concepts[i] for i in top], rotation=90, fontsize=7)
    plt.title("{} -- most active concepts".format(run.name))
    plt.tight_layout()
    plt.show()
