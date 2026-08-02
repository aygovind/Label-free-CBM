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

    def __init__(self, run, split, preds, labels, concept_acts, logits=None):
        self.run = run
        self.split = split
        self.preds = preds
        self.labels = labels
        self.concept_acts = concept_acts
        self.logits = logits

    @property
    def accuracy(self):
        return (self.preds == self.labels).float().mean().item()

    @property
    def correct(self):
        return self.preds == self.labels

    @property
    def wrong_indices(self):
        return (self.preds != self.labels).nonzero(as_tuple=True)[0]

    @property
    def confidence(self):
        """Softmax probability of the predicted class, per example.

        Ranking by this is what separates "the model was sure and right" from
        "sure and wrong" -- the second group is where the interesting failures are.
        """
        if self.logits is None:
            raise RuntimeError("no logits stored; re-run evaluate() to populate them")
        return torch.softmax(self.logits, dim=1).max(dim=1).values

    def __repr__(self):
        return "<Results {} {} acc={:.4f} n={}>".format(
            self.run.name, self.split, self.accuracy, len(self.labels))


def evaluate(run, split="val", batch_size=256, num_workers=4):
    """Run the model over a split, keeping predictions and concept activations."""
    data = get_data(run, split)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    preds, labels, acts, all_logits = [], [], [], []
    with torch.no_grad():
        for images, y in loader:
            logits, concept_act = run.model(images.to(run.device))
            preds.append(logits.argmax(dim=1).cpu())
            labels.append(y)
            acts.append(concept_act.cpu())
            all_logits.append(logits.cpu())

    return Results(run, split, torch.cat(preds), torch.cat(labels),
                   torch.cat(acts), torch.cat(all_logits))


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


# --------------------------------------------------------------- static sankey

def _save_fig(fig, save_path, dpi=150):
    """Write a figure, creating the parent directory so notebook paths just work."""
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print("saved", save_path)


def _flow_path(x0, x1, top0, bot0, top1, bot1):
    """Cubic-Bezier ribbon joining a slice of a left node to a slice of a right one."""
    from matplotlib.path import Path
    mid = (x0 + x1) / 2.0
    verts = [(x0, top0),
             (mid, top0), (mid, top1), (x1, top1),
             (x1, bot1),
             (mid, bot1), (mid, bot0), (x0, bot0),
             (x0, top0)]
    codes = [Path.MOVETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    return Path(verts, codes)


def sankey_static(run, classes, weight_cutoff=0.05, max_per_class=12,
                  save_path=None, figsize=None, gap=0.015):
    """Paper-style concept -> class flow diagram, drawn in matplotlib.

    Same content as sankey() but with no plotly dependency and a static figure that
    drops straight into a paper. Ribbon width is |w| from the sparse final layer;
    red raises the class logit, blue lowers it, and negative weights are labelled
    "NOT <concept>" following the paper's convention.

    Concepts are ordered by the class they most influence, which keeps each class's
    band contiguous and stops the ribbons crossing into an unreadable tangle.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if isinstance(classes, str):
        classes = [classes]
    for name in classes:
        if name not in run.classes:
            raise ValueError("{!r} not in classes; e.g. {}".format(name, run.classes[:5]))

    final_weight = run.model.final.weight.detach().cpu()

    edges = []   # (concept_label, class_position, |w|, is_positive)
    for pos, class_name in enumerate(classes):
        weights = final_weight[run.classes.index(class_name)]
        keep = (weights.abs() > weight_cutoff).nonzero(as_tuple=True)[0]
        keep = keep[torch.argsort(weights[keep].abs(), descending=True)]
        if max_per_class:
            keep = keep[:max_per_class]
        for ci in keep.tolist():
            w = weights[ci].item()
            label = run.concepts[ci] if w > 0 else "NOT {}".format(run.concepts[ci])
            edges.append((label, pos, abs(w), w > 0))

    if not edges:
        print("no concepts above |weight| > {}; lower weight_cutoff".format(weight_cutoff))
        return None

    concept_flow, concept_home = {}, {}
    for label, pos, w, _ in edges:
        concept_flow[label] = concept_flow.get(label, 0.0) + w
        if w > concept_home.get(label, (None, -1.0))[1]:
            concept_home[label] = (pos, w)
    concept_labels = sorted(concept_flow,
                            key=lambda l: (concept_home[l][0], -concept_flow[l]))

    class_flow = {}
    for _, pos, w, _ in edges:
        class_flow[pos] = class_flow.get(pos, 0.0) + w

    def _stack(keys, flows):
        """Assign each node a [top, bottom] band, normalised to fill height 1."""
        total = sum(flows[k] for k in keys)
        span = 1.0 - gap * max(len(keys) - 1, 0)
        spans, y = {}, 1.0
        for k in keys:
            h = span * flows[k] / total if total else 0.0
            spans[k] = [y, y - h]
            y -= h + gap
        return spans

    left = _stack(concept_labels, concept_flow)
    right = _stack(list(range(len(classes))), class_flow)

    if figsize is None:
        figsize = (11, max(4.0, 0.34 * len(concept_labels)))
    fig, ax = plt.subplots(figsize=figsize)

    x_left, x_right, node_w = 0.30, 0.70, 0.012

    # ribbons first so the node rectangles draw on top of their ends
    left_cursor = {k: left[k][0] for k in left}
    right_cursor = {k: right[k][0] for k in right}
    for label, pos, w, positive in sorted(edges, key=lambda e: (concept_labels.index(e[0]))):
        scale_l = (left[label][0] - left[label][1]) / concept_flow[label]
        scale_r = (right[pos][0] - right[pos][1]) / class_flow[pos]
        h_l, h_r = w * scale_l, w * scale_r
        t0, b0 = left_cursor[label], left_cursor[label] - h_l
        t1, b1 = right_cursor[pos], right_cursor[pos] - h_r
        left_cursor[label], right_cursor[pos] = b0, b1
        ax.add_patch(patches.PathPatch(
            _flow_path(x_left + node_w, x_right, t0, b0, t1, b1),
            facecolor="#d62728" if positive else "#1f77b4",
            alpha=0.45, edgecolor="none"))

    for label in concept_labels:
        top, bot = left[label]
        ax.add_patch(patches.Rectangle((x_left, bot), node_w, top - bot,
                                       facecolor="#444444", edgecolor="none"))
        ax.text(x_left - 0.012, (top + bot) / 2, label, ha="right", va="center", fontsize=9)

    for pos, class_name in enumerate(classes):
        top, bot = right[pos]
        ax.add_patch(patches.Rectangle((x_right, bot), node_w, top - bot,
                                       facecolor="#222222", edgecolor="none"))
        ax.text(x_right + node_w + 0.012, (top + bot) / 2, class_name,
                ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, 1.04)
    ax.axis("off")
    ax.set_title("{} -- concept contributions (|w| > {})\nred: raises logit    blue: lowers logit"
                 .format(run.name, weight_cutoff), fontsize=11)
    plt.tight_layout()
    if save_path:
        _save_fig(fig, save_path, dpi=200)
    plt.show()
    return fig


# ------------------------------------------------------------------- collages

def _collage_image(run, raw_data, proc_data, idx, size=224):
    """Undecoded image when available -- avoids showing the normalised, cropped tensor."""
    try:
        return np.asarray(raw_data[idx][0].convert("RGB").resize((size, size))) / 255.0
    except Exception:
        if proc_data is None or run is None:
            raise
        return _to_displayable(proc_data[idx][0], run.preprocess)


def select_examples(results, mode="random", n=10, seed=0):
    """Indices for one collage. Modes: random, confident_correct, confident_wrong."""
    if mode == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.randperm(len(results.labels), generator=g)[:n].tolist()

    conf = results.confidence
    if mode == "confident_correct":
        pool = results.correct.nonzero(as_tuple=True)[0]
    elif mode == "confident_wrong":
        pool = (~results.correct).nonzero(as_tuple=True)[0]
    else:
        raise ValueError("mode must be random, confident_correct or confident_wrong")

    if len(pool) == 0:
        return []
    order = torch.argsort(conf[pool], descending=True)
    return pool[order][:n].tolist()


def result_collage(run, results, mode="random", n=10, seed=0, ncols=5,
                   save_path=None, title=None):
    """Grid of example predictions, captioned with truth, prediction and confidence.

    Green frame = correct, red = wrong, so a "confident_wrong" sheet reads as failures
    at a glance without checking every caption.
    """
    import matplotlib.pyplot as plt

    chosen = select_examples(results, mode=mode, n=n, seed=seed)
    if not chosen:
        print("no examples for mode={}".format(mode))
        return None

    raw_data = get_data(run, results.split, raw=True)
    proc_data = get_data(run, results.split)
    conf = results.confidence

    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[len(chosen):]:
        ax.axis("off")

    for ax, idx in zip(axes, chosen):
        true_i, pred_i = int(results.labels[idx]), int(results.preds[idx])
        ok = true_i == pred_i
        ax.imshow(_collage_image(run, raw_data, proc_data, idx))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#2ca02c" if ok else "#d62728")
            spine.set_linewidth(3)
        caption = "#{}  p={:.2f}\ntrue: {}".format(idx, conf[idx].item(), run.classes[true_i])
        if not ok:
            caption += "\npred: {}".format(run.classes[pred_i])
        ax.set_title(caption, fontsize=7.5, color="#2ca02c" if ok else "#d62728")

    fig.suptitle(title or "{} -- {} ({})".format(run.name, mode.replace("_", " "), results.split),
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        _save_fig(fig, save_path, dpi=150)
    plt.show()
    return fig


def save_collages(run, results, out_dir=None, n=10, seed=0):
    """All three collages -- random, confidently correct, confidently wrong -- to disk."""
    out_dir = out_dir or os.path.join(run.load_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for mode in ("random", "confident_correct", "confident_wrong"):
        path = os.path.join(out_dir, "collage_{}.png".format(mode))
        result_collage(run, results, mode=mode, n=n, seed=seed, save_path=path)
        paths.append(path)
    return paths


# ------------------------------------------------- per-class accuracy / confusion

def per_class_accuracy(run, results):
    """Per-class accuracy rows, worst first: class, accuracy, correct, support."""
    labels, preds = results.labels.numpy(), results.preds.numpy()
    rows = []
    for ci, name in enumerate(run.classes):
        mask = labels == ci
        support = int(mask.sum())
        if support == 0:
            continue
        n_correct = int((preds[mask] == ci).sum())
        rows.append({"class": name, "class_idx": ci, "accuracy": n_correct / support,
                     "correct": n_correct, "support": support})
    rows.sort(key=lambda r: (r["accuracy"], -r["support"]))
    return rows


def _support_warning(rows):
    """Birds525 ships ~5 val images per class, which makes per-class accuracy coarse."""
    supports = sorted(r["support"] for r in rows)
    median = supports[len(supports) // 2]
    if median < 10:
        print("NOTE: median support is {} images/class -- per-class accuracy can only take "
              "{} distinct values, so 'worst classes' is noisy. Treat as a shortlist to "
              "inspect, not a ranking.".format(median, median + 1))
    return median


def plot_class_accuracy(run, results, worst_k=25, save_path=None):
    """Distribution of per-class accuracy, plus the worst-k classes by name."""
    import matplotlib.pyplot as plt

    rows = per_class_accuracy(run, results)
    _support_warning(rows)
    worst = rows[:worst_k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(5, 0.32 * worst_k)),
                                   gridspec_kw={"width_ratios": [1, 1.5]})

    accs = [r["accuracy"] for r in rows]
    ax1.hist(accs, bins=20, color="#4c72b0", edgecolor="white")
    ax1.axvline(results.accuracy, color="#d62728", ls="--",
                label="overall {:.1f}%".format(results.accuracy * 100))
    ax1.set_xlabel("per-class accuracy"); ax1.set_ylabel("classes")
    ax1.set_title("distribution over {} classes".format(len(rows)))
    ax1.legend(fontsize=8)

    ypos = range(len(worst))[::-1]
    ax2.barh(list(ypos), [r["accuracy"] for r in worst], color="#d62728")
    ax2.set_yticks(list(ypos))
    ax2.set_yticklabels(["{} ({}/{})".format(r["class"], r["correct"], r["support"])
                         for r in worst], fontsize=7.5)
    ax2.set_xlabel("accuracy"); ax2.set_xlim(0, 1)
    ax2.set_title("worst {} classes".format(len(worst)))

    fig.suptitle("{} -- per-class accuracy ({})".format(run.name, results.split), fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        _save_fig(fig, save_path, dpi=150)
    plt.show()
    return rows


def confused_pairs(run, results, k=20):
    """Most frequent (true -> predicted) mistakes, as a list of dicts."""
    import collections
    counts = collections.Counter(
        (int(t), int(p)) for t, p in zip(results.labels, results.preds) if t != p)
    return [{"true": run.classes[t], "pred": run.classes[p], "count": n}
            for (t, p), n in counts.most_common(k)]


def plot_confusion(run, results, worst_k=25, save_path=None, annotate=True):
    """Confusion submatrix over the worst-k classes and whatever they get mistaken for.

    The full matrix is 525x525 and unreadable, and with ~5 images per class it is
    almost entirely zeros -- so this restricts rows to the classes that actually fail
    and columns to the predictions they actually receive.
    """
    import matplotlib.pyplot as plt

    rows = per_class_accuracy(run, results)
    _support_warning(rows)
    worst = rows[:worst_k]
    row_idx = [r["class_idx"] for r in worst]

    labels, preds = results.labels.numpy(), results.preds.numpy()
    col_counts = {}
    for ci in row_idx:
        for p in preds[labels == ci]:
            col_counts[int(p)] = col_counts.get(int(p), 0) + 1
    keep = sorted(col_counts, key=lambda c: -col_counts[c])[:worst_k + 10]
    col_idx = sorted(set(keep) | set(row_idx), key=lambda c: run.classes[c])
    col_pos = {c: j for j, c in enumerate(col_idx)}

    # capping the columns can exclude a rare prediction one of these rows actually made;
    # bucket those into a trailing column so every row still sums to its support
    spill = sum(1 for ci in row_idx for p in preds[labels == ci] if int(p) not in col_pos)
    mat = np.zeros((len(row_idx), len(col_idx) + (1 if spill else 0)), dtype=int)
    for i, ci in enumerate(row_idx):
        for p in preds[labels == ci]:
            mat[i, col_pos.get(int(p), len(col_idx))] += 1

    col_names = [run.classes[c] for c in col_idx] + (["(other)"] if spill else [])
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(col_names)), max(6, 0.36 * len(row_idx))))
    im = ax.imshow(mat, cmap="Reds", aspect="auto")
    fig.colorbar(im, ax=ax, label="images", fraction=0.025)

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_idx)))
    ax.set_yticklabels(["{} ({}/{})".format(r["class"], r["correct"], r["support"])
                        for r in worst], fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true (worst classes)")

    # mark the diagonal so correct predictions are distinguishable from confusions
    for i, ci in enumerate(row_idx):
        if ci in col_pos:
            ax.add_patch(plt.Rectangle((col_pos[ci] - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="#2ca02c", lw=1.6))
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j]:
                    ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=6.5,
                            color="white" if mat[i, j] > mat.max() * 0.6 else "black")

    ax.set_title("{} -- confusions for the {} weakest classes ({})\n"
                 "green box = correct cell".format(run.name, len(row_idx), results.split),
                 fontsize=11)
    plt.tight_layout()
    if save_path:
        _save_fig(fig, save_path, dpi=150)
    plt.show()
    return mat


# ------------------------------------------------- cross-model qualitative figure

class ModelOutputs:
    """One run's predictions, retained after the model itself has been released.

    evaluate_many() builds these so several runs can be compared without five
    backbones resident on the GPU at once.
    """

    def __init__(self, name, backbone, dataset, classes, concepts, final_weight,
                 preds, labels, logits, concept_acts=None):
        self.name = name
        self.backbone = backbone
        self.dataset = dataset
        self.classes = classes
        self.concepts = concepts
        self.final_weight = final_weight
        self.preds = preds
        self.labels = labels
        self.logits = logits
        self.concept_acts = concept_acts

    @property
    def accuracy(self):
        return (self.preds == self.labels).float().mean().item()

    @property
    def correct(self):
        return self.preds == self.labels

    @property
    def confidence(self):
        return torch.softmax(self.logits, dim=1).max(dim=1).values

    def top_concepts(self, idx, k=2):
        """Highest-magnitude concept contributions for one example's prediction."""
        if self.concept_acts is None:
            return []
        weights = self.final_weight[int(self.preds[idx])]
        contrib = self.concept_acts[idx] * weights
        order = torch.argsort(contrib.abs(), descending=True)[:k]
        return [("" if contrib[i] >= 0 else "NOT ") + self.concepts[i] for i in order]

    def __repr__(self):
        return "<ModelOutputs {} acc={:.4f}>".format(self.name, self.accuracy)


def evaluate_many(load_dirs, split="val", device=None, batch_size=256,
                  keep_concept_acts=False):
    """Evaluate several runs in turn, freeing each backbone before loading the next.

    Five ViT-B/16 backbones will not fit comfortably on an 11GB card together, so only
    predictions and the final-layer weights are kept. Pass keep_concept_acts=True if you
    want per-example concept attributions (~20MB per run on birds525).
    """
    outs = []
    for d in load_dirs:
        run = load_run(d, device)
        res = evaluate(run, split=split, batch_size=batch_size)
        outs.append(ModelOutputs(
            run.name, run.backbone, run.dataset, run.classes, run.concepts,
            run.model.final.weight.detach().cpu(),
            res.preds, res.labels, res.logits,
            res.concept_acts if keep_concept_acts else None))
        print("  {:42s} acc {:.2f}%".format(run.name, outs[-1].accuracy * 100))
        del res, run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outs


def _pick(outs, name):
    for o in outs:
        if o.name == name or o.backbone == name:
            return o
    raise ValueError("{!r} matched no run; have {}".format(
        name, [o.name for o in outs]))


def story_examples(outs, strong=None, domain=None):
    """One representative index per narrative mode, plus how many images qualify.

    Candidates are ranked so the chosen image makes the point vividly -- for the
    "wins" modes that means both models were confident, not marginal.
    """
    ranked = sorted(outs, key=lambda o: -o.accuracy)
    domain_o = (_pick(outs, domain) if domain else
                next((o for o in outs if o.backbone == "bioclip"), ranked[-1]))
    # strong must differ from domain, or both "wins" rows are empty by construction
    # -- which happens whenever the domain model is also the most accurate one
    strong_o = (_pick(outs, strong) if strong else
                next(o for o in ranked if o is not domain_o))
    if strong_o is domain_o:
        raise ValueError("strong and domain resolved to the same run ({}); "
                         "pass them explicitly".format(strong_o.name))

    preds = torch.stack([o.preds for o in outs])
    correct = torch.stack([o.correct for o in outs])
    conf = torch.stack([o.confidence for o in outs])
    n = preds.shape[1]

    n_distinct = torch.tensor([len(set(preds[:, i].tolist())) for i in range(n)])
    s_ok, d_ok = strong_o.correct, domain_o.correct
    s_conf, d_conf = strong_o.confidence, domain_o.confidence

    modes = {
        "max_disagreement": (n_distinct >= 2, n_distinct.float() + conf.mean(0)),
        "strong_wins": (s_ok & ~d_ok, s_conf + d_conf),
        "domain_wins": (d_ok & ~s_ok, s_conf + d_conf),
        "unanimous_failure": (~correct.any(0), conf.mean(0)),
    }

    out = {}
    for mode, (mask, score) in modes.items():
        pool = mask.nonzero(as_tuple=True)[0]
        if len(pool) == 0:
            out[mode] = {"index": None, "count": 0}
        else:
            best = pool[torch.argmax(score[pool])].item()
            out[mode] = {"index": best, "count": int(len(pool))}
    out["_strong"], out["_domain"] = strong_o.name, domain_o.name
    return out


def story_figure(outs, split="val", strong=None, domain=None, top_concepts=0,
                 save_path=None, run_for_images=None, wrap=16):
    """Single figure: four narrative rows, each one image judged by every model.

    Rows are max disagreement, the strong model winning, the domain model winning, and
    everything failing. Both "wins" rows are shown deliberately -- a figure containing
    only the cases your preferred model wins reads as cherry-picking.
    """
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    picks = story_examples(outs, strong=strong, domain=domain)
    strong_name, domain_name = picks.pop("_strong"), picks.pop("_domain")

    order = ["max_disagreement", "strong_wins", "domain_wins", "unanimous_failure"]
    titles = {
        "max_disagreement": "models disagree most",
        "strong_wins": "{} right, {} wrong".format(_short(strong_name), _short(domain_name)),
        "domain_wins": "{} right, {} wrong".format(_short(domain_name), _short(strong_name)),
        "unanimous_failure": "every model wrong",
    }
    rows = [m for m in order if picks[m]["index"] is not None]
    if not rows:
        print("no qualifying examples in any mode")
        return None

    if run_for_images is not None:
        raw_data = get_data(run_for_images, split, raw=True)
        proc_data = get_data(run_for_images, split)
    else:
        import torchvision.transforms as T
        raw_data = data_utils.get_data("{}_{}".format(outs[0].dataset, split),
                                       preprocess=T.Lambda(lambda x: x))
        proc_data = None

    n_col = len(outs) + 1
    fig = plt.figure(figsize=(2.05 * n_col + 1.6, 2.75 * len(rows)))
    gs = gridspec.GridSpec(len(rows), n_col, figure=fig,
                           width_ratios=[1.25] + [1] * len(outs),
                           hspace=0.32, wspace=0.12)

    for r, mode in enumerate(rows):
        idx = picks[mode]["index"]
        truth = int(outs[0].labels[idx])

        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(_collage_image(run_for_images, raw_data, proc_data, idx))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#333333"); s.set_linewidth(1.2)
        ax.set_ylabel("{}\n({} imgs)".format(titles[mode], picks[mode]["count"]),
                      fontsize=8.5, fontweight="bold", labelpad=8)
        ax.set_title("#{}  truth:\n{}".format(
            idx, "\n".join(textwrap.wrap(outs[0].classes[truth], wrap))), fontsize=7.5)

        for c, o in enumerate(outs):
            cell = fig.add_subplot(gs[r, c + 1])
            cell.set_xticks([]); cell.set_yticks([])
            ok = bool(o.correct[idx])
            cell.set_facecolor("#e8f5e9" if ok else "#ffebee")
            for s in cell.spines.values():
                s.set_edgecolor("#2ca02c" if ok else "#d62728"); s.set_linewidth(1.8)

            if r == 0:
                cell.set_title("{}\n{:.1f}%".format(_short(o.name), o.accuracy * 100),
                               fontsize=8.5, fontweight="bold")

            body = "\n".join(textwrap.wrap(o.classes[int(o.preds[idx])], wrap))
            cell.text(0.5, 0.80, "OK" if ok else "X", ha="center", va="top",
                      fontsize=11, fontweight="bold",
                      color="#2ca02c" if ok else "#d62728", transform=cell.transAxes)
            cell.text(0.5, 0.62, body, ha="center", va="top", fontsize=7.2,
                      transform=cell.transAxes)
            cell.text(0.5, 0.06, "p={:.2f}".format(o.confidence[idx]), ha="center",
                      va="bottom", fontsize=7, color="#555555", transform=cell.transAxes)
            if top_concepts:
                cs = o.top_concepts(idx, k=top_concepts)
                if cs:
                    cell.text(0.5, 0.20, "\n".join(textwrap.wrap(", ".join(cs), 22)),
                              ha="center", va="bottom", fontsize=5.8, style="italic",
                              color="#444444", transform=cell.transAxes)

    fig.suptitle("Qualitative comparison on {} -- one image per failure mode".format(split),
                 fontsize=12, y=0.995)
    if save_path:
        _save_fig(fig, save_path, dpi=200)
    plt.show()
    return fig


def _short(name):
    """Compact run name for column headers."""
    return (name.replace("_birds525", "").replace("birds525_", "")
                .replace("__lam", " lam").replace("_vitb_concepts", "+vitbC"))
