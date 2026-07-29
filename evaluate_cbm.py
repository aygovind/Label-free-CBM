"""Command-line evaluation of a trained CBM run directory.

Mirrors what evaluate_cbm.ipynb does for accuracy, minus the plots, so that a
suite can train and evaluate in one unattended pass.

    python evaluate_cbm.py --load_dir saved_models/bioclip_birds525
"""
import argparse
import json
import os

import torch

import cbm
import data_utils
import utils


def evaluate(load_dir, device="cuda", batch_size=250, num_workers=2, save=True):
    """Evaluate the run in load_dir; returns a metrics dict and writes eval_metrics.json."""
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        run_args = json.load(f)

    dataset = run_args["dataset"]
    _, target_preprocess = data_utils.get_target_model(run_args["backbone"], device)
    model = cbm.load_cbm(load_dir, device)

    val_data = data_utils.get_data(dataset + "_val", preprocess=target_preprocess)
    accuracy = utils.get_accuracy_cbm(model, val_data, device, batch_size, num_workers)

    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location="cpu")
    nnz = (W_g.abs() > 1e-5).sum().item()
    total = W_g.numel()

    with open(os.path.join(load_dir, "concepts.txt"), "r") as f:
        n_concepts = len(f.read().split("\n"))

    metrics = {
        "run_dir": load_dir,
        "dataset": dataset,
        "backbone": run_args["backbone"],
        "clip_name": run_args["clip_name"],
        "lam": run_args["lam"],
        "val_accuracy": float(accuracy),
        "n_concepts": n_concepts,
        "n_classes": W_g.shape[0],
        "sparsity": {"non_zero": nnz, "total": total, "fraction_non_zero": nnz / total},
    }

    if save:
        with open(os.path.join(load_dir, "eval_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained CBM")
    ap.add_argument("--load_dir", type=str, required=True, help="run directory to evaluate")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=250)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    metrics = evaluate(args.load_dir, args.device, args.batch_size, args.num_workers)
    print(json.dumps(metrics, indent=2))
    print("\nAccuracy: {:.2f}%   concepts: {}   non-zero weights: {:.1f}%".format(
        metrics["val_accuracy"] * 100, metrics["n_concepts"],
        metrics["sparsity"]["fraction_non_zero"] * 100))


if __name__ == "__main__":
    main()
