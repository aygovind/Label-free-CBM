"""Local smoke tests -- run these before pushing or submitting a cluster job.

Three tiers, cheapest first. Each tier states its dependencies and is SKIPPED
(loudly, never silently) when they are missing, so a tier that did not run is
never mistaken for a tier that passed.

    config    no dependencies beyond pyyaml. Config merge, precedence, suites.
    backbone  needs torch + torchvision + open_clip. Feature dims and cache keys.
    e2e       needs the above + ~350 MB of downloads. Full tiny training run.

    python smoke_test.py                  # every tier
    python smoke_test.py --tier config    # just the cheap one
    python smoke_test.py --keep           # leave smoke_out/ behind for inspection
"""
import argparse
import importlib
import os
import shutil
import sys
import traceback

REPO = os.path.dirname(os.path.abspath(__file__))
SMOKE_OUT = os.path.join(REPO, "smoke_out")

_results = []


def check(label, got, want):
    passed = got == want
    _results.append(("PASS" if passed else "FAIL", label))
    print("  {}  {}{}".format("PASS" if passed else "FAIL", label,
                              "" if passed else "  (got {!r}, want {!r})".format(got, want)),
          flush=True)
    return passed


def ok(label):
    _results.append(("PASS", label))
    print("  PASS  {}".format(label), flush=True)


def skip(label, reason):
    _results.append(("SKIP", label))
    print("  SKIP  {}  ({})".format(label, reason), flush=True)


def step(message):
    """Announce slow work before doing it, so a long pause is never mistaken for a hang."""
    print("  ...   {}".format(message), flush=True)


def missing_modules(*names):
    out = []
    for n in names:
        try:
            importlib.import_module(n)
        except ImportError:
            out.append(n)
    return out


# ---------------------------------------------------------------- tier: config

def tier_config():
    print("\n[config] merge, precedence, suite expansion")
    import argparse as ap
    import config

    c = config.load_experiment("bioclip_birds525")
    check("experiment value wins over base", c.backbone, "bioclip")
    check("feature_layer from experiment", c.feature_layer, "visual.ln_post")
    check("unset key falls through to base", c.n_iters, 1000)
    check("derived run_name", c.run_name, "bioclip_birds525")

    c = config.load_experiment("bioclip_birds525", {"lam": 0.005})
    check("suite override wins over experiment", c.lam, 0.005)
    check("run_name encodes the override", c.run_name, "bioclip_birds525__lam0p005")

    args = ap.Namespace(config="bioclip_birds525", lam=0.9, batch_size=512)
    args = config.apply_to_args(args, ["--config", "bioclip_birds525", "--lam", "0.9"])
    check("explicit CLI flag beats config", args.lam, 0.9)
    check("unspecified flag taken from config", args.batch_size, 256)

    try:
        config.load_experiment("bioclip_birds525", {"lamda": 0.1})
        check("typo rejected", "accepted", "ValueError")
    except ValueError:
        ok("typo in a config key raises instead of being ignored")

    runs = config.expand_suite("bioclip_sweep")
    check("suite expands grid into separate runs", len(runs), 6)
    names = [config.load_experiment(e, o).run_name for e, o in runs]
    check("grid run names are unique", len(set(names)), 6)

    exp_dir = os.path.join(config.CONFIG_ROOT, "experiments")
    for name in sorted(f[:-5] for f in os.listdir(exp_dir) if f.endswith(".yaml")):
        config.load_experiment(name)
        ok("config loads: {}".format(name))


# -------------------------------------------------------------- tier: backbone

def tier_backbone():
    print("\n[backbone] feature dimensions and activation cache keys", flush=True)
    step("importing torch / torchvision (slow on first import)")
    absent = missing_modules("torch", "torchvision")
    if absent:
        skip("backbone tier", "missing {}".format(", ".join(absent)))
        return

    import torch
    step("importing data_utils (pulls in clip, pytorchcv)")
    import data_utils
    import utils

    # cache keys must differ per (backbone, clip_name, layer) or one silently
    # overwrites the other -- this is the collision that made bioclip alias CLIP
    for backbone, clip_name, layer in [("bioclip", "bioclip", "visual.ln_post"),
                                       ("resnet50", "RN50", "layer4")]:
        t, c, _ = utils.get_save_names(clip_name, backbone, layer, "d", "cs.txt", "avg", "A")
        check("cache keys distinct for {}/{}".format(backbone, clip_name), t == c, False)

    ckpt = data_utils.BIOCLIP_CKPT
    if not os.path.exists(ckpt):
        skip("BioCLIP backbone feature dim", "checkpoint not at {}".format(ckpt))
    elif missing_modules("open_clip"):
        skip("BioCLIP backbone feature dim", "missing open_clip")
    else:
        size_mb = os.path.getsize(ckpt) / 1e6
        step("loading BioCLIP from {} ({:.0f} MB) -- reading this off the PVC can take "
             "a minute or two, it is not hung".format(ckpt, size_mb))
        model, preprocess = data_utils.get_target_model("bioclip", "cpu")
        step("checkpoint loaded; running one forward pass on CPU")
        with torch.no_grad():
            feats = model(torch.randn(2, 3, 224, 224))
        check("bioclip returns ln_post features (768-d)", tuple(feats.shape), (2, 768))
        check("preprocess returned", preprocess is not None, True)


# ------------------------------------------------------------------- tier: e2e

def _subset_cifar(data_utils, per_class_train=20, per_class_val=10):
    """Shrink CIFAR10 to a stratified handful so the run finishes on a laptop CPU."""
    import numpy as np
    original = data_utils.get_data

    def limited(name, preprocess=None):
        ds = original(name, preprocess)
        if not (hasattr(ds, "data") and hasattr(ds, "targets")):
            return ds
        per_class = per_class_train if name.endswith("_train") else per_class_val
        targets = np.array(ds.targets)
        idx = np.sort(np.concatenate(
            [np.where(targets == c)[0][:per_class] for c in range(int(targets.max()) + 1)]))
        ds.data = ds.data[idx]
        ds.targets = [int(targets[i]) for i in idx]
        return ds

    data_utils.get_data = limited
    return original


def tier_e2e():
    print("\n[e2e] full training run + evaluation on a CIFAR10 subset")
    absent = missing_modules("torch", "torchvision", "ftfy", "regex", "tqdm")
    if absent:
        skip("e2e tier", "missing {}".format(", ".join(absent)))
        return

    import config
    import data_utils
    import utils
    from torch.utils.data import DataLoader as _DataLoader

    # utils.py hardcodes num_workers=8, which is wrong for a laptop (and for the
    # pod's 6 Gi /dev/shm). Force 0 here so the smoke test is not testing your
    # machine's shared-memory limits. Real runs still use the hardcoded value.
    original_dataloader = utils.DataLoader
    utils.DataLoader = lambda ds, bs, num_workers=0, pin_memory=False: _DataLoader(ds, bs)
    print("  note: forcing num_workers=0 for this run (utils.py hardcodes 8)")

    original_get_data = _subset_cifar(data_utils)
    try:
        import evaluate_cbm
        import train_cbm

        cfg = config.load_experiment("smoke_cifar10")
        cfg.run_name = "smoke"
        print("  training (downloads CIFAR10 + CLIP RN50 + ResNet50 on first run) ...")
        run_dir = train_cbm.train_cbm_and_save(cfg)
        ok("training completed -> {}".format(run_dir))

        for fname in ("W_c.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt",
                      "concepts.txt", "args.txt", "metrics.txt"):
            check("wrote {}".format(fname), os.path.exists(os.path.join(run_dir, fname)), True)

        with open(os.path.join(run_dir, "concepts.txt")) as f:
            n_concepts = len(f.read().split("\n"))
        check("concepts survived pruning", n_concepts > 0, True)

        print("  evaluating ...")
        metrics = evaluate_cbm.evaluate(run_dir, device="cpu", batch_size=32, num_workers=0)
        check("eval_metrics.json written",
              os.path.exists(os.path.join(run_dir, "eval_metrics.json")), True)
        check("accuracy is a real number in [0,1]",
              0.0 <= metrics["val_accuracy"] <= 1.0, True)
        ok("accuracy {:.1f}% on {} concepts  (a tiny subset -- the number is "
           "meaningless, only that it computed)".format(
               metrics["val_accuracy"] * 100, metrics["n_concepts"]))
    finally:
        data_utils.get_data = original_get_data
        utils.DataLoader = original_dataloader


TIERS = {"config": tier_config, "backbone": tier_backbone, "e2e": tier_e2e}


def main():
    ap = argparse.ArgumentParser(description="Smoke tests for the LF-CBM pipeline")
    ap.add_argument("--tier", choices=list(TIERS) + ["all"], default="all")
    ap.add_argument("--keep", action="store_true", help="keep smoke_out/ afterwards")
    args = ap.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, REPO)
    selected = list(TIERS) if args.tier == "all" else [args.tier]

    failed_hard = False
    for name in selected:
        try:
            TIERS[name]()
        except Exception:
            failed_hard = True
            _results.append(("FAIL", "{} tier raised".format(name)))
            print("  FAIL  {} tier raised an exception".format(name))
            traceback.print_exc()

    if not args.keep and os.path.isdir(SMOKE_OUT):
        shutil.rmtree(SMOKE_OUT)

    counts = {k: sum(1 for s, _ in _results if s == k) for k in ("PASS", "FAIL", "SKIP")}
    print("\n" + "-" * 66)
    print("{} passed, {} failed, {} skipped".format(
        counts["PASS"], counts["FAIL"], counts["SKIP"]))
    if counts["SKIP"]:
        print("\nSKIPPED checks did not run -- they are not passes:")
        for status, label in _results:
            if status == "SKIP":
                print("  - {}".format(label))
    raise SystemExit(1 if (counts["FAIL"] or failed_hard) else 0)


if __name__ == "__main__":
    main()
