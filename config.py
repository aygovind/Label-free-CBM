"""Config loading for LF-CBM experiments.

Layers, lowest precedence first:

    configs/base.yaml -> configs/experiments/<name>.yaml -> suite overrides -> CLI flags

`configs/base.yaml` also defines the set of legal keys: anything appearing in an
experiment or suite that is not declared there is rejected, so a typo like
`lamda: 0.002` fails loudly instead of silently doing nothing.
"""
import argparse
import itertools
import os

import yaml

CONFIG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
BASE_CONFIG = os.path.join(CONFIG_ROOT, "base.yaml")


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve(name, subdir):
    """Accept a bare name ('bioclip_birds525'), a filename, or an explicit path."""
    candidates = [name,
                  os.path.join(CONFIG_ROOT, subdir, name),
                  os.path.join(CONFIG_ROOT, subdir, name + ".yaml")]
    for c in candidates:
        if os.path.isfile(c):
            return c
    available = sorted(f[:-5] for f in os.listdir(os.path.join(CONFIG_ROOT, subdir))
                       if f.endswith(".yaml"))
    raise FileNotFoundError(
        "no {} config named {!r}; available: {}".format(subdir[:-1], name, ", ".join(available)))


def load_base():
    return _load_yaml(BASE_CONFIG)


def _check_keys(cfg, legal, source):
    unknown = sorted(set(cfg) - set(legal))
    if unknown:
        raise ValueError("unknown config key(s) {} in {} -- not declared in base.yaml".format(
            unknown, source))


def load_experiment(name, overrides=None):
    """Merge base.yaml + the named experiment + overrides into a Namespace."""
    cfg = load_base()
    path = _resolve(name, "experiments")

    exp = _load_yaml(path)
    _check_keys(exp, cfg, path)
    cfg.update(exp)

    if overrides:
        _check_keys(overrides, cfg, "overrides for {!r}".format(name))
        cfg.update(overrides)

    if not cfg.get("run_name"):
        cfg["run_name"] = default_run_name(name, overrides)
    return argparse.Namespace(**cfg)


def _slug(value):
    return str(value).replace("/", "").replace(" ", "-").replace(".", "p")


def default_run_name(experiment, overrides=None):
    """'bioclip_birds525' + {'lam': 0.002} -> 'bioclip_birds525__lam0p002'."""
    stem = os.path.splitext(os.path.basename(experiment))[0]
    parts = ["{}{}".format(k, _slug(v))
             for k, v in sorted((overrides or {}).items()) if k != "run_name"]
    return "__".join([stem] + parts)


def expand_suite(name):
    """Return [(experiment_name, overrides), ...] for every run a suite declares.

    A `grid` on an entry expands to the cartesian product of its values, so a
    parameter sweep needs one entry rather than one file per point.
    """
    suite = _load_yaml(_resolve(name, "suites"))
    legal = load_base()
    runs = []

    for entry in suite.get("runs") or []:
        if "experiment" not in entry:
            raise ValueError("every suite entry needs an 'experiment' key: {}".format(entry))
        experiment = entry["experiment"]
        fixed = {k: v for k, v in entry.items() if k not in ("experiment", "grid")}
        _check_keys(fixed, legal, "suite entry for {!r}".format(experiment))

        grid = entry.get("grid") or {}
        _check_keys(grid, legal, "grid for {!r}".format(experiment))
        if not grid:
            runs.append((experiment, fixed))
            continue

        keys = sorted(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            overrides = dict(fixed)
            overrides.update(dict(zip(keys, combo)))
            runs.append((experiment, overrides))

    if not runs:
        raise ValueError("suite {!r} declares no runs".format(name))
    return runs


def apply_to_args(args, argv):
    """Overlay args.config onto an argparse Namespace, keeping explicit CLI flags on top.

    argv is the raw argument list (sys.argv[1:]); any flag named there was set by
    the user and outranks the config file.
    """
    cfg = load_experiment(args.config)
    explicit = {a.split("=")[0].lstrip("-").replace("-", "_")
                for a in argv if a.startswith("--")}
    for key, value in vars(cfg).items():
        if key not in explicit:
            setattr(args, key, value)
    return args
