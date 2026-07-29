"""Run a suite of training + evaluation jobs sequentially.

This is the entrypoint a Kubernetes Job calls: one GPU allocation, many runs,
one log. A failing run is recorded and the suite continues, so a bad config in
position 2 does not cost you positions 3 through 8.

    python run_suite.py --suite bioclip_sweep
    python run_suite.py --suite bioclip_sweep --dry_run
"""
import argparse
import datetime
import json
import os
import time
import traceback

import config as config_lib


def _banner(text):
    print("\n" + "=" * 78, flush=True)
    print(text, flush=True)
    print("=" * 78, flush=True)


def run_suite(suite, dry_run=False, skip_eval=False, stop_on_error=False,
              save_dir=None, device=None, skip_existing=False):
    runs = config_lib.expand_suite(suite)

    _banner("suite {!r}: {} run(s)".format(suite, len(runs)))
    for i, (experiment, overrides) in enumerate(runs, 1):
        cfg = config_lib.load_experiment(experiment, overrides)
        print("  {:>2}. {:<44} {}".format(i, cfg.run_name, overrides or ""), flush=True)

    if dry_run:
        print("\ndry run -- nothing executed", flush=True)
        return []

    # imported late so that --dry_run works without torch installed
    import evaluate_cbm
    import train_cbm

    # created up front, and rewritten after every run, so that a job killed partway
    # through (DeadlineExceeded, eviction, OOM) still leaves a usable record
    out_dir = os.path.join("suite_results", "{}_{}".format(
        suite, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.json")
    print("\nprogress will be written to {} after each run".format(summary_path), flush=True)

    results = []
    suite_started = time.time()

    for i, (experiment, overrides) in enumerate(runs, 1):
        cfg = config_lib.load_experiment(experiment, overrides)
        if save_dir:
            cfg.save_dir = save_dir
        if device:
            cfg.device = device

        _banner("[{}/{}] {}".format(i, len(runs), cfg.run_name))
        record = {"run_name": cfg.run_name, "experiment": experiment,
                  "overrides": overrides, "status": "running"}
        started = time.time()

        run_dir = os.path.join(cfg.save_dir, cfg.run_name)
        finished_marker = os.path.join(run_dir, "eval_metrics.json")
        if skip_existing and os.path.exists(finished_marker):
            with open(finished_marker) as f:
                metrics = json.load(f)
            record.update(status="skipped", run_dir=run_dir, metrics=metrics,
                          total_seconds=0.0)
            print("already complete -- skipping (val accuracy {:.2f}%)".format(
                metrics.get("val_accuracy", float("nan")) * 100), flush=True)
            results.append(record)
            _write_summary(summary_path, suite, results, time.time() - suite_started)
            continue

        try:
            run_dir = train_cbm.train_cbm_and_save(cfg)
            record["run_dir"] = run_dir
            record["train_seconds"] = round(time.time() - started, 1)
            print("\ntrained in {:.1f} min".format(record["train_seconds"] / 60), flush=True)

            if skip_eval:
                record["status"] = "trained"
            else:
                print("evaluating {} ...".format(run_dir), flush=True)
                metrics = evaluate_cbm.evaluate(run_dir, device=cfg.device)
                record["metrics"] = metrics
                record["status"] = "ok"
                print("  val accuracy {:.2f}%  concepts {}  non-zero {:.1%}".format(
                    metrics["val_accuracy"] * 100, metrics["n_concepts"],
                    metrics["sparsity"]["fraction_non_zero"]), flush=True)

        except Exception as exc:
            record["status"] = "failed"
            record["error"] = "{}: {}".format(type(exc).__name__, exc)
            record["traceback"] = traceback.format_exc()
            print("\nRUN FAILED: {}".format(record["error"]), flush=True)
            traceback.print_exc()
            if stop_on_error:
                record.setdefault("total_seconds", round(time.time() - started, 1))
                results.append(record)
                _write_summary(summary_path, suite, results, time.time() - suite_started)
                break

        record.setdefault("total_seconds", round(time.time() - started, 1))
        results.append(record)
        _write_summary(summary_path, suite, results, time.time() - suite_started)
        print("elapsed for suite so far: {:.1f} min".format(
            (time.time() - suite_started) / 60), flush=True)

    _print_table(suite, results, time.time() - suite_started, summary_path)
    return results


def _write_summary(summary_path, suite, results, elapsed):
    with open(summary_path, "w") as f:
        json.dump({"suite": suite, "elapsed_seconds": round(elapsed, 1),
                   "complete": False, "runs": results}, f, indent=2)


def _print_table(suite, results, elapsed, summary_path):
    with open(summary_path, "w") as f:
        json.dump({"suite": suite, "elapsed_seconds": round(elapsed, 1),
                   "complete": True, "runs": results}, f, indent=2)

    _banner("suite finished in {:.1f} min -- {}".format(elapsed / 60, summary_path))
    print("{:<44} {:<9} {:>9} {:>9} {:>8}".format(
        "run", "status", "accuracy", "concepts", "minutes"))
    print("-" * 83)
    for r in results:
        m = r.get("metrics") or {}
        acc = "{:.2f}%".format(m["val_accuracy"] * 100) if "val_accuracy" in m else "-"
        mins = "{:.1f}".format(r.get("total_seconds", 0) / 60)
        print("{:<44} {:<9} {:>9} {:>9} {:>8}".format(
            r["run_name"][:44], r["status"], acc, m.get("n_concepts", "-"), mins))

    failed = [r["run_name"] for r in results if r["status"] == "failed"]
    if failed:
        print("\n{} run(s) failed: {}".format(len(failed), ", ".join(failed)))


def main():
    ap = argparse.ArgumentParser(description="Run a suite of CBM experiments in sequence")
    ap.add_argument("--suite", type=str, required=True,
                    help="name or path of a suite in configs/suites/")
    ap.add_argument("--dry_run", action="store_true",
                    help="list the runs a suite expands to, then exit")
    ap.add_argument("--skip_eval", action="store_true", help="train only, no evaluation")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip runs whose output dir already has eval_metrics.json. Use this "
                         "when resuming a suite that was killed partway through.")
    ap.add_argument("--stop_on_error", action="store_true",
                    help="abort the suite on the first failure (default: continue)")
    ap.add_argument("--save_dir", type=str, default=None, help="override save_dir for all runs")
    ap.add_argument("--device", type=str, default=None, help="override device for all runs")
    args = ap.parse_args()

    results = run_suite(args.suite, dry_run=args.dry_run, skip_eval=args.skip_eval,
                        stop_on_error=args.stop_on_error, save_dir=args.save_dir,
                        device=args.device, skip_existing=args.skip_existing)
    if any(r["status"] == "failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
