#!/usr/bin/env python3
"""OpenI entry: one seed, two rounds, checkpoint recovery upload before offline scoring."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p1.artifacts import sha256, verify_training, verify_frozen
from p1.config import load_config
from p1.openi import checked_context, package_results, publish, safe_extract_zip, validate_run_id, verify_file, verify_input_tree
from p1.phase1a import require_v100, summarize_cost, validate_phase1a_config


def sdk_version():
    for name in ("c2net", "c2net-beta"):
        try:
            return {"distribution": name, "version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"distribution": "unknown; module import succeeded", "version": None}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_id", required=True)
    p.add_argument("--mode", choices=["phase1a"], default="phase1a")
    p.add_argument("--data_source", default="", help="Absolute mounted archive; empty resolves exact archive_name under c2net.dataset_path")
    p.add_argument("--archive_name", default="p1_phase1a_input_v1.zip")
    p.add_argument("--input_bytes", type=int, required=True)
    p.add_argument("--input_sha256", required=True)
    p.add_argument("--work_root", default="/tmp/p1_openi_runs")
    p.add_argument("--config", default="configs/phase1a_v100_smoke.json")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    validate_run_id(args.run_id)
    os.chdir(ROOT)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = load_config(args.config)
    validate_phase1a_config(cfg)
    if cfg["rounds"] != 2:
        raise ValueError("This OpenI launch card is frozen to exactly two rounds")
    # c2net is provided by the selected OpenI image; never substitute a local token.
    from c2net.context import prepare, upload_output
    context = checked_context(prepare, upload_output)
    output = Path(context.output_path)
    prefix = f"p1_{args.run_id}"
    if output.exists() and list(output.glob(prefix + "_*")):
        raise FileExistsError("run_id already has staged artifacts")
    workspace = Path(args.work_root) / args.run_id
    if workspace.exists():
        raise FileExistsError("run_id already exists locally")
    run_root = workspace / "run"
    metadata = run_root / "metadata"
    metadata.mkdir(parents=True)
    status_file = workspace / (prefix + "_status.json")
    status = {"run_id": args.run_id, "mode": "phase1a", "seed": 42,
              "execution_status": "preflight", "analysis_status": "not_started",
              "scientific_evidence": False, "scientific_verdict": "INCONCLUSIVE",
              "recovery_upload": {"upload_call_returned": False, "remote_download_verified": False},
              "training_complete": False, "formal_training_started": False}
    started = time.perf_counter()

    def save_status():
        status["elapsed_seconds"] = time.perf_counter() - started
        text = json.dumps(status, indent=2)
        status_file.write_text(text, encoding="utf-8")
        (metadata / "status.json").write_text(text, encoding="utf-8")

    def stage(command):
        log_path = metadata / (command + ".log")
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/p1_cli.py"), command,
                                      "--config", str(runtime_config), "--out", str(results)],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log.write(line); log.flush(); print(line, end="", flush=True)
            if proc.wait():
                raise RuntimeError(f"Phase-1A {command} subprocess failed; inspect stage log")

    try:
        gpu = require_v100()
        if args.data_source:
            archive = Path(args.data_source)
            if not archive.is_absolute():
                raise ValueError("data_source must be an absolute mounted path")
        else:
            dataset_path = getattr(context, "dataset_path", None)
            if not dataset_path:
                raise RuntimeError("c2net dataset_path unavailable")
            matches = list(Path(dataset_path).rglob(args.archive_name))
            if len(matches) != 1:
                raise RuntimeError("Expected exactly one input archive under dataset mount")
            archive = matches[0]
        verify_file(archive, args.input_bytes, args.input_sha256)
        safe_extract_zip(archive, workspace / "input")
        inputs = workspace / "input/p1_phase1a_input"
        input_manifest = verify_input_tree(inputs)
        # Only verified official CIFAR bytes may be unpickled/extracted.
        from torchvision.datasets import CIFAR10
        from torchvision.datasets.utils import check_integrity, extract_archive
        data_root = inputs / "data_cache"
        official_archive = data_root / CIFAR10.filename
        if not check_integrity(official_archive, CIFAR10.tgz_md5):
            raise ValueError("Official CIFAR archive MD5 mismatch")
        extract_archive(official_archive, data_root, remove_finished=False)
        if not (ROOT / "vendor/FNBench").exists():
            (ROOT / "vendor").mkdir(exist_ok=True)
            subprocess.run(["git", "clone", str(inputs / "FNBench.bundle"), str(ROOT / "vendor/FNBench")], check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_fnbench.py")], check=True)
        results = run_root / "results"
        cfg.update(data_root=str(data_root), prepared_data_dir=str(inputs / "prepared"), output_dir=str(results))
        runtime_config = metadata / "runtime_config.json"
        runtime_config.write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, indent=2), encoding="utf-8")
        from p1.training import git_revision
        (metadata / "run.json").write_text(json.dumps({"run_id": args.run_id, "mode": "phase1a", "gpu": gpu,
            "code_git_sha": git_revision(), "input_sha256": args.input_sha256, "input_bytes": args.input_bytes,
            "config_sha256": sha256(runtime_config), "input_manifest": input_manifest,
            "packages": {n: importlib.metadata.version(n) for n in ("torch", "torchvision", "numpy")},
            "c2net_distribution": sdk_version()}, indent=2), encoding="utf-8")
        status["execution_status"] = "training"
        save_status()
        stage("train")
        verify_training(results)
        status.update(training_complete=True, execution_status="training_complete_analysis_pending")
        save_status()
        recovery = package_results(run_root, workspace / (prefix + "_recovery_outputs.tar.gz"))
        status["recovery_upload"] = publish([recovery, status_file], output, upload_output, mutable_status=status_file.name)
        status["execution_status"] = "analysis"
        status["analysis_status"] = "running"
        save_status()
        for command in ("evaluate", "select", "audit", "plot"):
            stage(command)
        verify_frozen(results)
        from scripts.replay_phase1a_validation import replay
        replay(results, run_root / "noise_replay")
        summarize_cost(results)
        status.update(execution_status="complete", analysis_status="complete")
        save_status()
        final = package_results(run_root, workspace / (prefix + "_outputs.tar.gz"))
        publish([final, status_file], output, upload_output, mutable_status=status_file.name)
        print("Phase-1A complete; upload calls returned. Remote download verification still required.")
        return 0
    except Exception as exc:
        failed_stage = status["execution_status"]
        status.update(execution_status="failed", analysis_status="failed" if status["training_complete"] else "not_started",
                      failed_stage=failed_stage, error_type=type(exc).__name__,
                      error_summary="Phase-1A failed; inspect sanitized stage logs. No scientific conclusion.")
        save_status()
        try:
            failed = package_results(run_root, workspace / (prefix + "_failed_outputs.tar.gz"))
            publish([failed, status_file], output, upload_output, mutable_status=status_file.name)
        except Exception as publish_error:
            print(f"Failure publication also failed: {type(publish_error).__name__}", file=sys.stderr)
        print(f"Phase-1A stopped at {failed_stage}: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
