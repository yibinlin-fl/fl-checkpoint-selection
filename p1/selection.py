"""Pure, deployment-facing selectors. NO clean validation/test labels or noise rates accepted."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from .artifacts import write_csv, sha256

# Explicit allowlist keeps auditor/oracle fields outside this API.
ALLOWED_COLUMNS = {"round", "client_id", "scenario", "n", "noisy_acc", "noisy_ce"}


def read_selection_scores(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or len(reader.fieldnames) != len(ALLOWED_COLUMNS) or set(reader.fieldnames) != ALLOWED_COLUMNS:
            raise ValueError(f"Selector input schema mismatch; only {sorted(ALLOWED_COLUMNS)} accepted")
        out = []
        for r in reader:
            if set(r) != ALLOWED_COLUMNS or any(v is None for v in r.values()):
                raise ValueError("Selector input row schema mismatch")
            if r["scenario"] not in ("uniform", "heterogeneous"):
                raise ValueError("Selector forbidden from reading clean/unknown scenario")
            out.append({"round": int(r["round"]), "client_id": int(r["client_id"]),
                "scenario": r["scenario"], "n": int(r["n"]),
                "noisy_acc": float(r["noisy_acc"]), "noisy_ce": float(r["noisy_ce"])})
    if not out:
        raise ValueError("No selection score rows")
    validate_scores(out)
    return out


def validate_scores(rows):
    if not rows:
        raise ValueError("No selection score rows")
    seen, grids = set(), defaultdict(dict)
    for r in rows:
        if set(r) != ALLOWED_COLUMNS:
            raise ValueError("Selector input schema mismatch")
        if r["scenario"] not in ("uniform", "heterogeneous"):
            raise ValueError("Selector forbidden scenario")
        if r["round"] < 1 or r["client_id"] < 0 or r["n"] < 1:
            raise ValueError("Invalid round/client/sample count")
        if not math.isfinite(r["noisy_acc"]) or not 0 <= r["noisy_acc"] <= 1:
            raise ValueError("Invalid noisy accuracy")
        if not math.isfinite(r["noisy_ce"]) or r["noisy_ce"] < 0:
            raise ValueError("Invalid noisy CE")
        key = r["scenario"], r["round"], r["client_id"]
        if key in seen:
            raise ValueError("Duplicate client score")
        seen.add(key)
        grids[(r["scenario"], r["round"])][r["client_id"]] = r["n"]
    reference = next(iter(grids.values()))
    if any(g != reference for g in grids.values()):
        raise ValueError("Incomplete client grid or changed sample counts")
    round_sets = defaultdict(set)
    for scenario, rnd in grids:
        round_sets[scenario].add(rnd)
    if any(v != next(iter(round_sets.values())) for v in round_sets.values()):
        raise ValueError("Scenarios must share checkpoints")


def compute_selection(rows, ema_alpha=0.6):
    """Select across checkpoints by noisy metrics only; ties favor earlier round."""
    validate_scores(rows)
    if not 0 < ema_alpha <= 1:
        raise ValueError("EMA alpha must be in (0, 1]")
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[r["scenario"]][r["round"]].append(r)
    selections = []
    for scenario in sorted(grouped):
        per_round = grouped[scenario]
        rounds = sorted(per_round)
        if not rounds:
            continue
        stat = {}
        for rnd in rounds:
            clients = per_round[rnd]
            n_total = sum(k["n"] for k in clients)
            stat[rnd] = {
                "macro_noisy_acc": sum(k["noisy_acc"] for k in clients) / len(clients),
                "weighted_noisy_acc": sum(k["noisy_acc"] * k["n"] for k in clients) / n_total,
                "weighted_noisy_ce": sum(k["noisy_ce"] * k["n"] for k in clients) / n_total,
            }
        prev = None
        for rnd in rounds:
            current = stat[rnd]["weighted_noisy_acc"]
            prev = current if prev is None else ema_alpha * current + (1 - ema_alpha) * prev
            stat[rnd]["ema_noisy_acc"] = prev

        selections.append({"scenario": scenario, "selector": "last", "selected_round": rounds[-1],
                           "selection_score": ""})
        for selector, max_direction in [("macro_noisy_acc", True), ("weighted_noisy_acc", True),
                                        ("weighted_noisy_ce", False), ("ema_noisy_acc", True)]:
            key = (lambda rnd: (-stat[rnd][selector], rnd)) if max_direction else \
                  (lambda rnd: (stat[rnd][selector], rnd))
            chosen = min(rounds, key=key)
            selections.append({"scenario": scenario, "selector": selector,
                 "selected_round": chosen, "selection_score": stat[chosen][selector]})
    return selections


def run_selection(out_dir, ema_alpha=0.6):
    out = Path(out_dir)
    if (out / "selection_frozen.json").exists():
        raise FileExistsError("Selection already frozen")
    scores = read_selection_scores(out / "validation_scores.csv")
    result = compute_selection(scores, ema_alpha=ema_alpha)
    write_csv(out / "selected_models.csv", result,
              ["scenario", "selector", "selected_round", "selection_score"])
    (out / "selection_frozen.json").write_text(json.dumps({
        "validation_scores.csv": sha256(out / "validation_scores.csv"),
        "selected_models.csv": sha256(out / "selected_models.csv"),
        "ema_alpha": ema_alpha,
    }, indent=2), encoding="utf-8")
    print("Selection complete. This module did NOT read private_audit files.")
    return result
