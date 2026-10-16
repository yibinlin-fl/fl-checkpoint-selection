"""Pure, deployment-facing selectors. NO clean validation/test labels or noise rates accepted."""
import csv
from collections import defaultdict
from pathlib import Path
from .evaluation import write_csv

# Explicit allowlist keeps auditor/oracle fields outside this API.
ALLOWED_COLUMNS = {"round", "client_id", "scenario", "n", "noisy_acc", "noisy_ce"}


def read_selection_scores(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames) != ALLOWED_COLUMNS:
            raise ValueError(f"Selector input schema mismatch; only {sorted(ALLOWED_COLUMNS)} accepted")
        out = []
        for r in reader:
            if r["scenario"] not in ("uniform", "heterogeneous"):
                raise ValueError("Selector forbidden from reading clean/unknown scenario")
            out.append({"round": int(r["round"]), "client_id": int(r["client_id"]),
                "scenario": r["scenario"], "n": int(r["n"]),
                "noisy_acc": float(r["noisy_acc"]), "noisy_ce": float(r["noisy_ce"])})
    if not out:
        raise ValueError("No selection score rows")
    return out


def compute_selection(rows, ema_alpha=0.6):
    """Select across checkpoints by noisy metrics only; ties favor earlier round."""
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
                           "selection_score": float("nan")})
        for selector, max_direction in [("macro_noisy_acc", True), ("weighted_noisy_acc", True),
                                        ("weighted_noisy_ce", False), ("ema_noisy_acc", True)]:
            key = (lambda rnd: (-stat[rnd][selector], rnd)) if max_direction else \
                  (lambda rnd: (stat[rnd][selector], rnd))
            chosen = min(rounds, key=key)
            selections.append({"scenario": scenario, "selector": selector,
                 "selected_round": chosen, "selection_score": stat[chosen][selector]})
    return selections


def run_selection(out_dir):
    import json
    out = Path(out_dir)
    cfg = json.loads((out / "config_used.json").read_text(encoding="utf-8"))
    scores = read_selection_scores(out / "validation_scores.csv")
    result = compute_selection(scores, ema_alpha=cfg.get("ema_alpha", 0.6))
    write_csv(out / "selected_models.csv", result,
              ["scenario", "selector", "selected_round", "selection_score"])
    print("Selection complete. This module did NOT read private_audit files.")
    return result
