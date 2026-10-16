"""Post-selection oracle evaluator, physically separate from the selector module."""
import csv
from pathlib import Path
from .evaluation import write_csv


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def audit_regret(selections, oracle_rows):
    truth = {int(r["round"]): float(r["test_accuracy"]) for r in oracle_rows}
    if not truth:
        raise ValueError("No test checkpoints")
    best = max(truth.values())
    result = []
    for s in selections:
        rnd = int(s["selected_round"])
        if rnd not in truth:
            raise ValueError(f"Selected checkpoint {rnd} not in oracle list")
        result.append({"scenario": s["scenario"], "selector": s["selector"],
                       "selected_round": rnd, "selected_test_acc": truth[rnd],
                       "oracle_best_test_acc": best,
                       "selection_regret_pp": (best - truth[rnd]) * 100})
    return result


def run_audit(out_dir):
    out = Path(out_dir)
    selections = read_csv(out / "selected_models.csv")
    oracle = read_csv(out / "private_audit" / "oracle_test.csv")
    regrets = audit_regret(selections, oracle)
    # An inaccessible-in-deployment diagnostic oracle is computed *only here*.
    clean = read_csv(out / "private_audit" / "clean_validation.csv")
    by_round = {}
    for r in clean:
        rnd = int(r["round"])
        numerator, denominator = by_round.get(rnd, (0.0, 0))
        by_round[rnd] = (numerator + float(r["clean_acc"]) * int(r["n"]),
                         denominator + int(r["n"]))
    clean_best_round = min(by_round, key=lambda rd: (-by_round[rd][0] / by_round[rd][1], rd))
    regrets.extend(audit_regret([{
        "scenario": "diagnostic_only", "selector": "clean_val_oracle",
        "selected_round": clean_best_round,
    }], oracle))
    test_best_round = min((int(r["round"]) for r in oracle),
                          key=lambda rd: (-float(next(o["test_accuracy"] for o in oracle if int(o["round"]) == rd)), rd))
    regrets.extend(audit_regret([{
        "scenario": "diagnostic_only", "selector": "test_oracle_upper_bound",
        "selected_round": test_best_round,
    }], oracle))
    write_csv(out / "regret_summary.csv", regrets,
              ["scenario", "selector", "selected_round", "selected_test_acc",
               "oracle_best_test_acc", "selection_regret_pp"])
    print("Audited decision table saved. Test oracle used AFTER all choices are fixed.")
    return regrets
