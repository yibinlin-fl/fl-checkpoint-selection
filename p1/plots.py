"""Optional diagnostic chart; no data-driven tuning or checkpoint choice."""
import csv
import json
from collections import defaultdict
from pathlib import Path


def run_plot(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out_dir)
    with (out / "validation_scores.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    series = defaultdict(lambda: defaultdict(list))
    for r in rows:
        series[r["scenario"]][int(r["round"])].append((float(r["noisy_acc"]), int(r["n"])))
    cfg = json.loads((out / "config_used.json").read_text(encoding="utf-8"))
    label = "SYNTHETIC DEBUG ONLY" if cfg["dataset"] == "synthetic-debug" else "CIFAR-10"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for scenario in sorted(series):
        rounds = sorted(series[scenario])
        scores = [sum(a*n for a,n in series[scenario][rd]) /
                  sum(n for a,n in series[scenario][rd]) for rd in rounds]
        ax.plot(rounds, scores, marker="o", label=scenario)
    ax.set(xlabel="Global checkpoint round", ylabel="Validation accuracy (observed)",
           title=f"{label}: same checkpoints, validation label quality")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "validation_accuracy.png", dpi=150)
    plt.close(fig)
    print("Saved validation_accuracy.png")
