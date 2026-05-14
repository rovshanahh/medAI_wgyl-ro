from pathlib import Path
import json
import csv
import statistics
import torch

RESULTS_DIR = Path("evaluation/results")
MODELS_DIR = Path("reference_data/models")
OUT_PATH = RESULTS_DIR / "final_metrics_summary.txt"

SPECIALIST_ROUTES = {
    "brain": "Brain MRI",
    "abdomen": "Abdomen CT",
    "bone": "Bone X-ray",
    "breast": "Breast Mammography",
    "retina": "Retina Fundus",
    "skin": "Skin Dermoscopy",
}

def newest(pattern):
    files = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def load_json(path):
    return json.loads(path.read_text()) if path and path.exists() else None

def load_csv(path):
    if not path or not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))

def fmt(x):
    if x is None:
        return "—"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

def pct(x):
    if x is None:
        return "—"
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)

def checkpoint_metrics():
    rows = []

    for path in sorted(MODELS_DIR.rglob("*.pth")):
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception:
            continue

        if not isinstance(ckpt, dict):
            continue

        labels = ckpt.get("class_names")
        if not labels:
            continue

        rows.append({
            "checkpoint": str(path),
            "labels": labels,
            "seed": ckpt.get("seed"),
            "best_val_acc": ckpt.get("best_val_acc"),
            "best_val_macro_f1": ckpt.get("best_val_macro_f1"),
            "test_acc": ckpt.get("test_acc"),
            "test_macro_f1": ckpt.get("test_macro_f1"),
            "final_val_acc": ckpt.get("final_val_acc"),
            "final_val_macro_f1": ckpt.get("final_val_macro_f1"),
            "confusion": ckpt.get("confusion"),
        })

    return rows

def group_specialists(rows):
    groups = {
        "Brain MRI": [],
        "Abdomen CT": [],
        "Bone X-ray": [],
        "Breast Mammography": [],
        "Retina Fundus": [],
        "Skin Dermoscopy": [],
        "Chest X-ray": [],
        "Other": [],
    }

    for row in rows:
        p = row["checkpoint"].lower()

        if "/brain/" in p or "brain" in p:
            groups["Brain MRI"].append(row)
        elif "/abdomen/" in p or "abdomen" in p:
            groups["Abdomen CT"].append(row)
        elif "/bone/" in p or "bone" in p:
            groups["Bone X-ray"].append(row)
        elif "/breast/" in p or "breast" in p:
            groups["Breast Mammography"].append(row)
        elif "/retina/" in p or "retina" in p:
            groups["Retina Fundus"].append(row)
        elif "/skin/" in p or "skin" in p:
            groups["Skin Dermoscopy"].append(row)
        elif "chest" in p:
            groups["Chest X-ray"].append(row)
        else:
            groups["Other"].append(row)

    return {k: v for k, v in groups.items() if v}

def avg_metric(rows, key):
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    vals = [float(v) for v in vals]
    return statistics.mean(vals) if vals else None

def policy_distribution(active_rows, ood_rows):
    counts = {}

    for row in active_rows + ood_rows:
        action = row.get("policy") or row.get("policy_action") or row.get("action")
        if action:
            counts[action] = counts.get(action, 0) + 1

    return counts

def main():
    active_csv = newest("active_route_evaluation_*.csv")
    active_json = newest("active_route_evaluation_*.json")
    ood_csv = newest("ood_evaluation_*.csv")
    ood_json = newest("ood_evaluation_*.json")

    active_rows = load_csv(active_csv)
    ood_rows = load_csv(ood_csv)
    ood_data = load_json(ood_json)

    ckpt_rows = checkpoint_metrics()
    grouped = group_specialists(ckpt_rows)

    lines = []

    lines.append("FINAL METRICS SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Active route CSV: {active_csv}")
    lines.append(f"OOD CSV: {ood_csv}")
    lines.append("")

    lines.append("1) ACTIVE ROUTE / DEMO PIPELINE RESULTS")
    lines.append("-" * 80)
    for row in active_rows:
        lines.append(str(row))
    lines.append("")

    lines.append("2) OOD EVALUATION RESULTS")
    lines.append("-" * 80)
    if ood_data:
        for key in ["valid_acceptance_rate", "ood_rejection_rate"]:
            if key in ood_data:
                lines.append(f"{key}: {ood_data[key]}")
    for row in ood_rows:
        lines.append(str(row))
    lines.append("")

    lines.append("3) SPECIALIST MODEL CHECKPOINT METRICS")
    lines.append("-" * 80)
    for group, rows in grouped.items():
        lines.append(f"\n{group}")
        lines.append("-" * 40)
        lines.append(f"checkpoints: {len(rows)}")
        lines.append(f"avg best_val_acc: {fmt(avg_metric(rows, 'best_val_acc'))}")
        lines.append(f"avg best_val_macro_f1: {fmt(avg_metric(rows, 'best_val_macro_f1'))}")
        lines.append(f"avg test_acc: {fmt(avg_metric(rows, 'test_acc'))}")
        lines.append(f"avg test_macro_f1: {fmt(avg_metric(rows, 'test_macro_f1'))}")

        for r in rows:
            lines.append(
                f"- {r['checkpoint']} | seed={r.get('seed')} | "
                f"best_val_acc={fmt(r.get('best_val_acc'))} | "
                f"best_val_macro_f1={fmt(r.get('best_val_macro_f1'))} | "
                f"test_acc={fmt(r.get('test_acc'))} | "
                f"test_macro_f1={fmt(r.get('test_macro_f1'))}"
            )
            if r.get("confusion"):
                lines.append(f"  confusion={r['confusion']}")
    lines.append("")

    lines.append("4) POLICY ACTION DISTRIBUTION")
    lines.append("-" * 80)
    counts = policy_distribution(active_rows, ood_rows)
    if counts:
        for k, v in sorted(counts.items()):
            lines.append(f"{k}: {v}")
    else:
        lines.append("Policy distribution could not be computed from CSV fields; use JSON if needed.")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nSaved:", OUT_PATH)

if __name__ == "__main__":
    main()
