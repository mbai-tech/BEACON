"""Print summary tables from a scene-complex metrics CSV."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


_METRICS_DIR = Path(__file__).resolve().parent.parent / "environment" / "data" / "metrics"
DEFAULT_INPUT = _METRICS_DIR / "metrics_scene_complex.csv"
DEFAULT_OUTPUT = _METRICS_DIR / "metrics_scene_complex_summary.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize success rate, path efficiency, and step-effort proxies from a scene-complex metrics CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to a metrics CSV produced by run_scene_complex_metrics.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to a text file where the printed summary will also be saved.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["scene_idx"] = int(row["scene_idx"])
            row["seed"] = int(row["seed"])
            row["success"] = row["success"] == "True"
            row["steps"] = int(row["steps"])
            row["path_length"] = float(row["path_length"])
            row["n_contacts"] = int(row["n_contacts"])
            row["n_sensed"] = int(row["n_sensed"])
            rows.append(row)
    return rows


def summarize_overall(rows: list[dict]) -> list[str]:
    by_planner: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_planner[row["planner"]].append(row)

    lines = ["OVERALL BY PLANNER"]
    for planner in sorted(by_planner):
        vals = by_planner[planner]
        successes = [v for v in vals if v["success"]]
        n = len(vals)
        success_rate = len(successes) / n if n else 0.0
        avg_steps = sum(v["steps"] for v in vals) / n if n else 0.0
        med_steps = statistics.median(v["steps"] for v in vals) if n else 0.0
        avg_len = sum(v["path_length"] for v in vals) / n if n else 0.0
        med_len = statistics.median(v["path_length"] for v in vals) if n else 0.0
        avg_contacts = sum(v["n_contacts"] for v in vals) / n if n else 0.0
        avg_sensed = sum(v["n_sensed"] for v in vals) / n if n else 0.0
        avg_len_success = (
            sum(v["path_length"] for v in successes) / len(successes)
            if successes else float("nan")
        )
        avg_steps_success = (
            sum(v["steps"] for v in successes) / len(successes)
            if successes else float("nan")
        )
        lines.append(
            f"{planner:>10} | episodes={n:>4} | success_rate={success_rate:>6.2%} "
            f"| avg_steps={avg_steps:>7.2f} | median_steps={med_steps:>6.1f} "
            f"| avg_path={avg_len:>6.3f}m | median_path={med_len:>6.3f}m "
            f"| avg_contacts={avg_contacts:>6.3f} | avg_sensed={avg_sensed:>6.3f} "
            f"| avg_path_success={avg_len_success:>6.3f}m "
            f"| avg_steps_success={avg_steps_success:>7.2f}"
        )
    return lines


def summarize_by_family(rows: list[dict]) -> list[str]:
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["planner"], row["family"])].append(row)

    lines = ["", "BY PLANNER AND FAMILY"]
    for planner, family in sorted(by_key):
        vals = by_key[(planner, family)]
        successes = [v for v in vals if v["success"]]
        n = len(vals)
        success_rate = len(successes) / n if n else 0.0
        avg_steps = sum(v["steps"] for v in vals) / n if n else 0.0
        avg_len = sum(v["path_length"] for v in vals) / n if n else 0.0
        avg_len_success = (
            sum(v["path_length"] for v in successes) / len(successes)
            if successes else float("nan")
        )
        lines.append(
            f"{planner:>10} | {family:<18} | n={n:>3} | success_rate={success_rate:>6.2%} "
            f"| avg_steps={avg_steps:>7.2f} | avg_path={avg_len:>6.3f}m "
            f"| avg_path_success={avg_len_success:>6.3f}m"
        )
    return lines


def summarize_relative(rows: list[dict], base_planner: str = "beacon") -> list[str]:
    by_planner = sorted({row["planner"] for row in rows})
    index = {(r["planner"], r["family"], r["scene_idx"]): r for r in rows}
    families = sorted({r["family"] for r in rows})
    scenes = sorted({r["scene_idx"] for r in rows})

    lines = ["", f"{base_planner.upper()} VS OTHERS (MATCHED SUCCESS EPISODES)"]
    for other in by_planner:
        if other == base_planner:
            continue
        path_ratios: list[float] = []
        step_ratios: list[float] = []
        common = 0
        both_success = 0
        for family in families:
            for scene_idx in scenes:
                a = index.get((base_planner, family, scene_idx))
                b = index.get((other, family, scene_idx))
                if not a or not b:
                    continue
                common += 1
                if a["success"] and b["success"] and b["path_length"] > 0 and b["steps"] > 0:
                    both_success += 1
                    path_ratios.append(a["path_length"] / b["path_length"])
                    step_ratios.append(a["steps"] / b["steps"])

        mean_path_ratio = sum(path_ratios) / len(path_ratios) if path_ratios else float("nan")
        mean_step_ratio = sum(step_ratios) / len(step_ratios) if step_ratios else float("nan")
        lines.append(
            f"{base_planner:>10} vs {other:<10} | common={common:>4} | both_success={both_success:>4} "
            f"| mean_path_ratio={mean_path_ratio:>6.3f} | mean_step_ratio={mean_step_ratio:>6.3f}"
        )
    return lines


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    report_lines = [
        f"Loaded {len(rows)} episodes from {input_path}",
        "Note: this CSV does not include wall-clock runtime columns, so 'planning time' is approximated by step counts only.",
    ]
    report_lines.extend(summarize_overall(rows))
    report_lines.extend(summarize_by_family(rows))
    report_lines.extend(summarize_relative(rows))

    report_text = "\n".join(report_lines) + "\n"
    print(report_text, end="")
    output_path.write_text(report_text)
    print(f"\nSaved summary report to: {output_path}")


if __name__ == "__main__":
    main()
