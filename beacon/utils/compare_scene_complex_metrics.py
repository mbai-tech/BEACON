"""Compare multiple scene-complex metrics CSVs and save table summaries."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METRICS_DIR = Path(__file__).resolve().parent.parent / "environment" / "data" / "metrics"
DEFAULT_OUTPUT = METRICS_DIR / "metrics_scene_complex_comparison.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine planner metrics CSVs from the metrics folder and save comparison tables."
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=str(METRICS_DIR),
        help="Folder containing metrics_*scene_complex.csv files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path for the combined comparison report.",
    )
    return parser.parse_args()


def discover_csvs(metrics_dir: Path) -> list[Path]:
    paths = sorted(metrics_dir.glob("metrics*_scene_complex.csv"))
    return [path for path in paths if path.name != "metrics_scene_complex_comparison.csv"]


def load_rows(csv_paths: list[Path]) -> list[dict]:
    seen: set[tuple[str, str, int, int]] = set()
    rows: list[dict] = []

    for path in csv_paths:
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = {
                    "planner": row["planner"],
                    "family": row["family"],
                    "scene_idx": int(row["scene_idx"]),
                    "seed": int(row["seed"]),
                    "success": row["success"] == "True",
                    "steps": int(row["steps"]),
                    "path_length": float(row["path_length"]),
                    "n_contacts": int(row["n_contacts"]),
                    "n_sensed": int(row["n_sensed"]),
                    "source_file": path.name,
                }
                key = (
                    normalized["planner"],
                    normalized["family"],
                    normalized["scene_idx"],
                    normalized["seed"],
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(normalized)

    return rows


def fmt_float(value: float, digits: int = 2) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_pct(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return lines


def summarize_overall(rows: list[dict]) -> list[str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["planner"]].append(row)

    table_rows: list[list[str]] = []
    for planner in sorted(grouped):
        vals = grouped[planner]
        successes = [row for row in vals if row["success"]]
        n = len(vals)
        success_rate = len(successes) / n if n else math.nan
        avg_steps = sum(row["steps"] for row in vals) / n if n else math.nan
        avg_path = sum(row["path_length"] for row in vals) / n if n else math.nan
        avg_contacts = sum(row["n_contacts"] for row in vals) / n if n else math.nan
        avg_sensed = sum(row["n_sensed"] for row in vals) / n if n else math.nan
        avg_path_success = (
            sum(row["path_length"] for row in successes) / len(successes)
            if successes else math.nan
        )
        avg_steps_success = (
            sum(row["steps"] for row in successes) / len(successes)
            if successes else math.nan
        )
        table_rows.append(
            [
                planner,
                str(n),
                fmt_pct(success_rate),
                fmt_float(avg_steps),
                fmt_float(avg_path, 3),
                fmt_float(avg_contacts, 3),
                fmt_float(avg_sensed, 3),
                fmt_float(avg_path_success, 3),
                fmt_float(avg_steps_success),
            ]
        )

    return ["OVERALL COMPARISON", *render_table(
        [
            "planner",
            "episodes",
            "success_rate",
            "avg_steps",
            "avg_path_m",
            "avg_contacts",
            "avg_sensed",
            "avg_path_success_m",
            "avg_steps_success",
        ],
        table_rows,
    )]


def summarize_by_family(rows: list[dict]) -> list[str]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["planner"], row["family"])].append(row)

    table_rows: list[list[str]] = []
    for planner, family in sorted(grouped):
        vals = grouped[(planner, family)]
        successes = [row for row in vals if row["success"]]
        n = len(vals)
        success_rate = len(successes) / n if n else math.nan
        avg_steps = sum(row["steps"] for row in vals) / n if n else math.nan
        avg_path = sum(row["path_length"] for row in vals) / n if n else math.nan
        avg_path_success = (
            sum(row["path_length"] for row in successes) / len(successes)
            if successes else math.nan
        )
        table_rows.append(
            [
                planner,
                family,
                str(n),
                fmt_pct(success_rate),
                fmt_float(avg_steps),
                fmt_float(avg_path, 3),
                fmt_float(avg_path_success, 3),
            ]
        )

    return ["", "BY FAMILY", *render_table(
        [
            "planner",
            "family",
            "episodes",
            "success_rate",
            "avg_steps",
            "avg_path_m",
            "avg_path_success_m",
        ],
        table_rows,
    )]


def summarize_relative(rows: list[dict], base_planner: str = "beacon") -> list[str]:
    planners = sorted({row["planner"] for row in rows})
    index = {(row["planner"], row["family"], row["scene_idx"]): row for row in rows}
    families = sorted({row["family"] for row in rows})
    scenes = sorted({row["scene_idx"] for row in rows})

    if base_planner not in planners:
        return ["", f"RELATIVE TO {base_planner.upper()}", f"{base_planner} not present in loaded CSVs."]

    table_rows: list[list[str]] = []
    for other in planners:
        if other == base_planner:
            continue

        common = 0
        both_success = 0
        path_ratios: list[float] = []
        step_ratios: list[float] = []
        for family in families:
            for scene_idx in scenes:
                base = index.get((base_planner, family, scene_idx))
                challenger = index.get((other, family, scene_idx))
                if not base or not challenger:
                    continue
                common += 1
                if (
                    base["success"]
                    and challenger["success"]
                    and challenger["path_length"] > 0
                    and challenger["steps"] > 0
                ):
                    both_success += 1
                    path_ratios.append(base["path_length"] / challenger["path_length"])
                    step_ratios.append(base["steps"] / challenger["steps"])

        mean_path_ratio = sum(path_ratios) / len(path_ratios) if path_ratios else math.nan
        mean_step_ratio = sum(step_ratios) / len(step_ratios) if step_ratios else math.nan
        table_rows.append(
            [
                other,
                str(common),
                str(both_success),
                fmt_float(mean_path_ratio, 3),
                fmt_float(mean_step_ratio, 3),
            ]
        )

    return ["", f"RELATIVE TO {base_planner.upper()}", *render_table(
        ["other_planner", "common_episodes", "both_success", "beacon_path_ratio", "beacon_step_ratio"],
        table_rows,
    )]


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_paths = discover_csvs(metrics_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No metrics CSVs found in {metrics_dir}")

    rows = load_rows(csv_paths)
    planners = sorted({row["planner"] for row in rows})
    lines = [
        f"Loaded {len(rows)} unique episodes from {len(csv_paths)} metrics CSV file(s).",
        f"Metrics directory: {metrics_dir}",
        f"Planners found: {', '.join(planners)}",
        "Note: planning time is not stored in these CSVs, so step count is used as a planning-effort proxy.",
        "",
    ]
    lines.extend(summarize_overall(rows))
    lines.extend(summarize_by_family(rows))
    lines.extend(summarize_relative(rows))

    report = "\n".join(lines) + "\n"
    print(report, end="")
    output_path.write_text(report)
    print(f"\nSaved comparison report to: {output_path}")


if __name__ == "__main__":
    main()
