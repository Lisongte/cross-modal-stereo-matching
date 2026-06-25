#!/usr/bin/env python3
"""Run rectification and stereo matching over dataset_testing."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


METRIC_KEYS = [
    "valid_count",
    "depth_mse_m2",
    "depth_rmse_m",
    "disp_epe_px",
    "d1_all",
    "abs_rel",
]


def parse_metrics(path: Path) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    return metrics


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def weighted_mean(values: list[tuple[float, float]]) -> float:
    weight_sum = sum(weight for _, weight in values)
    if weight_sum <= 0:
        return 0.0
    return sum(value * weight for value, weight in values) / weight_sum


def write_summary(out_dir: Path, rows: list[dict[str, float | str]]) -> None:
    summary = out_dir / "summary_metrics.txt"
    total_valid = sum(float(row["valid_count"]) for row in rows)

    with summary.open("w", encoding="utf-8") as f:
        f.write(f"frames: {len(rows)}\n")
        f.write(f"total_valid_count: {total_valid:.0f}\n")
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in rows]
            f.write(f"mean_{key}: {mean(values):.6f}\n")
        for key in ["depth_mse_m2", "depth_rmse_m", "disp_epe_px", "d1_all", "abs_rel"]:
            values = [(float(row[key]), float(row["valid_count"])) for row in rows]
            f.write(f"valid_weighted_mean_{key}: {weighted_mean(values):.6f}\n")

    table = out_dir / "per_frame_metrics.tsv"
    fieldnames = ["frame", "mode", "algorithm", *METRIC_KEYS]
    with table.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_testing", type=Path)
    parser.add_argument("--out", default="output/dataset_testing_config", type=Path)
    parser.add_argument("--config", default="config/matching.yml", type=Path)
    parser.add_argument("--calib", default="config/rgb_left_nir_right.yml", type=Path)
    parser.add_argument("--rectify-bin", default="build/rectify_pair", type=Path)
    parser.add_argument("--match-bin", default="build/stereo_match", type=Path)
    parser.add_argument(
        "--mode",
        choices=["raw_baseline", "no_sgm", "sgm4", "sgm8"],
        help="Override matching mode from config/matching.yml.",
    )
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rgb_dir = args.dataset / "rgb_left"
    nir_dir = args.dataset / "nir_right"
    gt_dir = args.dataset / "gt_depth"
    names = sorted(
        p.name
        for p in rgb_dir.glob("*.png")
        if (nir_dir / p.name).exists() and (gt_dir / p.name).exists()
    )
    if args.limit > 0:
        names = names[: args.limit]
    if not names:
        raise RuntimeError(f"no matched frames found under {args.dataset}")

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for index, name in enumerate(names, start=1):
        frame = Path(name).stem
        rectified_dir = args.out / "rectified" / frame
        matching_dir = args.out / "matching" / frame
        metrics_path = matching_dir / "metrics.txt"

        mode_label = args.mode if args.mode else "config"
        print(f"[{index}/{len(names)}] mode={mode_label} frame {frame}", flush=True)
        if not args.skip_existing or not metrics_path.exists():
            run(
                [
                    str(args.rectify_bin),
                    "--calib",
                    str(args.calib),
                    "--left",
                    str(rgb_dir / name),
                    "--right",
                    str(nir_dir / name),
                    "--left-depth",
                    str(gt_dir / name),
                    "--out",
                    str(rectified_dir),
                ]
            )
            match_cmd = [
                str(args.match_bin),
                "--config",
                str(args.config),
                "--left",
                str(rectified_dir / "left_rgb_rectified.png"),
                "--right",
                str(rectified_dir / "right_nir_rectified.png"),
                "--rectification",
                str(rectified_dir / "rectification.yml"),
                "--gt-depth",
                str(rectified_dir / "left_depth_rectified.png"),
                "--out",
                str(matching_dir),
            ]
            if args.mode:
                match_cmd.extend(["--mode", args.mode])
            run(match_cmd)
        metrics = parse_metrics(metrics_path)
        metrics["frame"] = frame
        rows.append(metrics)

    write_summary(args.out, rows)
    print(f"wrote {args.out / 'summary_metrics.txt'}", flush=True)
    print(f"wrote {args.out / 'per_frame_metrics.tsv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
