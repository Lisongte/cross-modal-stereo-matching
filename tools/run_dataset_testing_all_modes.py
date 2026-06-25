#!/usr/bin/env python3
"""Run dataset_testing for all supported matching modes and summarize results."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


MODES = ["raw_baseline", "no_sgm", "sgm4", "sgm8"]
SUMMARY_KEYS = [
    "frames",
    "total_valid_count",
    "mean_depth_rmse_m",
    "mean_disp_epe_px",
    "mean_d1_all",
    "mean_abs_rel",
    "valid_weighted_mean_depth_rmse_m",
    "valid_weighted_mean_disp_epe_px",
    "valid_weighted_mean_d1_all",
    "valid_weighted_mean_abs_rel",
]


def parse_summary(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key] = float(value.strip())
    return values


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def write_all_modes_summary(out_dir: Path, rows: list[dict[str, float | str]]) -> None:
    tsv_path = out_dir / "all_modes_summary.tsv"
    fields = ["mode", *SUMMARY_KEYS]
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    txt_path = out_dir / "all_modes_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("mode\tframes\tmean_RMSE\tmean_EPE\tmean_D1\tmean_AbsRel\tweighted_RMSE\tweighted_EPE\tweighted_D1\tweighted_AbsRel\n")
        for row in rows:
            f.write(
                f"{row['mode']}\t"
                f"{int(float(row['frames']))}\t"
                f"{float(row['mean_depth_rmse_m']):.6f}\t"
                f"{float(row['mean_disp_epe_px']):.6f}\t"
                f"{float(row['mean_d1_all']):.6f}\t"
                f"{float(row['mean_abs_rel']):.6f}\t"
                f"{float(row['valid_weighted_mean_depth_rmse_m']):.6f}\t"
                f"{float(row['valid_weighted_mean_disp_epe_px']):.6f}\t"
                f"{float(row['valid_weighted_mean_d1_all']):.6f}\t"
                f"{float(row['valid_weighted_mean_abs_rel']):.6f}\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_testing", type=Path)
    parser.add_argument("--out", default="output/dataset_testing_all_modes", type=Path)
    parser.add_argument("--config", default="config/matching.yml", type=Path)
    parser.add_argument("--calib", default="config/rgb_left_nir_right.yml", type=Path)
    parser.add_argument("--rectify-bin", default="build/rectify_pair", type=Path)
    parser.add_argument("--match-bin", default="build/stereo_match", type=Path)
    parser.add_argument("--runner", default="tools/run_dataset_testing.py", type=Path)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for index, mode in enumerate(MODES, start=1):
        mode_out = args.out / mode
        print(f"\n=== [{index}/{len(MODES)}] Running mode={mode} ===", flush=True)
        cmd = [
            sys.executable,
            str(args.runner),
            "--dataset",
            str(args.dataset),
            "--out",
            str(mode_out),
            "--config",
            str(args.config),
            "--calib",
            str(args.calib),
            "--rectify-bin",
            str(args.rectify_bin),
            "--match-bin",
            str(args.match_bin),
            "--mode",
            mode,
        ]
        if args.limit > 0:
            cmd.extend(["--limit", str(args.limit)])
        if args.skip_existing:
            cmd.append("--skip-existing")
        run(cmd)

        summary = parse_summary(mode_out / "summary_metrics.txt")
        row: dict[str, float | str] = {"mode": mode}
        row.update(summary)
        rows.append(row)

    write_all_modes_summary(args.out, rows)
    print(f"\nwrote {args.out / 'all_modes_summary.txt'}", flush=True)
    print(f"wrote {args.out / 'all_modes_summary.tsv'}", flush=True)

    print("\nFinal results:")
    print((args.out / "all_modes_summary.txt").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
