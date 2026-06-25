"""
Clean all generated files (checkpoints, logs, visualizations, inference results, __pycache__).

Usage:
    python clean.py          # clean all
    python clean.py --dry    # dry run (list what would be deleted)
"""

import os, sys, shutil, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Directories to COMPLETELY REMOVE
DIRS_TO_CLEAR = [
    ROOT / 'checkpoints',
    ROOT / 'logs',
    ROOT / 'visualizations',
    ROOT / 'inference_results',
]

# __pycache__ directories (recursively)
def find_pycache_dirs():
    return list(ROOT.rglob('__pycache__'))

def remove_path(path, dry=False):
    if not path.exists():
        return 0, 0
    size = 0
    files = 0
    if path.is_dir():
        for root, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(root, f)
                try:
                    size += os.path.getsize(fp)
                except OSError:
                    pass
                files += 1
        if not dry:
            shutil.rmtree(path, ignore_errors=True)
    return files, size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry', action='store_true', help='Dry run (list only, no deletion)')
    args = parser.parse_args()

    dry = args.dry
    if dry:
        print("[DRY RUN] No files will be deleted.\n")

    total_files = 0
    total_size = 0

    # 1. Main output directories
    for d in DIRS_TO_CLEAR:
        prefix = "[DRY] Would delete" if dry else "[OK] Deleted"
        f, s = remove_path(d, dry=dry)
        if f > 0:
            s_mb = s / (1024 * 1024)
            print(f"  {prefix} {d.relative_to(ROOT)}/  ({f} files, {s_mb:.1f} MB)")
            total_files += f
            total_size += s
        elif d.exists():
            print(f"  [--] {d.relative_to(ROOT)}/  (empty)")

    # 2. __pycache__ directories
    pycache_dirs = find_pycache_dirs()
    for pd in pycache_dirs:
        prefix = "[DRY] Would delete" if dry else "[OK] Deleted"
        f, s = remove_path(pd, dry=dry)
        if f > 0:
            print(f"  {prefix} {pd.relative_to(ROOT)}/  ({f} files, {s / 1024:.1f} KB)")
            total_files += f
            total_size += s

    # Summary
    total_mb = total_size / (1024 * 1024)
    print(f"\n{'[DRY RUN] Would clean' if dry else '[DONE] Cleaned'} {total_files} files ({total_mb:.1f} MB)")


if __name__ == '__main__':
    main()