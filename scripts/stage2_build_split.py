"""Build Stage 2 manifests: python scripts/stage2_build_split.py [--seed 0]."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage2.data.gsv_split import DEFAULT_DATAFRAMES_DIR, build_split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataframes-dir", type=Path, default=DEFAULT_DATAFRAMES_DIR)
    parser.add_argument("--cities", nargs="+", help="CSV stems; default: all available cities")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs/stage2/split")
    args = parser.parse_args(argv)
    cities = args.cities if args.cities is not None else sorted(
        path.stem for path in args.dataframes_dir.glob("*.csv")
    )
    try:
        manifest = build_split(cities, seed=args.seed, dataframes_dir=args.dataframes_dir)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Split build failed: {exc}\n")
    paths = manifest.write(args.output_dir)
    counts = manifest.summary["totals"]
    print(f"GO: seed={args.seed}, cities={len(manifest.summary['cities'])}, "
          f"places={counts['kept_places']}, excluded_places={counts['excluded_places']}, "
          f"duplicate_rows={counts['duplicate_rows']}")
    print(f"Rows={counts['output_images']}, SOURCE={counts['source_images']}, "
          f"SUPPORT={counts['support_images']}; every SOURCE place has exactly 2 SUPPORT views.")
    print(f"Manifest SHA256: {manifest.summary['manifest_sha256']}")
    for path in paths:
        print(path)
    return manifest


if __name__ == "__main__":
    main()
