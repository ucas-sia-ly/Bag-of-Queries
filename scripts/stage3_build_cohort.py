"""Build only an independent Stage3 dev cohort; keep the Stage2 split frozen."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage3.cohort import build_cohort


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="outputs/stage2/split/gsv_split.jsonl")
    parser.add_argument("--stage2-configs", nargs="+", help="Existing seed0/seed1 configs; defaults to support and prototype")
    parser.add_argument("--targets", default="outputs/stage3/targets/targets.jsonl")
    parser.add_argument("--output-dir", default="outputs/stage3/dev")
    parser.add_argument("--dev", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        summary = build_cohort(**vars(args))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Stage3 cohort build failed: {exc}\n")
    print(json.dumps(dict(output_dir=args.output_dir, dev=summary["dev"], seed=summary["seed"],
                          excluded_stage2_places=summary["stage2_audit"]["excluded_place_count"],
                          eligible_source_places=summary["eligible_source_place_count"],
                          full_support_images=summary["reference_definition"]["num_images"],
                          eval_created=False), indent=2))


if __name__ == "__main__":
    main()
