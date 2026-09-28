"""Export verified Stage2 masks, without model inference or vulnerability mining."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage3.contracts import ExportError
from src.stage3.export_targets import ROOT, export_targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-dir", type=Path, default=ROOT / "outputs/stage2/gsv_occlusion/support")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/stage3/targets")
    parser.add_argument("--stage2-commit", help="Explicit legacy provenance commit; artifacts/code must match Git")
    parser.add_argument("--mask-mode", choices=["connected_topk"], help="Explicit legacy mode, verified against producer hashes")
    parser.add_argument("--seeds", nargs="+", type=int, help="Existing Stage2 seeds only; default: all recorded seeds")
    parser.add_argument("--include-fused", action="store_true", help="Include fused with target_role=supplementary")
    args = parser.parse_args(argv)
    try:
        manifest = export_targets(**vars(args))
    except (ExportError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Stage3 export failed: {exc}\n")
    print(json.dumps({"output_dir": str(args.output_dir), "record_count": manifest["record_count"],
                      "stage2_commit": manifest["stage2_commit"], "bag_of_queries_head": manifest["bag_of_queries_head"]}, indent=2))


if __name__ == "__main__":
    main()
