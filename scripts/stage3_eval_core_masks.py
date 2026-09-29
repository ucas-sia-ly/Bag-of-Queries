"""Evaluate chosen dev Core masks and the old 6% ellipse, with exact translations."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.stage3.core_mask_eval import run_core_mask_eval


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptvpr-root",type=Path)
    parser.add_argument("--candidate-dir",type=Path)
    parser.add_argument("--output-dir",default="outputs/stage3/dev/core_mask_eval")
    parser.add_argument("--seed",type=int,default=0)
    args=parser.parse_args(argv)
    try:
        result=run_core_mask_eval(**vars(args))
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.exit(2,f"Core mask evaluation failed: {exc}\n")
    print(json.dumps({key:result[key] for key in ("status","chosen_masks","paired_rows","selected_weighted_coverage_threshold","statistics")},indent=2))


if __name__=="__main__": main()
