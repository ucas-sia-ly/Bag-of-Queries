"""Versioned target export contract. No model, mining or Stage2 runtime imports."""

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
TARGET_ROLES = {"attention": "primary", "fused": "supplementary"}
# Audited legacy producer: attention uses build_attention_mask's default;
# fused explicitly uses connected_topk. Unknown producers require a new adapter.
LEGACY_PRODUCER_SHA256 = {
    "src/analysis/masks.py": "c8c60d48fb3ec72c8380e3ab75d3db9f3403537c5befc951332ca0ff270cc407",
    "src/stage2/gsv_occlusion.py": "3ee84415937604b029bc4bd0f1569c2231bfadb3e9453a821920f4eb084129ef",
    "src/stage2/vulnerability.py": "72c28a8aabb844ef814cdc5fcc0eb805bcf1328ccf26194e79d14ec06185d50a",
}


class ExportError(ValueError):
    """Missing, inconsistent or unverified Stage2 inputs; nothing is published."""


def require(mapping, fields, context):
    if not isinstance(mapping, dict):
        raise ExportError(f"{context}: expected an object")
    missing = [key for key in fields if key not in mapping or mapping[key] is None or mapping[key] == ""]
    if missing:
        raise ExportError(f"{context}: missing required fields: {', '.join(missing)}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"


def binary_mask(mask, context):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not all(mask.shape) or not np.isin(mask, [0, 1]).all():
        raise ExportError(f"{context}: mask must be a nonempty 2D binary array (0/1)")
    return mask.astype(np.uint8)


def nearest_resize(mask, height, width):
    """Floor-coordinate nearest, matching Stage2 torch interpolate(nearest).

    Operates on binary values without interpolation/thresholding. The original
    mask must be resized FROM the exported 224 mask, not directly from tokens.
    """
    mask = binary_mask(mask, "nearest resize")
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise ExportError("Resize requires positive integer dimensions")
    rows = np.arange(height, dtype=np.int64) * mask.shape[0] // height
    cols = np.arange(width, dtype=np.int64) * mask.shape[1] // width
    return mask[rows[:, None], cols[None, :]]
