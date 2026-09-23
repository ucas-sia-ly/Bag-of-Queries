"""GSV support-only descriptor caches and full-database retrieval contexts."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import v2 as T

from src.analysis.retrieval import retrieval_metrics
from src.stage2.data.gsv_split import SplitRecord


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class GSVSplit:
    support: tuple[SplitRecord, ...]
    source: tuple[SplitRecord, ...]
    sha256: str

    @classmethod
    def read(cls, path):
        support, source, seen = [], [], set()
        counts = defaultdict(Counter)
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                digest.update(line)
                try:
                    row = json.loads(line)
                    record = SplitRecord(**dict(row, role=row["role"].upper()))
                    key = PurePosixPath(record.image_key)
                    if (key.is_absolute() or ".." in key.parts or "\\" in record.image_key
                            or len(key.parts) != 2 or key.parts[0] != record.city_id):
                        raise ValueError("image_key must be city/filename relative to Images")
                    if (type(record.local_place_id) is not int or record.local_place_id < 0
                            or record.place_key != f"{record.city_id}:{record.local_place_id}"):
                        raise ValueError("place identity mismatch")
                    if record.role not in ("SOURCE", "SUPPORT"):
                        raise ValueError("unknown role")
                    if record.image_key in seen:
                        raise ValueError("duplicate image / Source-Support leakage")
                except (KeyError, TypeError, AttributeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                seen.add(record.image_key)
                counts[record.place_key][record.role] += 1
                (support if record.role == "SUPPORT" else source).append(record)
        if len(counts) < 2:
            raise ValueError("Retrieval requires at least two places for global negatives")
        if any(c["SUPPORT"] != 2 or c["SOURCE"] < 1 for c in counts.values()):
            raise ValueError("Every split place must have exactly 2 SUPPORT and at least 1 SOURCE")
        return cls(tuple(support), tuple(source), digest.hexdigest())

    def sample_sources(self, count=100, seed=0):
        if not 1 <= count <= len(self.source):
            raise ValueError("Invalid SOURCE sample size")
        return tuple(sorted(self.source, key=lambda r: (
            hashlib.sha256(f"{seed}\0{r.image_key}".encode()).digest(), r.image_key,
        ))[:count])


def evaluation_transform(image_size=(224, 224)):
    """VPRDataModule deterministic resize/scaling/normalization; no RandAugment."""
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("image_size must contain two positive dimensions")
    return T.Compose([
        T.Resize(tuple(image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ])


class GSVImages(Dataset):
    def __init__(self, records, images_root, image_size=(224, 224)):
        self.records = records
        self.images_root = Path(images_root)
        self.transform = evaluation_transform(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path = self.images_root / self.records[index].image_key
        try:
            image = torchvision.io.decode_image(path, mode="RGB")
            return self.transform(image), index
        except Exception as exc:
            raise RuntimeError(f"Cannot decode GSV image: {path}") from exc


def validate_descriptors(values, rows=None, dim=None):
    if (not isinstance(values, torch.Tensor) or values.ndim != 2 or min(values.shape) == 0
            or values.dtype != torch.float32
            or (rows is not None and len(values) != rows)
            or (dim is not None and values.shape[1] != dim)):
        raise ValueError("Descriptors must be non-empty float32 [N,D] with the expected shape")
    for chunk in values.split(2048):
        if not torch.isfinite(chunk).all():
            raise ValueError("Non-finite descriptors")
        if not torch.allclose(chunk.norm(dim=1), torch.ones(len(chunk), device=chunk.device), atol=2e-5, rtol=0):
            raise ValueError("Descriptors must have unit L2 norm (no zero vectors)")


@torch.inference_mode()
def encode_images(model, dataset, *, device="cpu", batch_size=32, workers=0, destination=None):
    """Freeze a BoQ-compatible model and encode in dataset order, without AMP.

    ``destination`` may be a disk-backed float32 tensor to bound RAM usage.
    The model returns descriptors or the usual (descriptors, attentions) pair.
    """
    if not len(dataset) or batch_size < 1 or workers < 0:
        raise ValueError("Require a nonempty dataset, positive batch size and nonnegative workers")
    model.eval().requires_grad_(False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=str(device).startswith("cuda"))
    count, last_log = 0, time.monotonic()
    for images, indices in loader:
        if indices.tolist() != list(range(count, count + len(images))):
            raise ValueError("Unexpected descriptor/image ordering")
        output = model(images.to(device, non_blocking=True))
        values = output[0] if isinstance(output, (tuple, list)) else output
        validate_descriptors(values, rows=len(images))
        if destination is None:
            destination = torch.empty((len(dataset), values.shape[1]), dtype=torch.float32)
        if destination.shape != (len(dataset), values.shape[1]) or destination.dtype != torch.float32:
            raise ValueError("Descriptor destination shape/dtype mismatch")
        destination[count:count + len(images)].copy_(values.detach().cpu())
        count += len(images)
        if time.monotonic() - last_log >= 20 or count == len(dataset):
            print(f"Encoded {count}/{len(dataset)} images", flush=True)
            last_log = time.monotonic()
    return destination


def cache_identity(split, images_root, checkpoint, model_config, image_size, batch_size, device):
    """Bind cache to input order, source split, weights, code and preprocessing."""
    digest = hashlib.sha256()
    for record in split.support:
        stat = (Path(images_root) / record.image_key).stat()
        digest.update(json.dumps([record.image_key, stat.st_size, stat.st_mtime_ns]).encode())
    sources = ["src/backbones.py", "src/boq.py", "hubconf.py", "scripts/visualize_attention.py",
               "src/stage2/retrieval.py"]
    return {
        "schema_version": 1, "split_sha256": split.sha256,
        "checkpoint_sha256": file_sha256(checkpoint), "model": model_config,
        "images_root": str(Path(images_root).resolve()), "support_files_sha256": digest.hexdigest(),
        "num_support": len(split.support), "image_size": list(image_size), "mean": MEAN, "std": STD,
        "preprocessing": "decode RGB uint8; bicubic antialias; float32 /255; ImageNet Normalize; no augmentation",
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "dtype": "float32", "device": str(device), "batch_size": batch_size,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "tf32": torch.backends.cuda.matmul.allow_tf32, "xformers": False,
        "source_sha256": {name: file_sha256(ROOT / name) for name in sources},
    }


def load_support_cache(path, split, identity):
    if identity.get("split_sha256") != split.sha256 or identity.get("num_support") != len(split.support):
        raise ValueError("Support cache identity does not describe the current split")
    cache = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if cache.get("identity") != identity:
        raise ValueError("Support cache identity mismatch; choose a new cache path")
    if (cache.get("image_keys") != [r.image_key for r in split.support]
            or cache.get("place_keys") != [r.place_key for r in split.support]):
        raise ValueError("Support cache image/place ordering mismatch")
    validate_descriptors(cache["descriptors"], len(split.support), identity["model"]["descriptor_dim"])
    return cache["descriptors"]


def get_support_cache(path, model, split, images_root, identity, *, device="cpu", workers=0):
    path = Path(path)
    if path.exists():
        return load_support_cache(path, split, identity), True
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix(".building.raw")
    temporary = path.with_suffix(".building.pt")
    rows, dim = len(split.support), identity["model"]["descriptor_dim"]
    try:
        destination = torch.from_file(str(raw), shared=True, size=rows * dim, dtype=torch.float32).reshape(rows, dim)
        dataset = GSVImages(split.support, images_root, identity["image_size"])
        encode_images(model, dataset, device=device, batch_size=identity["batch_size"],
                      workers=workers, destination=destination)
        validate_descriptors(destination, rows, dim)
        # Explicit buffered IO handles large writes on non-ASCII paths. PyTorch
        # 2.5's filename fallback uses raw FileIO, which can short-write >2 GiB.
        with temporary.open("wb") as stream:
            torch.save({"identity": identity, "image_keys": [r.image_key for r in split.support],
                        "place_keys": [r.place_key for r in split.support], "descriptors": destination}, stream)
        result = load_support_cache(temporary, split, identity)
        temporary.replace(path)
    finally:
        raw.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return result, False


@torch.inference_mode()
def place_prototypes(descriptors, records):
    """Return sorted place keys and normalize(mean(support descriptors))."""
    validate_descriptors(descriptors, rows=len(records))
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        if record.role != "SUPPORT":
            raise ValueError("Prototypes may only contain SUPPORT images")
        grouped[record.place_key].append(index)
    if any(len(indices) < 2 for indices in grouped.values()):
        raise ValueError("Each prototype requires at least two SUPPORT images")
    keys = sorted(grouped)
    prototypes = torch.empty((len(keys), descriptors.shape[1]), device=descriptors.device)
    # Equal support counts allow bounded vectorized gathering; also accept variable counts.
    for start in range(0, len(keys), 512):
        chunk = [grouped[key] for key in keys[start:start + 512]]
        if len({len(ids) for ids in chunk}) == 1:
            means = descriptors[torch.tensor(chunk, device=descriptors.device)].mean(dim=1)
        else:
            means = torch.stack([descriptors[ids].mean(dim=0) for ids in chunk])
        if (means.norm(dim=1) <= 1e-12).any():
            raise ValueError("A SUPPORT mean cancels to zero; cannot form a prototype")
        prototypes[start:start + len(chunk)] = F.normalize(means, dim=1)
    validate_descriptors(prototypes)
    return keys, prototypes


class RetrievalContext:
    """References + place-level positives; all other places are global negatives.

    Query a single source with one or several descriptor variants. Metrics and
    deterministic tie handling are delegated unchanged to Stage 1 retrieval.
    """
    def __init__(self, references, reference_place_keys, *, mode="support"):
        validate_descriptors(references, rows=len(reference_place_keys))
        self.references = references
        self.reference_place_keys = tuple(reference_place_keys)
        self.mode = mode
        self._positives = defaultdict(list)
        for index, key in enumerate(reference_place_keys):
            self._positives[key].append(index)
        if len(self._positives) < 2:
            raise ValueError("Retrieval requires at least two places")

    @classmethod
    def from_support(cls, descriptors, records, *, mode="support", device="cpu"):
        if any(record.role != "SUPPORT" for record in records):
            raise ValueError("References may only contain SUPPORT images")
        if len({r.image_key for r in records}) != len(records):
            raise ValueError("Duplicate SUPPORT images")
        if any(count < 2 for count in Counter(r.place_key for r in records).values()):
            raise ValueError("Every reference place requires at least two SUPPORT images")
        validate_descriptors(descriptors, rows=len(records))
        if mode == "prototype":
            keys, references = place_prototypes(descriptors, records)
        elif mode == "support":
            keys, references = [r.place_key for r in records], descriptors
        else:
            raise ValueError("mode must be support or prototype")
        return cls(references.to(device), keys, mode=mode)

    def positive_indices(self, place_key):
        if place_key not in self._positives:
            raise ValueError(f"No SUPPORT references for place {place_key}")
        return tuple(self._positives[place_key])

    def negative_indices(self, place_key):
        mask = torch.ones(len(self.references), dtype=torch.bool, device=self.references.device)
        mask[list(self.positive_indices(place_key))] = False
        return mask.nonzero().flatten()

    def query(self, descriptors, place_key, *, clean_descriptor=None):
        if descriptors.ndim == 1:
            descriptors = descriptors.unsqueeze(0)
        descriptors = descriptors.to(self.references.device)
        validate_descriptors(descriptors, dim=self.references.shape[1])
        if clean_descriptor is None:
            if len(descriptors) != 1:
                raise ValueError("Multiple variants require an explicit clean_descriptor")
            clean_descriptor = descriptors[0]
        clean_descriptor = clean_descriptor.reshape(1, -1).to(self.references.device)
        validate_descriptors(clean_descriptor, rows=1, dim=self.references.shape[1])
        return retrieval_metrics(descriptors, self.references, self.positive_indices(place_key), clean_descriptor)
