#!/usr/bin/env python3
"""Portable validation of released, native YOLOv7-tiny EPFD checkpoints.

No training, automatic downloads, checkpoint selection, fusion, or label caching.
The default protocol is 640x640, batch 8, FP16, original-coordinate COCO bbox AP.
Use --help for single-checkpoint and manifest-based evaluation options.
"""

import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
METRICS = ("map50_95", "map50", "map75", "aps", "apm", "apl")
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng", ".webp", ".mpo"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path, help="JSON checkpoint list; default: bundled checkpoints.json")
    source.add_argument("--weights", type=Path, help="Evaluate one native checkpoint instead of the manifest")
    parser.add_argument("--suite", choices=("baseline", "distill", "all"),
                        help="Select a released checkpoint group; no training logs or private paths required")
    parser.add_argument("--weights-root", type=Path, default=Path("weights"))
    parser.add_argument("--ids", nargs="+", help="Subset of checkpoint IDs from the manifest")
    parser.add_argument("--checkpoint-sha256", help="Expected SHA256 for --weights")
    parser.add_argument("--yolo-root", type=Path, default=HERE.parent,
                        help="Compatible YOLOv7 project containing models/ and utils/; default: this project root")
    parser.add_argument("--data-root", type=Path, help="Dataset root containing images/val and labels/val")
    parser.add_argument("--dataset-reference", type=Path, default=HERE / "dataset_reference.json")
    parser.add_argument("--source-reference", type=Path, default=HERE / "source_reference.json")
    parser.add_argument("--allow-different-data", action="store_true",
                        help="Evaluate a different split, explicitly marked non-reference")
    parser.add_argument("--allow-different-source", action="store_true",
                        help="Allow modified model/inference source, explicitly marked non-reference")
    parser.add_argument("--device", default="0", help="One CUDA index, or cpu with --precision fp32")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0,
                        help="Data loader workers; 0 is portable across operating systems")
    parser.add_argument("--area-mode", choices=("custom", "standard"), default="custom")
    parser.add_argument("--output", type=Path, default=Path("results/epfd_validation"))
    parser.add_argument("--tolerance-pp", type=float, default=0.01,
                        help="Absolute metric tolerance in percentage points, not relative percent")
    parser.add_argument("--trust-checkpoints", action="store_true",
                        help="Required to deserialize trusted legacy PyTorch checkpoints (pickle)")
    parser.add_argument("--check-only", action="store_true",
                        help="Check source, data and weight hashes; no unpickling, inference or output writes")
    parser.add_argument("--self-check", action="store_true", help="Synthetic COCO checks only; no weights needed")
    args = parser.parse_args(argv)
    require(args.self_check or args.data_root is not None, "Specify --data-root.")
    require(args.batch_size > 0 and args.workers >= 0, "Invalid batch size or worker count.")
    require(math.isfinite(args.tolerance_pp) and args.tolerance_pp >= 0, "Invalid tolerance.")
    require(args.device == "cpu" or re.fullmatch(r"[0-9]+", args.device), "Use one CUDA index or cpu.")
    require(args.device != "cpu" or args.precision == "fp32", "CPU evaluation requires --precision fp32.")
    require(not args.weights or not args.ids, "--ids is only valid with a manifest.")
    require(not args.weights or not args.suite, "Use --weights for one file or --suite for a released group, not both.")
    require(not args.ids or not args.suite, "Select either --ids or --suite, not both.")
    require(not args.checkpoint_sha256 or args.weights, "--checkpoint-sha256 requires --weights.")
    return args


def select_checkpoints(args):
    manifest = read_json(args.manifest or HERE / "checkpoints.json")
    if args.weights:
        file_path = args.weights.resolve()
        require(file_path.is_file(), "Missing checkpoint: " + file_path.name)
        actual = sha256(file_path)
        known = next((item for item in manifest["checkpoints"] if item["sha256"] == actual), None)
        entry = dict(known) if known else {
            "id": re.sub(r"[^A-Za-z0-9._-]", "_", file_path.stem),
            "group": "single", "training_seed": None, "configuration": "unspecified"}
        entry.update(file=file_path.name, _path=file_path,
                     sha256=args.checkpoint_sha256 or (known["sha256"] if known else None))
        entries = [entry]
    else:
        entries = [dict(entry) for entry in manifest["checkpoints"]]
        ids = [entry["id"] for entry in entries]
        require(len(ids) == len(set(ids)), "Duplicate checkpoint IDs in manifest.")
        if args.suite in ("baseline", "distill"):
            group = {"baseline": "baseline_selected", "distill": "distill_configs_seed1"}[args.suite]
            entries = [entry for entry in entries if entry["group"] == group]
        if args.ids:
            require(set(args.ids).issubset(ids), "Unknown checkpoint ID in --ids.")
            entries = [entry for entry in entries if entry["id"] in args.ids]
        for entry in entries:
            name = Path(entry["file"])
            require(not name.is_absolute() and ".." not in name.parts,
                    "Manifest weight files must be relative to --weights-root.")
            entry["_path"] = args.weights_root.resolve() / name
    require(bool(entries), "No checkpoints selected.")
    for entry in entries:
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", entry["id"]), "Unsafe checkpoint ID.")
        require(entry["_path"].is_file(), "Missing checkpoint: " + entry["file"])
        actual = sha256(entry["_path"])
        expected = entry.get("sha256")
        require(expected is None or actual == expected.lower(), "Checkpoint SHA256 mismatch: " + entry["id"])
        entry["actual_sha256"] = actual
    return entries


def verify_source(root, reference, allow_different, groups=None):
    require((root / "models/yolo.py").is_file() and (root / "utils/general.py").is_file(),
            "--yolo-root must contain the compatible YOLOv7 models/ and utils/ source.")
    groups = groups or []
    common = reference["files"]
    versions = reference.get("groups", {})
    required = set(common)
    for group in groups:
        required.update(versions.get(group, {}).get("files", {}))
    hashes = {}
    for relative in sorted(required):
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "Source reference paths must be relative.")
        path = root / relative
        hashes[relative] = sha256(path) if path.is_file() else None
    common_match = all(hashes[key] == value for key, value in common.items())
    group_matches = {group: common_match and group in versions and all(
        hashes[key] == value for key, value in versions.get(group, {}).get("files", {}).items())
        for group in groups}
    mismatches = [group for group in groups if group in versions and not group_matches[group]]
    require((common_match and not mismatches) or allow_different,
            "Inference source differs from the reference for " + (", ".join(mismatches) or "shared files")
            + ". Evaluate baseline/distillation in separate processes with their matching source versions, "
            "or explicitly use --allow-different-source (non-reference evaluation).")
    return hashes, group_matches


def build_ground_truth(data_root, class_names):
    """Preserve original YOLO coordinates and historical, portable split fingerprints."""
    import cv2
    from PIL import Image
    image_root, label_root = data_root / "images/val", data_root / "labels/val"
    require(image_root.is_dir() and label_root.is_dir(), "Expected images/val and labels/val under --data-root.")
    files = sorted((path for path in image_root.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                   key=lambda path: path.relative_to(image_root).as_posix())
    require(bool(files), "No validation images found.")
    samples, images, annotations, inventory = [], [], [], []
    for image_id, path in enumerate(files, 1):
        relative = path.relative_to(image_root).as_posix()
        label = (label_root / relative).with_suffix(".txt")
        with Image.open(str(path)) as image:
            require(image.getexif().get(274, 1) in (None, 1), "Unsupported EXIF orientation: " + relative)
            width, height = image.size
            image.verify()
        decoded = cv2.imread(str(path))
        require(decoded is not None and decoded.shape[:2] == (height, width) and min(width, height) > 9,
                "Invalid image: " + relative)
        images.append(dict(id=image_id, file_name=relative, width=width, height=height))
        samples.append(dict(id=image_id, path=str(path), width=width, height=height))
        rows = label.read_text(encoding="utf-8-sig").splitlines() if label.is_file() else []
        seen = set()
        for line in rows:
            if not line.strip():
                continue
            values = tuple(float(value) for value in line.split())
            require(len(values) == 5 and all(math.isfinite(value) for value in values),
                    "Invalid YOLO label: " + relative)
            category, cx, cy, bw, bh = values
            require(category.is_integer() and 0 <= category < len(class_names)
                    and 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1
                    and values not in seen, "Invalid/duplicate YOLO box: " + relative)
            require(cx - bw / 2 >= -1e-6 and cy - bh / 2 >= -1e-6
                    and cx + bw / 2 <= 1 + 1e-6 and cy + bh / 2 <= 1 + 1e-6,
                    "GT box extends outside its original image: " + relative)
            seen.add(values)
            box = [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height]
            annotations.append(dict(id=len(annotations) + 1, image_id=image_id,
                                    category_id=int(category) + 1, bbox=box,
                                    area=box[2] * box[3], iscrowd=0))
        inventory.append([relative, sha256(path), sha256(label) if label.is_file() else None])
    gt = dict(info=dict(description="EPFD original-coordinate YOLO bounding boxes", version="1"),
              licenses=[], images=images, annotations=annotations,
              categories=[dict(id=index + 1, name=name) for index, name in enumerate(class_names)])
    require(bool(annotations), "No validation annotations found.")
    fingerprints = dict(images=len(images), annotations=len(annotations),
                        image_label_bytes_sha256=object_hash(inventory),
                        coco_gt_semantic_sha256=object_hash(gt))
    return gt, samples, fingerprints


class ValidationImages:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import cv2
        import numpy as np
        import torch
        from utils.datasets import letterbox
        sample = self.samples[index]
        image = cv2.imread(sample["path"])
        require(image is not None and image.shape[:2] == (sample["height"], sample["width"]),
                "Image changed during evaluation.")
        height, width = image.shape[:2]
        ratio = 640.0 / max(height, width)
        if ratio != 1:
            resized = (int(width * ratio), int(height * ratio))
            require(min(resized) > 0, "Invalid resized dimensions.")
            image = cv2.resize(image, resized, interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
        resized_h, resized_w = image.shape[:2]
        image, ratios, pad = letterbox(image, new_shape=(640, 640), auto=False, scaleup=False, stride=32)
        require(image.shape[:2] == (640, 640), "Expected fixed square input.")
        meta = {key: sample[key] for key in ("id", "width", "height")}
        meta.update(gain_x=resized_w / width * ratios[0], gain_y=resized_h / height * ratios[1],
                    pad_x=pad[0], pad_y=pad[1])
        image = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)))
        return image, meta


def collate_images(batch):
    import torch
    images, metadata = zip(*batch)
    return torch.stack(images), list(metadata)


def inverse_boxes(boxes, meta):
    boxes = boxes.clone()
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - meta["pad_x"]) / meta["gain_x"]).clamp(0, meta["width"])
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - meta["pad_y"]) / meta["gain_y"]).clamp(0, meta["height"])
    return boxes


def load_detector(path, class_names, torch, device):
    # Invoked only after explicit --trust-checkpoints consent and hash verification.
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    checkpoint = torch.load(str(path), **options)
    require(isinstance(checkpoint, dict), "Expected a native checkpoint dictionary, not an export or state_dict.")
    model, model_source = checkpoint.get("ema"), "ema"
    if model is None:
        require(checkpoint.get("epoch") == -1 and checkpoint.get("optimizer") is None,
                "Raw checkpoint without EMA is not the historical evaluation protocol.")
        model, model_source = checkpoint.get("model"), "model (EMA folded in by strip_optimizer)"
    require(isinstance(model, torch.nn.Module), "Checkpoint does not contain a native detector.")
    names = model.names
    if isinstance(names, dict):
        names = [names[index] for index in range(len(names))]
    head = model.model[-1]
    require(list(names) == class_names and int(head.nc) == len(class_names), "Checkpoint class order mismatch.")
    require(type(head).__name__ == "IDetect" and head.forward.__name__ == "forward"
            and any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()),
            "Expected the original unfused YOLOv7-tiny detector.")
    model = model.float().to(device).eval()
    require(max(int(model.stride.max()), 32) == 32, "Unexpected detector stride.")
    for module in model.modules():
        require(not module._forward_hooks and not module._forward_pre_hooks, "Unexpected inference hooks.")
        if isinstance(getattr(module, "stride", None), torch.Tensor):
            module.stride = module.stride.to(device)
        if isinstance(getattr(module, "grid", None), list):
            module.grid = [torch.zeros(1, device=device) for _ in module.grid]
        if isinstance(module, torch.nn.Upsample):
            module.recompute_scale_factor = None
    return model, model_source


def predict(entry, samples, class_names, args, torch, device):
    import numpy as np
    from tqdm import tqdm
    from utils.general import non_max_suppression
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(sha256(entry["_path"]) == entry["actual_sha256"], "Checkpoint changed before inference.")
    model, model_source = load_detector(entry["_path"], class_names, torch, device)
    if args.precision == "fp16":
        model.half()
    loader = torch.utils.data.DataLoader(ValidationImages(samples), batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.workers,
                                        pin_memory=device.type == "cuda", collate_fn=collate_images)
    predictions, seen = [], []
    for images, metadata in tqdm(loader, desc=entry["id"]):
        images = images.to(device, non_blocking=True)
        images = (images.half() if args.precision == "fp16" else images.float()) / 255.0
        with torch.no_grad():
            outputs = model(images, augment=False)[0]
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                detections = non_max_suppression(outputs, conf_thres=0.001, iou_thres=0.6,
                                                 multi_label=True, agnostic=False, labels=())
            require("time limit" not in buffer.getvalue().lower(), "NMS timeout; refusing incomplete results.")
        require(len(detections) == len(metadata), "Incomplete detection batch.")
        for detection, meta in zip(detections, metadata):
            seen.append(meta["id"])
            detection = detection.detach().float().cpu()
            if not len(detection):
                continue
            require(torch.isfinite(detection).all().item(), "Nonfinite prediction.")
            boxes = inverse_boxes(detection[:, :4], meta)
            for box, score, category in zip(boxes.tolist(), detection[:, 4].tolist(), detection[:, 5].tolist()):
                require(category.is_integer() and 0 <= category < len(class_names) and 0 <= score <= 1,
                        "Invalid prediction category or score.")
                x0, y0, x1, y1 = box
                if x1 > x0 and y1 > y0:
                    predictions.append(dict(image_id=meta["id"], category_id=int(category) + 1,
                                            bbox=[x0, y0, x1 - x0, y1 - y0], score=score))
    require(seen == [sample["id"] for sample in samples], "Every validation image must be evaluated exactly once.")
    require(sha256(entry["_path"]) == entry["actual_sha256"], "Checkpoint changed during inference.")
    del model, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, model_source


def compare_metrics(actual, expected, tolerance, reference_settings_match):
    if not expected:
        return {"status": "not_checked", "reason": "No expected metrics for this checkpoint."}
    differences = {key: None if actual[key] is None or expected[key] is None
                   else actual[key] - expected[key] for key in METRICS}
    within = all((actual[key] is None and expected[key] is None)
                 or (differences[key] is not None and abs(differences[key]) <= tolerance) for key in METRICS)
    return {"status": ("pass" if within else "fail") if reference_settings_match else "non_reference",
            "within_numeric_tolerance": within, "tolerance_percentage_points": tolerance,
            "delta_percentage_points": differences}


def save_summaries(output, rows, entries):
    fields = ["id", "group", "configuration", "training_seed"] + list(METRICS) + ["verification"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    for group in sorted({row["group"] for row in rows}):
        selected = [row for row in rows if row["group"] == group]
        seeds = [row["training_seed"] for row in selected]
        same_config = len({row["configuration"] for row in selected}) == 1
        repeated_seeds = same_config and None not in seeds and len(set(seeds)) == len(seeds) and len(seeds) > 1
        description = "Selected same-configuration runs with distinct training seeds." if repeated_seeds else (
            "Descriptive across-configuration statistics; NOT repeated-seed uncertainty."
            if len(selected) > 1 else "Single checkpoint; no sample standard deviation.")
        item = {"group": group, "count": len(selected), "statistics_kind": description,
                "planned_count": sum(entry["group"] == group for entry in entries), "metrics_percent": {}}
        for metric in METRICS:
            values = [row[metric] for row in selected if row[metric] is not None]
            item["metrics_percent"][metric] = {"valid_n": len(values),
                "mean": statistics.mean(values) if values else None,
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None}
        groups.append(item)
    write_json(output / "group_summary.json", groups)


def main(argv=None):
    args = arguments(argv)
    import coco_metrics
    if args.self_check:
        coco_metrics.self_check()
        return 0
    entries = select_checkpoints(args)
    dataset_reference = read_json(args.dataset_reference)
    source_reference = read_json(args.source_reference)
    root = args.yolo_root.resolve()
    selected_groups = sorted({entry["group"] for entry in entries})
    source_hashes, source_matches = verify_source(root, source_reference, args.allow_different_source, selected_groups)
    gt, samples, fingerprints = build_ground_truth(args.data_root.resolve(), dataset_reference["class_names"])
    data_match = all(fingerprints[key] == dataset_reference[key] for key in fingerprints)
    require(data_match or args.allow_different_data,
            "Validation split differs from the recorded EPFD split. Check files, labels and names; "
            "use --allow-different-data only for a separately reported evaluation.")
    print("Preflight: %d weights, %d images, %d annotations; data match=%s; source match=%s."
          % (len(entries), len(samples), len(gt["annotations"]), data_match, source_matches))
    if args.check_only:
        print("Checks completed. No checkpoint deserialization, detector inference or output writes.")
        return 0
    require(args.trust_checkpoints,
            "Legacy .pt checkpoints use pickle and can execute code. Inspect the source and hashes, "
            "then pass --trust-checkpoints only for trusted files.")
    require(not args.output.exists(), "Output already exists. Choose a new --output; existing results are never overwritten.")
    require(os.environ.get("WORLD_SIZE", "1") == "1", "Run one process, not distributed evaluation.")
    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, str(root))
    import torch
    import cv2
    import numpy as np
    require(args.device == "cpu" or torch.cuda.is_available(), "CUDA unavailable; CPU FP32 is a non-reference alternative.")
    device = torch.device("cpu" if args.device == "cpu" else "cuda:0")
    reference_matches = {group: (data_match and source_matches[group] and args.precision == "fp16"
                         and device.type == "cuda" and args.batch_size == 8 and args.area_mode == "custom")
                         for group in selected_groups}
    reference_settings_match = all(reference_matches.values())
    coco_metrics.self_check()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"input_size": [640, 640], "batch_size": args.batch_size, "precision": args.precision,
                "confidence_threshold": 0.001, "nms_iou": 0.6, "nms_max_det": 300,
                "coco_max_dets": [1, 10, 100], "area_mode": args.area_mode,
                "augment": False, "fusion": False, "rectangular_batches": False,
                "evaluation_seed": 1, "reference_settings_match": reference_settings_match,
                "reference_settings_match_by_group": reference_matches,
                "dataset": fingerprints, "inference_source_sha256": source_hashes,
                "validation_code_sha256": {name: sha256(HERE / name)
                                           for name in ("validate_epfd.py", "coco_metrics.py")},
                "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                "numpy": np.__version__, "opencv": cv2.__version__,
                                "device": "cpu" if device.type == "cpu" else torch.cuda.get_device_name(0)},
                "checkpoints": [{key: value for key, value in entry.items() if not key.startswith("_")}
                                for entry in entries],
                "note": "A matching protocol does not guarantee bitwise agreement across software/GPU versions."}
    write_json(args.output / "evaluation_protocol.json", protocol)
    write_json(args.output / "ground_truth_coco.json", gt)
    rows, failed = [], False
    for entry in entries:
        predictions, model_source = predict(entry, samples, dataset_reference["class_names"], args, torch, device)
        result = coco_metrics.evaluate_coco(gt, predictions, args.area_mode)
        actual = {key: None if result["metrics"][key] is None else 100 * result["metrics"][key] for key in METRICS}
        comparison = compare_metrics(actual, entry.get("expected_percent"), args.tolerance_pp,
                                     reference_matches[entry["group"]])
        failed |= comparison["status"] == "fail"
        folder = args.output / entry["id"]
        folder.mkdir()
        result.update(metrics_percent=actual, verification=comparison, model_source=model_source,
                      checkpoint_file=entry["file"], checkpoint_sha256=entry["actual_sha256"])
        write_json(folder / "predictions_coco.json", predictions)
        write_json(folder / "metrics.json", result)
        rows.append(dict(id=entry["id"], group=entry["group"], configuration=entry["configuration"],
                         training_seed=entry["training_seed"], verification=comparison["status"], **actual))
        save_summaries(args.output, rows, entries)
        print(entry["id"] + ": " + ", ".join(key + "=" + ("NA" if value is None else "%.4f" % value)
                                             for key, value in actual.items()) + " [%s]" % comparison["status"])
    _, _, after = build_ground_truth(args.data_root.resolve(), dataset_reference["class_names"])
    require(after == fingerprints, "Dataset changed during evaluation; results are not validated.")
    after_source, _ = verify_source(root, source_reference, True, selected_groups)
    require(after_source == source_hashes, "Inference source changed during evaluation.")
    write_json(args.output / "completed.json", {"completed_checkpoints": len(rows),
               "metric_comparisons_passed": None if not any(row["verification"] in ("pass", "fail") for row in rows)
               else not failed, "reference_settings_match": reference_settings_match})
    print("Saved summary.csv, group_summary.json and per-checkpoint predictions/metrics.")
    return 2 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
