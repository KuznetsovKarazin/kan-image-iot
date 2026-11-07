"""Deterministic COCO subset; image bytes stay outside Git."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(cache, out):
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    archive = cache / "annotations_trainval2014.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        raw = z.read("annotations/instances_val2014.json")
    ann = json.loads(raw)
    ids = set(map(int, (ROOT / "scripts/mscoco_minival_ids.txt").read_text().split()))
    candidates = {im["id"]: im for im in ann["images"] if im["id"] in ids}
    positive = set()
    for a in ann["annotations"]:
        if a["category_id"] == 1 and a["image_id"] in candidates:
            im = candidates[a["image_id"]]
            if a["area"] / (im["width"] * im["height"]) > .005:
                positive.add(im["id"])
    records = []
    for label in [0, 1]:
        eligible = [i for i in candidates if int(i in positive) == label]
        eligible.sort(key=lambda i: hashlib.sha256(f"micro-pilot-20261107:{i}".encode()).hexdigest())
        if len(eligible) < 640:
            raise RuntimeError(f"Insufficient eligible images for class {label}: {len(eligible)}")
        for j, i in enumerate(eligible[:640]):
            im = candidates[i]
            name = im["file_name"]
            records.append({"image_id": i, "label": label,
                "split": "calibration" if j < 128 else "evaluation",
                "filename": name,
                "url": "https://s3.amazonaws.com/images.cocodataset.org/val2014/" + name,
                "coco_license_id": im["license"]})
    # Written before downloads and before any predictions.
    (out / "selection.json").write_text(json.dumps(records, indent=2))
    raw_dir, proc_dir = cache / "original", cache / "processed224"
    raw_dir.mkdir(exist_ok=True)
    proc_dir.mkdir(exist_ok=True)

    def get_image(rec):
        path = raw_dir / rec["filename"]
        if not path.exists():
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(rec["url"], timeout=30) as r:
                        data = r.read()
                    with Image.open(io.BytesIO(data)) as im:
                        im.verify()
                    path.write_bytes(data)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        dest = proc_dir / rec["filename"]
        with Image.open(path) as im:
            im.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS).save(dest)
        return dict(rec, original_sha256=digest(path), processed_sha256=digest(dest))

    finished = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(get_image, r) for r in records]
        for future in as_completed(futures):
            rec = future.result()
            finished[rec["image_id"]] = rec
            if len(finished) % 128 == 0:
                print(f"Downloaded and verified {len(finished)}/{len(records)}", flush=True)
    manifest = [finished[r["image_id"]] for r in records]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    meta = {"annotation_url": URL, "archive_sha256": digest(archive),
        "annotation_member_sha256": hashlib.sha256(raw).hexdigest(),
        "minival_ids_sha256": digest(ROOT / "scripts/mscoco_minival_ids.txt"),
        "eligible_count": len(candidates), "eligible_positive": len(positive),
        "n_calibration": 256, "n_evaluation": 1024, "pillow": PIL.__version__,
        "label_rule": "category_id=1 and annotation area / image area > 0.005",
        "licenses": ann["licenses"], "selection_sha256": digest(out / "selection.json")}
    (out / "data_provenance.json").write_text(json.dumps(meta, indent=2))
    print("Image preparation complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results/images")
    a = p.parse_args()
    prepare(a.cache, a.out)
