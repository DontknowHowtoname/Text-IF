"""Prepare TNO Image Fusion Dataset into a unified ir/ + vi/ folder structure.

TNO has heterogeneous layouts across 4 camera systems:
  - Athena_images/<scene>/[view_x/]  IR_xxx.bmp + VIS_xxx.bmp
  - DHV_images/<scene>/               IR_xxx.bmp + VIS_xxx.bmp
  - DHV_images/Fire_sequence/part_x/{thermal,dhv}/  RADxx.bmp + DHVxx.bmp
  - FEL_images/<scene>/{thermal,visual}/  <id>i.bmp + <id>v.bmp
  - Triclobs_images/<scene>/          IRxx.bmp + Visxx.bmp (+ NIRxx.bmp)
  - tank/                             LWIR.tif + Vis.tif

Strategy: build a list of (ir_path, vis_path, key) by:
  1) Grouping all image files by their **scene dir** (parent or grandparent).
  2) Within each scene, classify each file as IR or VIS by name + path tokens.
  3) Match by trailing number; dedup so each VIS file is used at most once,
     preferring IR (thermal) > NIR variants.

Output: dataset/TNO/{ir,vi}/TNO_XXXX.png as 8-bit grayscale.
"""

from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

SRC = Path(r"D:/StudyFiles/MachineLearning/datasets/1008029/TNO_Image_Fusion_Dataset")
DST = Path(r"d:/StudyFiles/MachineLearning/codes/Text-IF/dataset/TNO")
IR_OUT = DST / "ir"
VI_OUT = DST / "vi"

IMG_EXT = {".bmp", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
NUM_RE = re.compile(r"(\d+)")

# Folder tokens that indicate IR/thermal modality
IR_DIR_TOKENS = {"thermal", "ir", "lwir", "nir"}
VIS_DIR_TOKENS = {"visual", "vis", "vi", "dhv"}


def classify(path: Path) -> tuple[str | None, str]:
    """Return (modality, key) where modality in {'ir','vis',None}.

    modality is inferred from filename + ancestor folder names.
    key is the trailing number used to pair across modalities.
    """
    name = path.name
    stem = path.stem
    # aggregate ancestor folder names (lowercased)
    ancestors = " ".join(p.name.lower() for p in path.relative_to(SRC).parents[:-1])

    # Determine modality by filename tokens first, then by folder tokens.
    nlow = name.lower()
    ir_score, vis_score = 0, 0
    # Filename tokens (use [^a-z] so underscore counts as separator)
    if re.search(r"(^|[^a-z])lwir([^a-z]|$)", nlow): ir_score += 5
    if re.search(r"(^|[^a-z])rad\d", nlow): ir_score += 5
    if re.search(r"(^|[^a-z])ir([^a-z]|$)", nlow): ir_score += 4
    if re.search(r"(^|[^a-z])nir([^a-z]|$)", nlow): ir_score += 2
    if re.search(r"\di\.(bmp|tif|tiff|png|jpg|jpeg)$", nlow): ir_score += 3
    if re.search(r"(^|[^a-z])vis([^a-z]|$)", nlow): vis_score += 4
    if re.search(r"(^|[^a-z])dhv\d", nlow): vis_score += 4
    if re.search(r"\dv\.(bmp|tif|tiff|png|jpg|jpeg)$", nlow): vis_score += 3
    # Ancestor folder tokens
    for tok in IR_DIR_TOKENS:
        if re.search(rf"(^|\W){tok}(\W|$)", ancestors): ir_score += 1
    for tok in VIS_DIR_TOKENS:
        if re.search(rf"(^|\W){tok}(\W|$)", ancestors): vis_score += 1

    modality = None
    if ir_score > vis_score and ir_score >= 2:
        modality = "ir"
    elif vis_score > ir_score and vis_score >= 2:
        modality = "vis"

    key = (NUM_RE.findall(stem) or [""])[-1]
    return modality, key


def scene_key(path: Path) -> str:
    """Group files that belong to the same scene.

    For FEL/DHV Fire_sequence with sibling thermal/visual folders we use
    the grandparent; otherwise the parent.
    """
    rel_parts = path.relative_to(SRC).parts
    if len(rel_parts) >= 2 and rel_parts[-2].lower() in IR_DIR_TOKENS | VIS_DIR_TOKENS:
        # Use grandparent (the scene dir) as grouping key.
        return "/".join(rel_parts[:-2])
    return "/".join(rel_parts[:-1])


def to_gray_png_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        lo, hi = np.percentile(img, [2, 98])
        if hi <= lo:
            hi = lo + 1
        img = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        mx = float(img.max()) if img.max() > 0 else 1.0
        img = (img.astype(np.float32) / mx * 255).astype(np.uint8)
    return img


def main(dry_run: bool = False) -> None:
    if not dry_run:
        IR_OUT.mkdir(parents=True, exist_ok=True)
        VI_OUT.mkdir(parents=True, exist_ok=True)

    # group all image files by scene
    scenes: dict[str, list[Path]] = defaultdict(list)
    for f in SRC.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXT:
            scenes[scene_key(f)].append(f)

    pairs: list[tuple[Path, Path, str]] = []
    for scene, files in sorted(scenes.items()):
        ir_files: dict[str, list[Path]] = defaultdict(list)  # key -> list (priority order)
        vis_files: dict[str, list[Path]] = defaultdict(list)
        for f in files:
            modality, key = classify(f)
            if modality == "ir":
                ir_files[key].append(f)
            elif modality == "vis":
                vis_files[key].append(f)
        # match by key
        used_vis: set[Path] = set()
        for key, irs in sorted(ir_files.items()):
            if key and key in vis_files:
                v = vis_files[key][0]
                if v in used_vis:
                    continue
                used_vis.add(v)
                # prefer LWIR > RAD > IR > NIR — choose ir file with highest score
                ir = sorted(irs, key=lambda p: -classify(p)[0].count("ir") or 0)[0]
                pairs.append((ir, v, f"{scene}__{key}"))
            elif not key and len(vis_files.get("", [])) == 1 and len(irs) == 1:
                v = vis_files[""][0]
                if v not in used_vis:
                    used_vis.add(v)
                    pairs.append((irs[0], v, scene))

    pairs = sorted(set(pairs), key=lambda x: x[2])
    print(f"Found {len(pairs)} IR/VIS pairs across TNO.")
    if dry_run:
        for ir, vis, key in pairs:
            print(f"  [{key}]  IR={ir.relative_to(SRC)}  VIS={vis.relative_to(SRC)}")
        return

    written = 0
    for idx, (ir, vis, _key) in enumerate(pairs, start=1):
        try:
            ir_img = cv2.imdecode(np.fromfile(str(ir), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            vi_img = cv2.imdecode(np.fromfile(str(vis), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if ir_img is None or vi_img is None:
                print(f"  [skip] cannot read {ir.name}/{vis.name}")
                continue
            if ir_img.shape[:2] != vi_img.shape[:2]:
                vi_img = cv2.resize(vi_img, (ir_img.shape[1], ir_img.shape[0]))
            ir_g = to_gray_png_uint8(ir_img)
            vi_g = to_gray_png_uint8(vi_img)
            out_name = f"TNO_{idx:04d}.png"
            cv2.imwrite(str(IR_OUT / out_name), ir_g)
            cv2.imwrite(str(VI_OUT / out_name), vi_g)
            written += 1
        except Exception as e:
            print(f"  [err] {_key}: {e}")

    print(f"Wrote {written} paired images to {DST}")
    print(f"  ir/: {len(list(IR_OUT.glob('*.png')))} files")
    print(f"  vi/: {len(list(VI_OUT.glob('*.png')))} files")


if __name__ == "__main__":
    import sys
    main(dry_run="--dry" in sys.argv)
