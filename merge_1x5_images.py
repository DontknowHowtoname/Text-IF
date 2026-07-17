"""Merge infrared, visible, and 3 fused outputs side by side (1x5).

Order:
  1. LLVIP infrared  (test/<id>.jpg)
  2. LLVIP visible   (test/<id>.jpg)
  3. reconstruction_best fused  (fused_<id>.jpg)
  4. textif_unet_best fused     (fused_<id>.jpg)
  5. textif_full_recon_v2_eval_ft_myweight fused (<id>.png)
"""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET = Path(r"D:\StudyFiles\MachineLearning\datasets\LLVIP")
FUSED_A = Path(r"E:\gtj\codes\myRestormer_codex\results\reconstruction_eval\reconstruction_best_20260410-105457\fused")
FUSED_B = Path(r"E:\gtj\codes\myRestormer_codex\results\textif_unet_eval_3\textif_unet_best_20260403-014803\fused")
FUSED_C = ROOT / "results" / "textif_full_recon_v2_eval_ft_myweight" / "fused"
OUT = ROOT / "results" / "merged_1x5_LLVIP"
OUT.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def main() -> None:
    # IDs from FUSED_C (uses bare id.png), intersect with fused_<id>.jpg sets.
    c_ids = {p.stem for p in FUSED_C.glob("*.png")}
    a_ids = {p.stem[len("fused_"):] for p in FUSED_A.glob("fused_*.jpg")}
    b_ids = {p.stem[len("fused_"):] for p in FUSED_B.glob("fused_*.jpg")}
    ids = sorted(c_ids & a_ids & b_ids)
    print(f"Found {len(ids)} common image IDs")

    done = skipped = 0
    for img_id in ids:
        paths = [
            DATASET / "infrared" / "test" / f"{img_id}.jpg",
            DATASET / "visible" / "test" / f"{img_id}.jpg",
            FUSED_A / f"fused_{img_id}.jpg",
            FUSED_B / f"fused_{img_id}.jpg",
            FUSED_C / f"{img_id}.png",
        ]
        if not all(p.exists() for p in paths):
            skipped += 1
            continue

        imgs = [load_rgb(p) for p in paths]
        h = min(im.height for im in imgs)
        resized = []
        for im in imgs:
            if im.height != h:
                w = round(im.width * h / im.height)
                im = im.resize((w, h), Image.BILINEAR)
            resized.append(im)

        total_w = sum(im.width for im in resized)
        canvas = Image.new("RGB", (total_w, h), (255, 255, 255))
        x = 0
        for im in resized:
            canvas.paste(im, (x, 0))
            x += im.width
        canvas.save(OUT / f"{img_id}_merged.png")
        done += 1

    print(f"Done. {done} merged into {OUT}; {skipped} skipped.")


if __name__ == "__main__":
    main()
