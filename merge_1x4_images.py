"""Merge infrared, visible, baseline, recon-v2 detection images side by side (1x4)."""
import os
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET = Path(r"D:\StudyFiles\MachineLearning\datasets\LLVIP")
BASELINE = ROOT / "results" / "LLVIP_visible_baseline_detection" / "visualizations"
RECON_V2 = ROOT / "results" / "textif_full_recon_v2_eval_LLVIP_detection" / "visualizations"
OUT = ROOT / "results" / "merged_1x4_LLVIP"
OUT.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def main() -> None:
    ids = sorted(
        {p.stem.replace("_det", "") for p in BASELINE.glob("*_det.png")}
        & {p.stem.replace("_det", "") for p in RECON_V2.glob("*_det.png")}
    )
    print(f"Found {len(ids)} common image IDs")

    missing_report = []
    for idx, img_id in enumerate(ids):
        ir_path = DATASET / "infrared" / "test" / f"{img_id}.jpg"
        vis_path = DATASET / "visible" / "test" / f"{img_id}.jpg"
        base_path = BASELINE / f"{img_id}_det.png"
        recon_path = RECON_V2 / f"{img_id}_det.png"

        if not all(p.exists() for p in [ir_path, vis_path, base_path, recon_path]):
            missing_report.append(img_id)
            continue

        imgs = [load_rgb(p) for p in [ir_path, vis_path, base_path, recon_path]]

        # Normalize heights to the smallest, preserving aspect ratios.
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

        if (idx + 1) % 200 == 0:
            print(f"  processed {idx + 1}/{len(ids)}")

    print(f"Done. {len(ids) - len(missing_report)} merged into {OUT}")
    if missing_report:
        print(f"Skipped {len(missing_report)} IDs with missing sources, e.g. {missing_report[:5]}")


if __name__ == "__main__":
    main()
