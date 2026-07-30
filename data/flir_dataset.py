"""FLIR-align-3class dataset with attribute-composed text descriptions."""
import os
import json
import random
import sys

from PIL import Image
import torch
from torch.utils.data import Dataset, default_collate
import clip

# Make sibling 'scripts' package importable when run from project root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.build_text_descriptions import compose_sentence, maybe_fallback


class FLIRPromptDataSet(Dataset):
    """FLIR-align-3class dataset with per-image attribute-composed text.

    Args:
        ir_dir: path to infrared image folder (e.g. .../infrared/train)
        vis_dir: path to visible image folder
        label_dir: path to YOLO label folder
        attrs_cache: path to attrs.json (must exist, built by build_text_descriptions.py)
        transform: torchvision transform applied to both IR and Vis
        phase: 'train' or 'test'
        fallback_prob: probability of using static fallback template
        seed: optional base seed for deterministic text sampling (None = random per epoch)
    """

    def __init__(self, ir_dir, vis_dir, label_dir, attrs_cache,
                 transform=None, phase='train', fallback_prob=0.075, seed=None):
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.label_dir = label_dir
        self.transform = transform
        self.phase = phase
        self.fallback_prob = fallback_prob
        self._rng = random.Random(seed) if seed is not None else random

        # Load attribute cache
        assert os.path.exists(attrs_cache), \
            f"attrs cache not found: {attrs_cache}. Run scripts/build_text_descriptions.py first."
        with open(attrs_cache) as f:
            self.attrs = json.load(f)

        # Match IR/Vis/labels by stem
        def _stem(fn):
            return os.path.splitext(fn)[0]

        def _list_imgs(d):
            return {fn for fn in os.listdir(d)
                    if fn.lower().endswith(('.jpg', '.jpeg', '.png'))}

        ir_files = {fn for fn in os.listdir(ir_dir)
                    if fn.lower().endswith(('.jpg', '.jpeg', '.png'))}
        vis_files = _list_imgs(vis_dir)
        label_files = {fn for fn in os.listdir(label_dir) if fn.endswith('.txt')}

        ir_stems = {_stem(f): f for f in ir_files}
        vis_stems = {_stem(f): f for f in vis_files}
        label_stems = {_stem(f): f for f in label_files}

        common = sorted(set(ir_stems) & set(vis_stems) & set(label_stems) & set(self.attrs))
        self.samples = common
        self.ir_files = {s: ir_stems[s] for s in common}
        self.vis_files = {s: vis_stems[s] for s in common}
        self.label_files = {s: label_stems[s] for s in common}

    def __len__(self):
        return len(self.samples)

    def _load_bboxes(self, stem):
        """Load YOLO bboxes for a sample. Returns [N, 5] tensor.

        Each row: (class, cx_norm, cy_norm, w_norm, h_norm). Empty if no label
        file or file is empty. Used downstream for thermal-saliency supervision
        of cross-attention maps.
        """
        fn = self.label_files.get(stem)
        if fn is None:
            return torch.zeros(0, 5, dtype=torch.float32)
        path = os.path.join(self.label_dir, fn)
        boxes = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        c, cx, cy, w, h = (float(x) for x in parts[:5])
                    except ValueError:
                        continue
                    boxes.append([c, cx, cy, w, h])
        return torch.tensor(boxes, dtype=torch.float32) if boxes \
            else torch.zeros(0, 5, dtype=torch.float32)

    def __getitem__(self, idx):
        stem = self.samples[idx]
        ir_path = os.path.join(self.ir_dir, self.ir_files[stem])
        vis_path = os.path.join(self.vis_dir, self.vis_files[stem])

        # IR is grayscale L mode; convert to RGB for 3-channel consistency with model
        ir = Image.open(ir_path).convert('RGB')
        vis = Image.open(vis_path).convert('RGB')

        if self.transform is not None:
            ir = self.transform(ir)
            vis = self.transform(vis)

        attrs = self.attrs[stem]

        # Online text composition
        fallback = maybe_fallback(attrs, prob=self.fallback_prob, rng=self._rng)
        text = fallback if fallback is not None else compose_sentence(attrs, rng=self._rng)

        # CLIP tokenize returns [N] tensor for a single string
        tokens = clip.tokenize([text])[0]  # [77]

        # BBox info for thermal-saliency supervision (attention target).
        bboxes = self._load_bboxes(stem)

        return {
            'ir': ir,
            'vis': vis,
            'text': tokens,
            'text_str': text,
            'attrs': attrs,
            'stem': stem,
            'bboxes': bboxes,
        }

    def collate_fn(self, batch):
        """Custom collate that keeps variable-length 'bboxes' and string fields
        as lists while stacking fixed-shape tensors.

        Default collate tries to stack bboxes (per-sample N varies) which
        fails. This delegates fixed-shape keys to default_collate and treats
        'bboxes', 'text_str', 'attrs', 'stem' as lists.
        """
        list_keys = {'bboxes', 'text_str', 'attrs', 'stem'}
        fixed = [{k: v for k, v in s.items() if k not in list_keys}
                 for s in batch]
        out = default_collate(fixed)
        for k in list_keys:
            out[k] = [s[k] for s in batch]
        return out
