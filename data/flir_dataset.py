"""FLIR-align-3class dataset with attribute-composed text descriptions."""
import os
import json
import random
import sys

from PIL import Image
import torch
from torch.utils.data import Dataset
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

        return {
            'ir': ir,
            'vis': vis,
            'text': tokens,
            'text_str': text,
            'attrs': attrs,
            'stem': stem,
        }
