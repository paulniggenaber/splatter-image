# entirely new code file from Taya
"""
Save all ground-truth views (the real dataset images) of some test objects as PNGs, named by view index
and angle from the input camera, e.g. for a "what the data looks like" figure.

  python dump_views.py --objects 263 --out results/views_dump
Output: <out>/obj<k>/<view>_<angle>deg.png (128x128), view 000 is the input.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from datasets.dataset_factory import get_dataset
from eval_budget import FreeCacheDataset, load_model


def main(a):
    _, cfg = load_model("cars", None, torch.device("cpu"), "model_latest.pth")
    test = get_dataset(cfg, "test")
    loader = DataLoader(Subset(FreeCacheDataset(test), a.objects), batch_size=1)
    for obj, data in zip(a.objects, loader):
        c = data["camera_centers"][0]
        n = torch.nn.functional.normalize(c, dim=-1)
        ang = torch.rad2deg(torch.acos((n * n[:1]).sum(-1).clamp(-1, 1)))
        d = os.path.join(a.out, f"obj{obj}")
        os.makedirs(d, exist_ok=True)
        for v in range(data["gt_images"].shape[1]):
            img = (data["gt_images"][0, v].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(d, f"{v:03d}_{ang[v]:.0f}deg.png"))
        print(f"object {obj}: {data['gt_images'].shape[1]} views saved to {d}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--objects", nargs="+", type=int, default=[263])
    p.add_argument("--out", default="results/views_dump")
    main(p.parse_args())
