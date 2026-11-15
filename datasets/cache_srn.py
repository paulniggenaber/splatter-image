"""
Fill the SRN disk cache: one .pt file per object (see SRNDataset.load_example_id).

    export SHAPENET_DATASET_ROOT=$HOME/data/SRN
    export SRN_CACHE_DIR=$HOME/data/SRN_cache
    python -m datasets.cache_srn --category cars --splits train val --workers 8

Add --first N to cache only the first N objects of each split.
Safe to run again: objects that are already cached are skipped.
"""
import argparse
import os
import time
from multiprocessing import Process

import torch
from hydra import compose, initialize


def run_shard(category, split, first, shard, n_shards):
    torch.set_num_threads(1)
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="default_config", overrides=["+dataset={}".format(category)])
    from datasets.dataset_factory import get_dataset
    ds = get_dataset(cfg, split)
    n = len(ds) if first is None else min(first, len(ds))
    done = 0
    for i in range(shard, n, n_shards):
        example_id = ds.get_example_id(i)
        if os.path.isfile(ds._cache_path(example_id)):
            continue
        ds.load_example_id(example_id, ds.intrins[i])
        for cache in [ds.all_rgbs, ds.all_world_view_transforms, ds.all_view_to_world_transforms,
                      ds.all_full_proj_transforms, ds.all_camera_centers]:
            del cache[example_id]          # keep the RAM use flat
        done += 1
    print("split {} shard {}/{}: cached {} new objects".format(split, shard, n_shards, done), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="cars", choices=["cars", "chairs"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--first", type=int, default=None)
    args = parser.parse_args()
    assert os.environ.get("SRN_CACHE_DIR"), "Set SRN_CACHE_DIR first"

    for split in args.splits:
        start = time.time()
        procs = [Process(target=run_shard, args=(args.category, split, args.first, s, args.workers))
                 for s in range(args.workers)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        print("split {} done in {:.0f} s".format(split, time.time() - start), flush=True)
