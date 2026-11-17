"""
Extract part of an SRN zip (srn_chairs.zip) into the folder layout the dataset class
expects:  <root>/srn_chairs/chairs_{train,val,test}/<object id>/{rgb,pose,intrinsics.txt}.

Only the first N objects (sorted by id, the order the dataset uses) of each split are
extracted, because the full zip is 2 million small files.

    python -m datasets.extract_srn_zip ~/data/SRN/srn_chairs.zip ~/data/SRN --category chairs \
        --train 2500 --val 100 --test 100 --workers 4
"""
import argparse
import os
import re
import zipfile
from multiprocessing import Pool


def split_and_id(name, category):
    # chairs_train/chairs_2.0_train/<id>/rgb/000000.png   (train has an extra level)
    # chairs_val/<id>/rgb/000000.png
    parts = name.split("/")
    if len(parts) < 3 or not parts[0].startswith(category + "_"):
        return None
    split = parts[0][len(category) + 1:]
    rest = parts[1:]
    if re.fullmatch(category + r"_2\.0_" + split, rest[0]):
        rest = rest[1:]
    if len(rest) < 2 or not re.fullmatch(r"[0-9a-f]{32}", rest[0]):
        return None
    return split, rest[0], "/".join(rest[1:])


def extract_chunk(args):
    zip_path, dest, items = args
    with zipfile.ZipFile(zip_path) as z:
        for member, out_rel in items:
            out = os.path.join(dest, out_rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(member) as src, open(out, "wb") as dst:
                dst.write(src.read())
    return len(items)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("root")
    ap.add_argument("--category", default="chairs")
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--val", type=int, default=0)
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    want = {"train": a.train, "val": a.val, "test": a.test}

    with zipfile.ZipFile(a.zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
    by_split = {}
    for n in names:
        r = split_and_id(n, a.category)
        if r is not None:
            by_split.setdefault(r[0], {}).setdefault(r[1], []).append((n, r[2]))

    items = []
    for split, k in want.items():
        ids = sorted(by_split.get(split, {}).keys())[:k]
        print("split {}: {} objects in the zip, extracting the first {}".format(split, len(by_split.get(split, {})), len(ids)))
        for oid in ids:
            for member, rel in by_split[split][oid]:
                items.append((member, os.path.join("srn_{}".format(a.category),
                                                   "{}_{}".format(a.category, split), oid, rel)))
    chunks = [(a.zip_path, a.root, items[i::a.workers * 8]) for i in range(a.workers * 8)]
    with Pool(a.workers) as p:
        done = sum(p.imap_unordered(extract_chunk, chunks))
    print("extracted {} files".format(done))
