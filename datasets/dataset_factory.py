# Each dataset module asserts at import time that its data path is set.
# Importing lazily means only the dataset you actually use needs its path.
def get_dataset(cfg, name):
    category = cfg.data.category

    if category in ("cars", "chairs"):
        from .srn import SRNDataset
        return SRNDataset(cfg, name)

    elif category in ("hydrants", "teddybears"):
        from .co3d import CO3DDataset
        return CO3DDataset(cfg, name)

    elif category == "nmr":
        from .nmr import NMRDataset
        return NMRDataset(cfg, name)

    elif category == "objaverse":
        from .objaverse import ObjaverseDataset
        return ObjaverseDataset(cfg, name)

    elif category == "gso":
        from .gso import GSODataset
        return GSODataset(cfg, name)

    else:
        raise ValueError(f"Unknown dataset category: {category}")
