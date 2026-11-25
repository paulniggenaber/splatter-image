# Lazy imports: each dataset module is imported only when it is used (was: all at the top).
# ======================================================================
# Each dataset module asserts at import time that its data path is set, and the CO3D, NMR,
# Objaverse and GSO loaders need extra packages the SRN experiments do not use. Importing them
# all at the top made every SRN run fail; importing lazily means only the dataset you actually
# use needs its path. Was:
#   from .srn import SRNDataset
#   from .co3d import CO3DDataset
#   from .nmr import NMRDataset
#   from .objaverse import ObjaverseDataset
#   from .gso import GSODataset
# ======================================================================

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

    # Fail loudly on an unknown category (the original silently returned None).
    # ======================================================================
    else:
        raise ValueError(f"Unknown dataset category: {category}")
    # ======================================================================
