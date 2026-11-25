# entirely new code file from Paul
"""
Stand-alone mirror-plane predictor: one photo in, the object's mirror plane out.

It is trained on its own (train_plane.py) and has no connection to the Splatter
Image U-Net, so its loss can never disturb the reconstruction.

Plane convention (same as scene/symmetry.py): all points p with n . p = d, in the
frame of the input camera.

Two facts about SRN make this easy to learn:
  * the object centre is at (0, 0, r) in the input camera frame (r = 1.3, checked
    on the data to 1e-6), and the mirror plane passes through the object centre,
    so d = r * n_z. Only the normal n (2 degrees of freedom) has to be predicted.
  * n and -n describe the same plane. To avoid a sign jump the network predicts
    the symmetric matrix n n^T (6 numbers), which has no sign. At test time n is
    the top eigenvector of the predicted matrix.
"""
import torch
import torch.nn as nn

# order of the 6 numbers: xx yy zz xy xz yz
_IDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
LOSS_WEIGHTS = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)     # off-diagonals count twice (Frobenius norm)


def matrix_target(n):
    """n (B, 3) unit normals -> (B, 6) the numbers of n n^T."""
    return torch.stack([n[:, i] * n[:, j] for i, j in _IDX], dim=1)


def matrix_loss(raw, n):
    w = torch.tensor(LOSS_WEIGHTS, device=raw.device, dtype=raw.dtype)
    return (((raw - matrix_target(n)) ** 2) * w).sum(dim=1).mean()


@torch.no_grad()
def normal_from_raw(raw):
    """raw (B, 6) -> unit normal (B, 3): top eigenvector of the predicted matrix,
    sign chosen so that n_z >= 0."""
    B = raw.shape[0]
    M = raw.new_zeros(B, 3, 3)
    for k, (i, j) in enumerate(_IDX):
        M[:, i, j] = raw[:, k]
        M[:, j, i] = raw[:, k]
    _, vecs = torch.linalg.eigh(M.float())
    n = vecs[:, :, -1]                                   # eigenvalues ascending: last = largest
    sign = torch.where(n[:, 2:3] >= 0, torch.ones_like(n[:, 2:3]), -torch.ones_like(n[:, 2:3]))
    return n * sign


def plane_from_normal(n, radius):
    """n (B, 3) -> plane (B, 4) = (n, d) with d = radius * n_z."""
    return torch.cat([n, radius * n[:, 2:3]], dim=1)


class PlanePredictor(nn.Module):
    def __init__(self, arch="resnet18", pretrained=True, radius=1.3):
        super().__init__()
        self.arch = arch
        self.radius = radius
        if arch == "resnet18":
            import torchvision
            net = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            net.fc = nn.Linear(512, 6)
            nn.init.normal_(net.fc.weight, std=0.01)
            nn.init.zeros_(net.fc.bias)
            self.net = net
        elif arch == "small":
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                nn.Linear(128 * 16, 256), nn.SiLU(), nn.Linear(256, 6))
        else:
            raise ValueError(arch)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def raw(self, img):
        """img (B, 3, H, W) in [0, 1] -> (B, 6) numbers of the predicted n n^T."""
        return self.net((img - self.mean) / self.std)

    @torch.no_grad()
    def forward(self, img):
        """img (B, 3, H, W) in [0, 1] -> plane (B, 4), for use at test time."""
        return plane_from_normal(normal_from_raw(self.raw(img)), self.radius)


def save_predictor(model, path, extra=None):
    torch.save({"arch": model.arch, "radius": model.radius,
                "state_dict": model.state_dict(), **(extra or {})}, path)


def load_predictor(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = PlanePredictor(arch=ckpt["arch"], pretrained=False, radius=ckpt["radius"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
