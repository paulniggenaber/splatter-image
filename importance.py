"""
Gaussian importance scores, selection criteria and budget-aware masking for Splatter Image.

Put this file in the repo root (next to eval.py). Everything works on ONE object:
`pc` is the per-object dict   {k: v[0].contiguous() for k, v in reconstruction.items()}
with N = 128*128 = 16384 pixel-aligned Gaussians in row-major order (index = y*W + x):
    xyz (N,3)  opacity (N,1)  scaling (N,3)  rotation (N,4)  features_dc (N,1,3)  features_rest (N,3,3)

Key trick (no CUDA changes needed): if every Gaussian is rendered with colour 1, then
    C_p = sum_i w_{i,p} * 1 + T_p * bg   =>   d(sum_p C_p) / d colour_i = sum_p w_{i,p}
so ONE backward pass through the stock rasteriser gives the exact per-Gaussian contribution
(alpha * G * transmittance, summed over pixels) for that view.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



# ----------------------------------------------------------------------------- helpers
def matrix_to_quaternion(R):
    """Batched rotation matrix (...,3,3) -> quaternion (w,x,y,z), numerically stable branch choice
    (the repo's utils.general_utils.matrix_to_quaternion only handles a single matrix)."""
    m = R.reshape(-1, 3, 3)
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    cand = torch.stack([
        torch.stack([1 + m00 + m11 + m22, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], -1),
        torch.stack([m[:, 2, 1] - m[:, 1, 2], 1 + m00 - m11 - m22, m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0]], -1),
        torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 0, 1] + m[:, 1, 0], 1 - m00 + m11 - m22, m[:, 1, 2] + m[:, 2, 1]], -1),
        torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1], 1 - m00 - m11 + m22], -1),
    ], 1)                                                     # (B, 4 candidates, 4)
    diag = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    best = diag.argmax(-1)
    q = cand[torch.arange(m.shape[0]), best]
    q = q / q.norm(dim=-1, keepdim=True)
    return q.reshape(*R.shape[:-2], 4)


def build_rotation(q):
    """Quaternion (w,x,y,z) -> rotation matrix. Same convention as utils.general_utils.build_rotation
    (which is hard-coded to device='cuda'); this one works on any device."""
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def subset(pc, idx):
    """Keep only Gaussians `idx` (LongTensor). Works for every key of the dict."""
    return {k: v[idx].contiguous() for k, v in pc.items()}


def detached(pc):
    return {k: v.detach() for k, v in pc.items()}


def _render(pc, cam, bg, cfg, override_color=None):
    from gaussian_renderer import render_predicted  # lazy: CPU tests don't need CUDA
    wvt, fpt, cc = cam
    return render_predicted(pc, wvt, fpt, cc, bg, cfg, override_color=override_color)["render"]


def cams_from_data(data, idxs):
    """List of (world_view, full_proj, camera_center) for view indices `idxs` of object 0."""
    return [(data["world_view_transforms"][0, i],
             data["full_proj_transforms"][0, i],
             data["camera_centers"][0, i]) for i in idxs]


# ----------------------------------------------------------------------------- oracle scores
def contribution_scores(pc, cams, cfg):
    """Multi-view contribution  sum_v sum_p alpha_i G_i(p) T_{i,p,v}.  Needs NO ground truth."""
    pc = detached(pc)
    N, dev = pc["xyz"].shape[0], pc["xyz"].device
    bg = torch.zeros(3, device=dev)
    score = torch.zeros(N, device=dev)
    with torch.enable_grad():
        for cam in cams:
            ones = torch.ones(N, 3, device=dev, requires_grad=True)
            img = _render(pc, cam, bg, cfg, override_color=ones)
            (g,) = torch.autograd.grad(img[0].sum(), ones)
            score += g[:, 0]
    return score


def error_sensitivity_scores(pc, cams, gts, bg, cfg, signed=False):
    """
    SNIP / Optimal-Brain-Damage style saliency of each Gaussian's opacity:
        abs:    sum_v | alpha_i * dL_v/dalpha_i |
        signed: sum_v  -alpha_i * dL_v/dalpha_i   (first-order estimate of the loss INCREASE
                                                    if Gaussian i is removed; can be negative)
    L_v = MSE(render_v, gt_v). Needs ground-truth images -> only use views disjoint from eval views.
    """
    pc = detached(pc)
    score = torch.zeros(pc["xyz"].shape[0], device=pc["xyz"].device)
    with torch.enable_grad():
        op = pc["opacity"].clone().requires_grad_(True)
        pc_g = dict(pc, opacity=op)
        for cam, gt in zip(cams, gts):
            img = _render(pc_g, cam, bg, cfg)
            (g,) = torch.autograd.grad(F.mse_loss(img, gt), op)
            s = (op * g).squeeze(-1).detach()
            score += (-s) if signed else s.abs()
    return score


# ----------------------------------------------------------------------------- cheap criteria
def sobel_edges(img):
    """img (3,H,W) in [0,1] -> edge magnitude (H*W,) normalised to [0,1]."""
    gray = img.mean(0, keepdim=True)[None]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=img.device)[None, None]
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, kx.transpose(-1, -2), padding=1)
    e = (gx ** 2 + gy ** 2).sqrt().flatten()
    return e / (e.max() + 1e-8)


def cheap_scores(pc, criterion, input_image=None, lam_edge=0.5, seed=0):
    op = pc["opacity"].squeeze(-1)
    if criterion == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.rand(op.shape[0], generator=g).to(op.device)
    if criterion == "opacity":
        return op
    if criterion == "scale":          # ellipsoid volume
        return pc["scaling"].prod(-1)
    if criterion == "opacity_area":   # alpha * area of the two largest axes ("how much it covers")
        s = pc["scaling"].sort(-1, descending=True).values
        return op * s[:, 0] * s[:, 1]
    if criterion == "edge_opacity":   # PhGS-style prior (deterministic top-k instead of sampling)
        assert input_image is not None
        return lam_edge * sobel_edges(input_image) + (1 - lam_edge) * op
    raise ValueError(criterion)


def topk_indices(score, k):
    return torch.topk(score, k, sorted=False).indices


# ----------------------------------------------------------------------------- ToMe-style merging
def merge_blocks(pc, H=128, W=128, b=2):
    """
    Replace every b x b block of pixel-aligned Gaussians by ONE Gaussian (N -> N / b^2):
      weights  w_j = alpha_j / sum alpha      (uniform if the block is fully transparent)
      mean     mu  = sum w_j mu_j
      cov      S   = sum w_j (S_j + (mu_j - mu)(mu_j - mu)^T)      (moment matching)
      opacity  a   = 1 - prod (1 - alpha_j)
      colour   SH  = sum w_j SH_j
    """
    N = pc["xyz"].shape[0]
    assert N == H * W, "merge_blocks expects one Gaussian per pixel"

    def blocks(t):  # (N, ...) -> (N/b^2, b^2, ...)
        rest = t.shape[1:]
        t = t.reshape(H // b, b, W // b, b, *rest).transpose(1, 2)
        return t.reshape((H // b) * (W // b), b * b, *rest)

    a = blocks(pc["opacity"].squeeze(-1))                       # (M, b2)
    asum = a.sum(1, keepdim=True)
    w = torch.where(asum > 1e-12, a / asum.clamp_min(1e-12), torch.full_like(a, 1.0 / a.shape[1]))

    mu_j = blocks(pc["xyz"])                                    # (M, b2, 3)
    mu = (w[..., None] * mu_j).sum(1)                           # (M, 3)

    R = build_rotation(pc["rotation"])                          # (N, 3, 3), normalises quats
    cov = R @ torch.diag_embed(pc["scaling"] ** 2) @ R.transpose(1, 2)
    cov_j = blocks(cov)                                         # (M, b2, 3, 3)
    d = mu_j - mu[:, None]
    cov_m = (w[..., None, None] * (cov_j + d[..., :, None] * d[..., None, :])).sum(1)

    evals, evecs = torch.linalg.eigh(cov_m)                     # columns = principal axes
    flip = torch.where(torch.det(evecs) < 0, -1.0, 1.0).to(evecs.dtype)
    evecs = evecs * flip[:, None, None]                         # proper rotation (det = +1)
    out = {
        "xyz": mu,
        "opacity": (1 - (1 - a).prod(1)).unsqueeze(-1),
        "scaling": evals.clamp_min(1e-12).sqrt(),
        "rotation": matrix_to_quaternion(evecs),
    }
    for k in pc:
        if k.startswith("features"):
            f = blocks(pc[k])
            out[k] = (w.reshape(*w.shape, *([1] * (f.dim() - 2))) * f).sum(1)
    return {k: v.contiguous() for k, v in out.items()}


# ----------------------------------------------------------------------------- Level 2 / 3
INPUT_CHANNELS = {"all": 11, "rgb": 3, "gauss": 8}   # ablation: what does the head need?


class ImportanceHead(nn.Module):
    """
    Tiny CNN: (input RGB and/or predicted Gaussian parameter maps) -> one logit per pixel/Gaussian.
    inputs="all":   rgb 3 + opacity 1 + log-scale 3 + distance |xyz| 1 + colour (SH dc) 3 = 11 ch
    inputs="rgb":   only the input image (3 ch)       -> "is importance predictable from the image?"
    inputs="gauss": only the predicted Gaussians (8 ch) -> "is the image needed at all?"
    Dilated convs give a ~30 px receptive field at ~80k parameters.
    """
    def __init__(self, in_ch=11, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )

    @staticmethod
    def make_input(rgb, recon, H=128, W=128, inputs="all"):
        """rgb (B,3,H,W); recon = batched model output dict (B,N,...) -> (B,C,H,W), C = INPUT_CHANNELS[inputs]."""
        B = rgb.shape[0]

        def to_map(t):  # (B,N,C) -> (B,C,H,W)
            return t.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        gauss = [to_map(recon["opacity"]),
                 to_map(recon["scaling"].clamp_min(1e-8).log()),
                 to_map(recon["xyz"].norm(dim=-1, keepdim=True)),
                 to_map(recon["features_dc"].reshape(B, H * W, 3))]
        feats = {"all": [rgb] + gauss, "rgb": [rgb], "gauss": gauss}[inputs]
        return torch.cat(feats, 1).detach()

    def forward(self, x):
        return self.net(x).flatten(1)  # (B, N)


def ranking_loss(logits, target, n_pairs=8192):
    """Pairwise logistic ranking loss: only the ORDER matters for top-k pruning. logits/target (B,N)."""
    B, N = logits.shape
    i = torch.randint(0, N, (B, n_pairs), device=logits.device)
    j = torch.randint(0, N, (B, n_pairs), device=logits.device)
    li, lj = logits.gather(1, i), logits.gather(1, j)
    sign = torch.sign(target.gather(1, i) - target.gather(1, j))
    keep = sign != 0
    return F.softplus(-sign * (li - lj))[keep].mean()


def head_loss(logits, target, kind="rank"):
    """kind="rank": pairwise ranking (order only); kind="mse": regression on the log-importance target."""
    if kind == "rank":
        return ranking_loss(logits, target)
    if kind == "mse":
        return F.mse_loss(logits, target)
    raise ValueError(kind)


def save_head(head, path, **meta):
    torch.save({"state_dict": head.state_dict(), **meta}, path)


def load_head(path, device):
    """Loads a head saved by save_head (or an old plain state_dict = inputs 'all')."""
    ckpt = torch.load(path, map_location=device)
    if "state_dict" in ckpt:
        state, inputs = ckpt["state_dict"], ckpt.get("inputs", "all")
    else:
        state, inputs = ckpt, "all"
    head = ImportanceHead(in_ch=INPUT_CHANNELS[inputs]).to(device)
    head.load_state_dict(state)
    return head.eval(), inputs


def straight_through_topk_mask(logits, keep_frac, tau=1.0):
    """
    Forward: exact hard top-k mask (training sees exactly the budget used at test time).
    Backward: gradient of sigmoid(logits / tau) (straight-through estimator).
    Multiply opacity by the mask: masked Gaussians get alpha = 0 and the rasteriser skips them.
    """
    B, N = logits.shape
    k = max(1, int(round(keep_frac * N)))
    soft = torch.sigmoid(logits / tau)
    hard = torch.zeros_like(soft).scatter_(1, torch.topk(logits, k, dim=1).indices, 1.0)
    return hard + soft - soft.detach()


def storage_bytes(n_gaussians, sh_degree=1, bytes_per_float=4):
    """xyz 3 + scale 3 + rot 4 + opacity 1 + SH 3*(deg+1)^2 floats  (= 23 floats for cars)."""
    return n_gaussians * (11 + 3 * (sh_degree + 1) ** 2) * bytes_per_float
