import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Standard channel LayerNorm. Mean IS subtracted: the bias-free variant
    amplifies by up to ~300x on near-black pixels, which these images are full of."""

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        mu = xf.mean(1, keepdim=True)
        var = xf.var(1, keepdim=True, unbiased=False)
        out = (xf - mu) * torch.rsqrt(var + self.eps)
        return out.to(x.dtype) * self.weight + self.bias


class MDTA(nn.Module):
    """Attention over channels, not pixels -> cost is linear in H*W."""

    def __init__(self, c, num_heads=1):
        super().__init__()
        while c % num_heads != 0:            # keep head count legal for any width
            num_heads -= 1
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3, bias=False)
        self.project_out = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = F.normalize(q.reshape(b, self.num_heads, -1, h * w), dim=-1)
        k = F.normalize(k.reshape(b, self.num_heads, -1, h * w), dim=-1)
        v = v.reshape(b, self.num_heads, -1, h * w)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        return self.project_out((attn @ v).reshape(b, c, h, w))


class GDFN(nn.Module):
    def __init__(self, c, expansion=2.66):
        super().__init__()
        hid = int(c * expansion)
        self.project_in = nn.Conv2d(c, hid * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hid * 2, hid * 2, 3, padding=1, groups=hid * 2, bias=False)
        self.project_out = nn.Conv2d(hid, c, 1, bias=False)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, c, num_heads=1):
        super().__init__()
        self.norm1, self.mdta = LayerNorm2d(c), MDTA(c, num_heads)
        self.norm2, self.gdfn = LayerNorm2d(c), GDFN(c)

    def forward(self, x):
        x = x + self.mdta(self.norm1(x))
        return x + self.gdfn(self.norm2(x))


# --------------------------------------------------------------------------
# Noise-level estimation, used as a conditioning channel (FFDNet-style).
# The measured speckle level varies ~1.8x across images (p10 0.17 -> p90 0.31),
# so a single model without this must learn one compromise strength for all.
# log() turns the multiplicative speckle additive; the Laplacian kills locally
# planar signal; MAD is a texture-robust sigma.
# --------------------------------------------------------------------------
_LAP = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]]).view(1, 1, 3, 3)


def estimate_noise(x, eps=1e-3):
    """x: (B,1,H,W) -> (B,1,1,1) relative speckle level. No gradient needed."""
    with torch.no_grad():
        z = torch.log(x.detach().float().clamp(min=eps))
        r = F.conv2d(z, _LAP.to(z.device, z.dtype))
        rf = r.reshape(r.shape[0], -1)
        med = rf.median(dim=1, keepdim=True).values
        mad = (rf - med).abs().median(dim=1, keepdim=True).values
        return (mad / 0.6745 / 6.0).view(-1, 1, 1, 1)


# --------------------------------------------------------------------------
# Global skip. Phase 1 used a fixed bicubic upsample of the input, because the
# answer was close to bicubic. On SEM data a bicubic upsample of a *clean*
# downsample caps at ~24.7 dB, so the network now has to move a long way from
# it, and whether the fixed skip helps or anchors the output is an open
# question. "learned" starts numerically identical to "bicubic" and is free to
# drift, which makes the three modes a fair A/B.
# --------------------------------------------------------------------------
def _cubic_w(t, A=-0.75):
    t = abs(t)
    if t <= 1:
        return (A + 2) * t ** 3 - (A + 3) * t ** 2 + 1
    if t < 2:
        return A * t ** 3 - 5 * A * t ** 2 + 8 * A * t - 4 * A
    return 0.0


def bicubic_ps_weight(scale=2, ksize=5):
    """Conv2d(1, scale**2, ksize) weights that reproduce bicubic upsampling
    exactly once followed by PixelShuffle(scale). Verified to 3e-7."""
    r = ksize // 2
    offs = list(range(-r, r + 1))
    w = torch.zeros(scale * scale, 1, ksize, ksize)
    for py in range(scale):
        for px in range(scale):
            sy, sx = (py + 0.5) / scale - 0.5, (px + 0.5) / scale - 0.5
            wy = torch.tensor([_cubic_w(sy - o) for o in offs])
            wx = torch.tensor([_cubic_w(sx - o) for o in offs])
            w[py * scale + px, 0] = torch.outer(wy, wx)
    return w


def config_from_state_dict(sd, ck=None, scale=2):
    """Recover the constructor args from the weights themselves.

    Some early checkpoints do not record every option, so trusting the saved
    config silently builds the wrong model. The shapes never lie. `ck` is the
    checkpoint dict, consulted only for log_input/noise_cond -- one extra input
    channel could be either, so the shape alone cannot tell them apart.
    """
    ck = ck or {}
    n_lvl = 2 + max(int(k.split(".")[1]) for k in sd if k.startswith("downs."))
    dim, in_ch = sd["conv_in.weight"].shape[:2]
    nb = [1 + max(int(k.split(".")[2]) for k in sd
                  if k.startswith(f"encoder_blocks.{i}.")) for i in range(n_lvl)]
    heads = [sd[f"encoder_blocks.{i}.0.mdta.temperature"].shape[1] for i in range(n_lvl)]
    w = sd["final_upsample.0.weight"].shape[0] // scale ** 2
    skip = ck.get("skip", "learned" if "skip_up.weight" in sd else "bicubic")
    return dict(dim=dim, num_blocks=tuple(nb), num_heads=tuple(heads),
                wide_head=0 if w == 1 else w, skip=skip,
                log_input=bool(ck.get("log_input", in_ch >= 2)),
                noise_cond=bool(ck.get("noise_cond", in_ch >= 3)))


def load_state_dict_compat(model, sd):
    """Load a checkpoint that may predate a parameter being added.

    Older runs used a bias-free LayerNorm, so their state_dict has no
    `norm*.bias`. Those parameters initialise to zero here, which is exactly
    the bias-free behaviour -- so filling them in is lossless, not a fudge.
    Anything else missing or unexpected is still a hard error.
    """
    res = model.load_state_dict(sd, strict=False)
    allowed = lambda k: k.endswith(("norm1.bias", "norm2.bias"))
    bad_missing = [k for k in res.missing_keys if not allowed(k)]
    if bad_missing or res.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch\n  missing: {bad_missing[:6]}\n"
            f"  unexpected: {list(res.unexpected_keys)[:6]}")
    if res.missing_keys:
        print(f"  note: filled {len(res.missing_keys)} LayerNorm bias tensors with "
              f"zeros (checkpoint predates the bias term; numerically identical)",
              flush=True)
    return model


class MiniRestormer(nn.Module):
    """
    Options (all default to the original behaviour so existing checkpoints load):

      num_heads    int, or per-level tuple e.g. (1,2,4). Free: same params/FLOPs,
                   just splits attention into independent subspaces.
      log_input    add log(x) as an extra input channel. Speckle is multiplicative,
                   so log makes it additive and signal-independent.
      noise_cond   add an estimated-noise-level channel, so the model can adapt
                   its denoising strength per image instead of averaging.
      skip         "bicubic" fixed upsample of the input (Phase 1 default),
                   "learned" the same thing as a conv initialised to bicubic,
                   "none" no global skip at all.
      wide_head    0 = original 1-channel reconstruction head.
                   >0 = keep N channels through the upsample before projecting
                   to 1, removing a 1-channel bottleneck at full resolution.
    """

    def __init__(self, inp_channels=1, out_channels=1, dim=32,
                 num_blocks=(2, 2, 2), num_heads=1, scale=2,
                 log_input=False, noise_cond=False, wide_head=0,
                 skip="bicubic"):
        super().__init__()
        assert skip in ("bicubic", "learned", "none"), f"unknown skip mode: {skip}"
        self.levels = len(num_blocks)
        self.scale = scale
        self.skip = skip
        if skip == "learned":
            self.skip_up = nn.Conv2d(1, scale ** 2, 5, padding=2, bias=False,
                                     padding_mode="replicate")
            with torch.no_grad():
                self.skip_up.weight.copy_(bicubic_ps_weight(scale, 5))
        self.log_input = log_input
        self.noise_cond = noise_cond

        heads = ([num_heads] * self.levels if isinstance(num_heads, int)
                 else list(num_heads))
        assert len(heads) == self.levels

        in_ch = inp_channels + int(log_input) + int(noise_cond)
        self.conv_in = nn.Conv2d(in_ch, dim, 3, padding=1)

        self.encoder_blocks, self.downs = nn.ModuleList(), nn.ModuleList()
        d = dim
        for i in range(self.levels):
            self.encoder_blocks.append(nn.Sequential(
                *[TransformerBlock(d, heads[i]) for _ in range(num_blocks[i])]))
            if i < self.levels - 1:
                self.downs.append(nn.Conv2d(d, d * 2, 4, stride=2, padding=1))
                d *= 2

        self.bottleneck = nn.Sequential(
            *[TransformerBlock(d, heads[-1]) for _ in range(num_blocks[-1])])

        self.ups, self.decoder_blocks = nn.ModuleList(), nn.ModuleList()
        self.decoder_blocks.append(nn.Sequential(
            *[TransformerBlock(d, heads[-1]) for _ in range(num_blocks[-1])]))
        for i in range(self.levels - 2, -1, -1):
            self.ups.append(nn.Sequential(nn.Conv2d(d, d // 2 * 4, 1), nn.PixelShuffle(2)))
            d //= 2
            self.decoder_blocks.append(nn.Sequential(
                *[TransformerBlock(d, heads[i]) for _ in range(num_blocks[i])]))

        self.conv_out = nn.Conv2d(d, d, 3, padding=1)
        if wide_head:
            self.final_upsample = nn.Sequential(
                nn.Conv2d(d, wide_head * scale ** 2, 3, padding=1),
                nn.PixelShuffle(scale),
                nn.Conv2d(wide_head, wide_head, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(wide_head, out_channels, 3, padding=1))
        else:
            self.final_upsample = nn.Sequential(
                nn.Conv2d(d, out_channels * scale ** 2, 3, padding=1),
                nn.PixelShuffle(scale),
                nn.Conv2d(out_channels, out_channels, 3, padding=1))
        nn.init.zeros_(self.final_upsample[-1].bias)

    def forward(self, x):
        xf = x.float()
        if self.skip == "bicubic":
            base = F.interpolate(xf, scale_factor=self.scale, mode="bicubic",
                                 align_corners=False)
        elif self.skip == "learned":
            base = F.pixel_shuffle(self.skip_up(xf), self.scale)
        else:
            base = None

        feats = [x]
        if self.log_input:
            feats.append(torch.log(x.clamp(min=1e-3)))
        if self.noise_cond:
            feats.append(estimate_noise(x).expand_as(x).to(x.dtype))
        curr = self.conv_in(torch.cat(feats, dim=1) if len(feats) > 1 else x)

        enc = []
        for i in range(self.levels):
            curr = self.encoder_blocks[i](curr)
            enc.append(curr)
            if i < self.levels - 1:
                curr = self.downs[i](curr)

        curr = self.bottleneck(curr)
        curr = self.decoder_blocks[0](curr + enc[-1])
        for i in range(1, self.levels):
            curr = self.ups[i - 1](curr)
            curr = self.decoder_blocks[i](curr + enc[-(i + 1)])

        out = self.final_upsample(self.conv_out(curr)).float()
        return out if base is None else out + base
