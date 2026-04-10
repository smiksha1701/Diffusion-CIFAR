"""
U-Net denoising model with:
  - Sinusoidal time embedding
  - Optional class-label embedding (classifier-free guidance ready)
  - Residual blocks with GroupNorm
  - Multi-head self-attention at selected resolutions
  - Configurable channel multipliers and depth
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Helpers ──────────────────────────────────────────────────────────────────

def zero_module(module: nn.Module) -> nn.Module:
    """Zero-init a module's weights (used for final conv so residual starts at 0)."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal positional embedding for timesteps.
    t : (B,) integer timesteps
    returns : (B, dim)
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ─── Building blocks ──────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """Projects sinusoidal embedding → dense time vector."""

    def __init__(self, sinusoidal_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sinusoidal_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_embedding(t, self.net[0].in_features))


class ResBlock(nn.Module):
    """
    Residual block: Conv → GroupNorm → SiLU, injecting time (and optional class)
    embeddings via scale-shift on the second normalisation layer.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch * 2))

        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1))

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # time/class conditioning: scale-shift
        scale, shift = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Single-head (or multi-head) self-attention on spatial feature maps."""

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        assert channels % num_heads == 0
        self.norm = nn.GroupNorm(32, channels)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, self.num_heads, C // self.num_heads * 3, H * W)
        q, k, v = qkv.chunk(3, dim=2)               # each (B, heads, C/heads, HW)
        scale = q.shape[2] ** -0.5
        # (B, heads, HW, C/h) @ (B, heads, C/h, HW) → (B, heads, HW, HW)
        attn = torch.softmax(
            torch.einsum("bhdc,bhec->bhde", q.transpose(-2, -1), k.transpose(-2, -1)) * scale,
            dim=-1,
        )
        out = torch.einsum("bhde,bhec->bhdc", attn, v.transpose(-2, -1))  # (B, heads, HW, C/h)
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ─── U-Net ────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net that predicts the noise ε given noisy image xₜ, timestep t,
    and an optional class label y.

    Args:
        image_size          : spatial resolution (32 for CIFAR)
        in_channels         : image channels (3)
        model_channels      : base channel count
        channel_mults       : per-level multiplier, e.g. (1, 2, 2, 2)
        num_res_blocks      : residual blocks per level
        attention_resolutions: set of spatial sizes where attention is applied
        dropout             : dropout probability
        num_classes         : None → unconditional; int → class-conditional
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions=(16,),
        dropout: float = 0.1,
        num_classes: int = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        time_emb_dim = model_channels * 4

        # ── Time embedding ─────────────────────────────────────────────────────
        self.time_embed = TimestepEmbedding(model_channels, time_emb_dim)

        # ── Class embedding (for CFG) ──────────────────────────────────────────
        if num_classes is not None:
            # +1 for "unconditional" null token (index = num_classes)
            self.class_embed = nn.Sequential(
                nn.Embedding(num_classes + 1, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.class_embed = None

        ch = model_channels
        chs = [ch]  # track skip-connection channel counts

        # ── Input projection ───────────────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        # ── Encoder ───────────────────────────────────────────────────────────
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        cur_size = image_size

        for level, mult in enumerate(channel_mults):
            out_ch = model_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(ch, out_ch, time_emb_dim, dropout))
                if cur_size in attention_resolutions:
                    level_blocks.append(AttentionBlock(out_ch, num_heads=max(1, out_ch // 64)))
                ch = out_ch
                chs.append(ch)
            self.down_blocks.append(level_blocks)
            if level < len(channel_mults) - 1:
                self.down_samples.append(Downsample(ch))
                chs.append(ch)
                cur_size //= 2
            else:
                self.down_samples.append(None)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.mid_res1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads=max(1, ch // 64))
        self.mid_res2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = model_channels * mult
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                level_blocks.append(ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout))
                if cur_size in attention_resolutions:
                    level_blocks.append(AttentionBlock(out_ch, num_heads=max(1, out_ch // 64)))
                ch = out_ch
            self.up_blocks.append(level_blocks)
            if level > 0:
                self.up_samples.append(Upsample(ch))
                cur_size *= 2
            else:
                self.up_samples.append(None)

        # ── Output projection ─────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = zero_module(nn.Conv2d(ch, in_channels, 3, padding=1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x : (B, C, H, W)  noisy image
        t : (B,)           integer timesteps
        y : (B,)           class labels (or None)
        returns: (B, C, H, W) predicted noise
        """
        emb = self.time_embed(t)

        if self.class_embed is not None:
            if y is None:
                # fully unconditional forward
                y_null = torch.full((x.shape[0],), self.num_classes, device=x.device)
                emb = emb + self.class_embed(y_null)
            else:
                emb = emb + self.class_embed(y)

        # ── Encoder ──────────────────────────────────────────────────────────
        h = self.input_conv(x)
        skips = [h]

        for level_blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in level_blocks:
                h = block(h, emb) if isinstance(block, ResBlock) else block(h)
                skips.append(h)
            if downsample is not None:
                h = downsample(h)
                skips.append(h)

        # ── Bottleneck ───────────────────────────────────────────────────────
        h = self.mid_res1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, emb)

        # ── Decoder ──────────────────────────────────────────────────────────
        for level_blocks, upsample in zip(self.up_blocks, self.up_samples):
            for block in level_blocks:
                if isinstance(block, ResBlock):
                    h = torch.cat([h, skips.pop()], dim=1)
                    h = block(h, emb)
                else:
                    h = block(h)
            if upsample is not None:
                h = upsample(h)

        return self.out_conv(F.silu(self.out_norm(h)))
