"""
Standalone sampling script — loads a checkpoint and generates images.

Usage:
    # Generate 64 images with DDIM (class-conditional)
    python sample.py --checkpoint checkpoints/step_0700000.pt --n 64

    # Specific class (e.g. class 3)
    python sample.py --checkpoint checkpoints/step_0700000.pt --class_id 3 --n 64

    # CIFAR-100, unconditional DDPM
    python sample.py --checkpoint checkpoints/step_0700000.pt --sampler ddpm --cfg_scale 1.0

    # All outputs are saved to --out_path (default: samples/final.png)
"""

import argparse
import os
import torch
from torchvision.utils import save_image

from config import Config
from diffusion import GaussianDiffusion
from model import UNet


def load_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_cfg = ckpt.get("config", {})

    # Re-create config from saved values so architecture matches exactly
    cfg = Config()
    for k, v in saved_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.__post_init__()

    model = UNet(
        image_size=cfg.image_size,
        in_channels=cfg.channels,
        model_channels=cfg.model_channels,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=0.0,  # no dropout at inference
        num_classes=cfg.num_classes if cfg.class_cond else None,
    ).to(device)

    # Prefer EMA weights
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()

    diffusion = GaussianDiffusion(
        model,
        timesteps=cfg.timesteps,
        noise_schedule=cfg.noise_schedule,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
    ).to(device)

    return diffusion, cfg


@torch.no_grad()
def generate(
    diffusion: GaussianDiffusion,
    cfg: Config,
    n: int,
    class_id: int = None,
    sampler: str = None,
    ddim_steps: int = None,
    ddim_eta: float = None,
    cfg_scale: float = None,
    device: torch.device = None,
) -> torch.Tensor:
    sampler    = sampler    or cfg.sampler
    ddim_steps = ddim_steps or cfg.ddim_steps
    ddim_eta   = ddim_eta   if ddim_eta is not None else cfg.ddim_eta
    cfg_scale  = cfg_scale  if cfg_scale is not None else cfg.cfg_scale

    shape = (n, cfg.channels, cfg.image_size, cfg.image_size)

    if cfg.class_cond:
        if class_id is not None:
            y = torch.full((n,), class_id, device=device, dtype=torch.long)
        else:
            # Cycle through all classes evenly
            y = torch.arange(cfg.num_classes, device=device).repeat(
                n // cfg.num_classes + 1
            )[:n]
    else:
        y = None

    if sampler == "ddim":
        samples = diffusion.ddim_sample(
            shape, y=y, cfg_scale=cfg_scale,
            num_steps=ddim_steps, eta=ddim_eta, device=device,
        )
    else:
        samples = diffusion.ddpm_sample(
            shape, y=y, cfg_scale=cfg_scale, device=device,
        )

    return (samples.clamp(-1, 1) + 1) / 2  # → [0, 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n",           type=int,   default=64)
    parser.add_argument("--class_id",    type=int,   default=None)
    parser.add_argument("--sampler",     default=None, choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps",  type=int,   default=None)
    parser.add_argument("--ddim_eta",    type=float, default=None)
    parser.add_argument("--cfg_scale",   type=float, default=None)
    parser.add_argument("--out_path",    default="samples/final.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    diffusion, cfg = load_from_checkpoint(args.checkpoint, device)
    print(f"Loaded: {cfg.dataset} | classes: {cfg.num_classes}")

    imgs = generate(
        diffusion, cfg,
        n=args.n,
        class_id=args.class_id,
        sampler=args.sampler,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        cfg_scale=args.cfg_scale,
        device=device,
    )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    nrow = int(args.n ** 0.5)
    save_image(imgs, args.out_path, nrow=nrow)
    print(f"Saved {args.n} images -> {args.out_path}")
