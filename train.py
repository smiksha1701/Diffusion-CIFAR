"""
Training script for the CIFAR diffusion model.

Usage:
    # CIFAR-10 (default)
    python train.py

    # CIFAR-100
    python train.py --dataset cifar100

    # Resume from a checkpoint
    python train.py --resume checkpoints/step_100000.pt
"""

import argparse
import copy
import os
import time

import torch
import torch.optim as optim
from torchvision.utils import save_image

from config import Config
from dataset import get_loaders
from diffusion import GaussianDiffusion
from model import UNet


# ─── EMA ──────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.copy_(self.decay * s + (1 - self.decay) * m)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


# ─── LR warmup ────────────────────────────────────────────────────────────────

def warmup_lambda(step: int, warmup_steps: int):
    return min(1.0, step / max(warmup_steps, 1))


# ─── Main ─────────────────────────────────────────────────────────────────────

def train(cfg: Config, resume: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Dataset: {cfg.dataset}  |  Classes: {cfg.num_classes}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.sample_dir, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet(
        image_size=cfg.image_size,
        in_channels=cfg.channels,
        model_channels=cfg.model_channels,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=cfg.dropout,
        num_classes=cfg.num_classes if cfg.class_cond else None,
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        timesteps=cfg.timesteps,
        noise_schedule=cfg.noise_schedule,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
    ).to(device)

    ema = EMA(model, cfg.ema_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: warmup_lambda(s, cfg.warmup_steps)
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, _ = get_loaders(cfg.dataset, cfg.batch_size, cfg.num_workers)

    # ── Resume ────────────────────────────────────────────────────────────────
    step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]
        print(f"Resumed from step {step}")

    # ── Compile (optional speedup on PyTorch >= 2.0) ──────────────────────────
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile()")
    except Exception:
        pass

    # ── Fixed eval labels (one of each class, tiled to fill the grid) ─────────
    if cfg.class_cond:
        n = cfg.sample_grid
        eval_labels = torch.arange(cfg.num_classes, device=device).repeat(
            n // cfg.num_classes + 1
        )[:n]
    else:
        eval_labels = None

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    data_iter = iter(train_loader)
    t0 = time.time()
    loss_accum = 0.0

    while step < cfg.total_steps:
        try:
            x0, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x0, y = next(data_iter)

        x0 = x0.to(device)
        y  = y.to(device) if cfg.class_cond else None

        loss = diffusion.loss(x0, y, cfg_dropout=cfg.cfg_dropout if cfg.class_cond else 0.0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        loss_accum += loss.item()
        step += 1

        if step % cfg.log_every == 0:
            avg_loss = loss_accum / cfg.log_every
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Step {step:7d}/{cfg.total_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr_now:.2e} | "
                f"{elapsed:.1f}s"
            )
            loss_accum = 0.0
            t0 = time.time()

        if step % cfg.sample_every == 0:
            _save_samples(diffusion, ema.shadow, cfg, eval_labels, step, device)

        if step % cfg.save_every == 0:
            _save_checkpoint(
                cfg, step, model, ema, optimizer, scheduler
            )

    print("Training complete.")
    _save_checkpoint(cfg, step, model, ema, optimizer, scheduler)


def _save_samples(diffusion, ema_model, cfg, eval_labels, step, device):
    ema_model.eval()
    shape = (cfg.sample_grid, cfg.channels, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        if cfg.sampler == "ddim":
            samples = diffusion.ddim_sample(
                shape,
                y=eval_labels,
                cfg_scale=cfg.cfg_scale if cfg.class_cond else 1.0,
                num_steps=cfg.ddim_steps,
                eta=cfg.ddim_eta,
                device=device,
            )
        else:
            samples = diffusion.ddpm_sample(
                shape,
                y=eval_labels,
                cfg_scale=cfg.cfg_scale if cfg.class_cond else 1.0,
                device=device,
            )
    samples = (samples.clamp(-1, 1) + 1) / 2  # → [0, 1]
    path = os.path.join(cfg.sample_dir, f"step_{step:07d}.png")
    save_image(samples, path, nrow=int(cfg.sample_grid ** 0.5))
    print(f"  Saved samples → {path}")
    ema_model.train()


def _save_checkpoint(cfg, step, model, ema, optimizer, scheduler):
    path = os.path.join(cfg.checkpoint_dir, f"step_{step:07d}.pt")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg.__dict__,
        },
        path,
    )
    print(f"  Saved checkpoint → {path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, choices=["cifar10", "cifar100"])
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.dataset:     cfg.dataset = args.dataset; cfg.__post_init__()
    if args.batch_size:  cfg.batch_size = args.batch_size
    if args.lr:          cfg.lr = args.lr
    if args.total_steps: cfg.total_steps = args.total_steps
    if args.cfg_scale:   cfg.cfg_scale = args.cfg_scale

    train(cfg, resume=args.resume)
