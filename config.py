"""
Central configuration for CIFAR diffusion model.
Change DATASET between 'cifar10' and 'cifar100'.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset: str = "cifar10"          # "cifar10" | "cifar100"
    image_size: int = 32
    channels: int = 3

    # ── U-Net ─────────────────────────────────────────────────────────────────
    model_channels: int = 128         # base channel width
    channel_mults: Tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16,)  # apply attn at these sizes
    dropout: float = 0.1

    # ── Diffusion ─────────────────────────────────────────────────────────────
    timesteps: int = 1000
    noise_schedule: str = "cosine"    # "linear" | "cosine"
    # linear schedule bounds (only used when noise_schedule="linear")
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # ── Classifier-free guidance ──────────────────────────────────────────────
    class_cond: bool = True           # condition on class labels
    cfg_dropout: float = 0.1          # prob of dropping label during training
    cfg_scale: float = 2.0            # guidance scale at inference (1 = no guidance)

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 128
    lr: float = 2e-4
    ema_decay: float = 0.9999
    total_steps: int = 700_000
    warmup_steps: int = 5_000
    grad_clip: float = 1.0
    num_workers: int = 0

    # ── Sampling ──────────────────────────────────────────────────────────────
    sampler: str = "ddim"             # "ddpm" | "ddim"
    ddim_steps: int = 50
    ddim_eta: float = 0.0             # 0 = deterministic DDIM

    # ── Logging / checkpointing ───────────────────────────────────────────────
    log_every: int = 500
    save_every: int = 10_000
    sample_every: int = 10_000
    sample_grid: int = 64             # images in the eval grid
    checkpoint_dir: str = "checkpoints"
    sample_dir: str = "samples"

    # ── Derived (auto-filled) ─────────────────────────────────────────────────
    num_classes: int = field(init=False)

    def __post_init__(self):
        if self.dataset == "cifar10":
            self.num_classes = 10
        elif self.dataset == "cifar100":
            self.num_classes = 100
        else:
            raise ValueError(f"Unknown dataset: {self.dataset!r}")


# One shared instance; override fields before importing elsewhere.
cfg = Config()
