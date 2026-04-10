"""
DDPM forward / reverse process  +  DDIM sampler.

References:
  DDPM  – Ho et al., 2020  (https://arxiv.org/abs/2006.11239)
  DDIM  – Song et al., 2020 (https://arxiv.org/abs/2010.02502)
  CFG   – Ho & Salimans, 2021 (https://arxiv.org/abs/2207.12598)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ─── Noise schedules ──────────────────────────────────────────────────────────

def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s=0.008):
    """
    Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic
    Models" (Nichol & Dhariwal, 2021).
    """
    t = torch.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999)


# ─── Diffusion ────────────────────────────────────────────────────────────────

class GaussianDiffusion(nn.Module):
    """
    Wraps a denoising U-Net with the full DDPM training + sampling logic.

    Attributes exposed on self (all on the correct device after .to(device)):
        betas, alphas, alphas_cumprod, alphas_cumprod_prev,
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
        posterior_mean_coef1/2, posterior_variance, posterior_log_variance_clipped
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        noise_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.model = model
        self.T = timesteps

        # ── Precompute schedule tensors ───────────────────────────────────────
        if noise_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif noise_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule: {noise_schedule!r}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())

        # q(x_{t-1} | x_t, x_0) posterior coefficients
        self.register_buffer(
            "posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod),
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            posterior_variance.clamp(min=1e-20).log(),
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: tuple):
        """Index arr with t, then broadcast to shape."""
        return arr[t].reshape(t.shape[0], *([1] * (len(shape) - 1)))

    # ── Forward (noising) process ─────────────────────────────────────────────

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Sample xₜ ~ q(xₜ | x₀)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omc = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_omc * noise, noise

    # ── Training loss ─────────────────────────────────────────────────────────

    def loss(
        self,
        x0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        cfg_dropout: float = 0.0,
    ) -> torch.Tensor:
        """
        Simple ε-prediction MSE loss (DDPM Eq. 14).

        For classifier-free guidance training, randomly replace labels with the
        null token (done by passing cfg_dropout > 0).
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)

        # Optionally drop labels for CFG training
        if y is not None and cfg_dropout > 0.0:
            drop_mask = torch.rand(B, device=x0.device) < cfg_dropout
            # The model uses index num_classes as the null token
            y = y.clone()
            y[drop_mask] = self.model.num_classes

        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise)
        pred = self.model(xt, t, y)
        return F.mse_loss(pred, noise)

    # ── DDPM reverse step ─────────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample(
        self,
        xt: torch.Tensor,
        t: int,
        y: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """One DDPM reverse step: sample x_{t-1} ~ p_θ(x_{t-1} | xₜ)."""
        t_batch = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)

        eps = self._predict_eps_cfg(xt, t_batch, y, cfg_scale)

        # Reconstruct x₀ estimate
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t_batch, xt.shape)
        sqrt_omc = self._extract(self.sqrt_one_minus_alphas_cumprod, t_batch, xt.shape)
        x0_pred = (xt - sqrt_omc * eps) / sqrt_ac

        # Posterior mean
        coef1 = self._extract(self.posterior_mean_coef1, t_batch, xt.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t_batch, xt.shape)
        mean = coef1 * x0_pred + coef2 * xt

        if t == 0:
            return mean
        log_var = self._extract(self.posterior_log_variance_clipped, t_batch, xt.shape)
        return mean + (0.5 * log_var).exp() * torch.randn_like(xt)

    @torch.no_grad()
    def ddpm_sample(
        self,
        shape: tuple,
        y: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Full DDPM reverse chain: T steps."""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(x, t, y, cfg_scale)
        return x

    # ── DDIM sampler ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple,
        y: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        num_steps: int = 50,
        eta: float = 0.0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        DDIM sampling with `num_steps` steps and stochasticity η.
        η=0 → deterministic; η=1 → DDPM-equivalent variance.
        """
        # Uniformly spaced sub-sequence of timesteps
        step_seq = np.linspace(0, self.T - 1, num_steps, dtype=int)[::-1]

        x = torch.randn(shape, device=device)

        for i, t_cur in enumerate(step_seq):
            t_prev = step_seq[i + 1] if i + 1 < len(step_seq) else -1

            t_batch = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            eps = self._predict_eps_cfg(x, t_batch, y, cfg_scale)

            ac_cur = self.alphas_cumprod[t_cur]
            ac_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            x0_pred = (x - (1 - ac_cur).sqrt() * eps) / ac_cur.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            sigma = (
                eta
                * ((1 - ac_prev) / (1 - ac_cur)).sqrt()
                * (1 - ac_cur / ac_prev).sqrt()
            )
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

            x = ac_prev.sqrt() * x0_pred + (1 - ac_prev - sigma ** 2).clamp(min=0).sqrt() * eps + sigma * noise

        return x

    # ── CFG helper ────────────────────────────────────────────────────────────

    def _predict_eps_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        Classifier-free guidance: blend conditional and unconditional predictions.
        cfg_scale=1 → conditional only (no guidance boost).
        """
        if y is None or cfg_scale == 1.0:
            return self.model(x, t, y)

        # Unconditional: pass null label token
        y_null = torch.full_like(y, self.model.num_classes)

        # Double the batch for a single forward pass
        x2 = torch.cat([x, x], dim=0)
        t2 = torch.cat([t, t], dim=0)
        y2 = torch.cat([y, y_null], dim=0)

        eps2 = self.model(x2, t2, y2)
        eps_cond, eps_uncond = eps2.chunk(2, dim=0)

        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)
