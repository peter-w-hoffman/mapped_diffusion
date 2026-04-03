import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from common import IMG_SIZE, D, device

# 
def base_m11_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])


class PrecondNormalizeMNIST:
    """
    PIL -> tensor in DDPM model space:
      [0,1] -> [-1,1] -> apply M (or identity) -> normalize in preconditioned space
    """
    def __init__(self, M_cpu, mean_cpu, std_cpu, target_std=1.0, eps=1e-6):
        self.to_tensor = transforms.ToTensor()
        self.M = M_cpu.to(dtype=torch.float32)
        self.mean = mean_cpu.to(dtype=torch.float32)
        self.std = std_cpu.to(dtype=torch.float32)
        self.target_std = float(target_std)
        self.eps = float(eps)

    def __call__(self, img):
        x = self.to_tensor(img)                # [1,28,28] in [0,1]
        x = x * 2.0 - 1.0                      # [1,28,28] in [-1,1]
        x_flat = x.view(-1)                    # [784]
        x_pre = self.M @ x_flat                # [784]
        x_norm = (x_pre - self.mean) / (self.std + self.eps)
        x_norm = x_norm * self.target_std
        return x_norm.view(1, IMG_SIZE, IMG_SIZE)


class DDPMPreconditioner:
    """
    Holds:
      - preconditioner matrix M or identity
      - inverse/pseudoinverse
      - mean/std in the preconditioned space
      - apply / undo maps between raw [-1,1] and DDPM model space
    """
    def __init__(
        self,
        mode="identity",          # "identity" or "M"
        m_path="M_784.pt",
        target_std=1.0,
        eps=1e-6,
        device=device,
        M_cpu=None,
        M_inv_cpu=None,
        mean_cpu=None,
        std_cpu=None,
    ):
        self.mode = mode
        self.m_path = m_path
        self.target_std = float(target_std)
        self.eps = float(eps)
        self.device = torch.device(device)

        if M_cpu is None:
            if self.mode == "identity":
                self.M_cpu = torch.eye(D, dtype=torch.float32)
            else:
                self.M_cpu = torch.load(self.m_path, map_location="cpu").to(dtype=torch.float32)
        else:
            self.M_cpu = M_cpu.to(dtype=torch.float32).cpu()

        if M_inv_cpu is None:
            try:
                self.M_inv_cpu = torch.linalg.inv(self.M_cpu)
            except RuntimeError:
                self.M_inv_cpu = torch.linalg.pinv(self.M_cpu)
        else:
            self.M_inv_cpu = M_inv_cpu.to(dtype=torch.float32).cpu()

        self.mean_cpu = torch.tensor(0.0, dtype=torch.float32) if mean_cpu is None else mean_cpu.to(dtype=torch.float32).cpu()
        self.std_cpu = torch.tensor(1.0, dtype=torch.float32) if std_cpu is None else std_cpu.to(dtype=torch.float32).cpu()

        self._sync_to_device()

    def _sync_to_device(self):
        self.M = self.M_cpu.to(self.device)
        self.M_inv = self.M_inv_cpu.to(self.device)
        self.mean = self.mean_cpu.to(self.device)
        self.std = self.std_cpu.to(self.device)

    @torch.no_grad()
    def fit_stats(self, root="./data", batch_size=512):
        """
        Compute GLOBAL scalar mean/std after:
            x_raw in [-1,1] -> Mx
        using MNIST train split, matching your original notebook.
        """
        ds = datasets.MNIST(root=root, train=True, download=True, transform=base_m11_transform())
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        n = 0
        s1 = torch.tensor(0.0, device=self.device)
        s2 = torch.tensor(0.0, device=self.device)

        for x, _ in loader:
            x = x.to(self.device, non_blocking=(self.device.type == "cuda"))
            x_flat = x.view(x.size(0), -1)
            x_pre = x_flat @ self.M.T
            s1 += x_pre.sum()
            s2 += (x_pre * x_pre).sum()
            n += x_pre.numel()

        mean = s1 / n
        var = s2 / n - mean * mean
        std = torch.sqrt(torch.clamp(var, min=self.eps))

        self.mean_cpu = mean.detach().cpu()
        self.std_cpu = std.detach().cpu()
        self._sync_to_device()
        return self.mean_cpu, self.std_cpu

    def make_train_transform(self):
        return PrecondNormalizeMNIST(
            M_cpu=self.M_cpu,
            mean_cpu=self.mean_cpu,
            std_cpu=self.std_cpu,
            target_std=self.target_std,
            eps=self.eps,
        )

    def apply(self, x_raw_m11):
        """
        x_raw_m11: [B,1,28,28] in [-1,1]
        -> DDPM model space
        """
        x_flat = x_raw_m11.view(x_raw_m11.size(0), -1)
        x_pre = x_flat @ self.M.T
        x_norm = (x_pre - self.mean) / (self.std + self.eps)
        x_norm = x_norm * self.target_std
        return x_norm.view(x_raw_m11.size(0), 1, IMG_SIZE, IMG_SIZE)

    def undo(self, x_model):
        """
        x_model: [B,1,28,28] in DDPM model space
        -> original image space [-1,1]
        """
        x_flat = x_model.view(x_model.size(0), -1)
        x_pre = x_flat / self.target_std
        x_pre = x_pre * (self.std + self.eps) + self.mean
        x_raw = x_pre @ self.M_inv.T
        return x_raw.view(x_model.size(0), 1, IMG_SIZE, IMG_SIZE)

    def state_dict(self):
        return {
            "mode": self.mode,
            "m_path": self.m_path,
            "target_std": self.target_std,
            "eps": self.eps,
            "M": self.M_cpu,
            "M_inv": self.M_inv_cpu,
            "mean": self.mean_cpu,
            "std": self.std_cpu,
        }

    @classmethod
    def from_state_dict(cls, state, device=device):
        return cls(
            mode=state["mode"],
            m_path=state.get("m_path", "M_784.pt"),
            target_std=state["target_std"],
            eps=state["eps"],
            device=device,
            M_cpu=state["M"],
            M_inv_cpu=state["M_inv"],
            mean_cpu=state["mean"],
            std_cpu=state["std"],
        )


def get_ddpm_train_loader(precond, root="./data", batch_size=128, train=True, shuffle=True):
    ds = datasets.MNIST(
        root=root,
        train=train,
        download=True,
        transform=precond.make_train_transform(),
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(precond.device.type == "cuda"),
        drop_last=True,
    )
    return ds, loader


class DiffusionSchedule:
    def __init__(self, T=400, beta_start=1e-4, beta_end=0.02, device=device):
        self.T = int(T)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.device = torch.device(device)

        self.betas = torch.linspace(self.beta_start, self.beta_end, self.T, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = torch.cat([torch.ones(1, device=self.device), self.alpha_bar[:-1]], dim=0)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

    def extract(self, a_1d, t):
        return a_1d.gather(0, t).view(-1, 1, 1, 1)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return self.extract(self.sqrt_alpha_bar, t) * x0 + self.extract(self.sqrt_one_minus_alpha_bar, t) * noise

    def state_dict(self):
        return {
            "T": self.T,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
        }

    @classmethod
    def from_state_dict(cls, state, device=device):
        return cls(
            T=state["T"],
            beta_start=state["beta_start"],
            beta_end=state["beta_end"],
            device=device,
        )


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _gn_groups(ch):
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_gn_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)

        self.norm2 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet28(nn.Module):
    """
    Same UNet structure as your notebook.
    """
    def __init__(self, in_ch=1, base=64, temb_dim=256):
        super().__init__()
        self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.d1a = ResBlock(base, base, temb_dim)
        self.d1b = ResBlock(base, base, temb_dim)
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)

        self.d2a = ResBlock(base * 2, base * 2, temb_dim)
        self.d2b = ResBlock(base * 2, base * 2, temb_dim)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)

        self.d3a = ResBlock(base * 4, base * 4, temb_dim)
        self.d3b = ResBlock(base * 4, base * 4, temb_dim)
        self.down3 = nn.Conv2d(base * 4, base * 8, 3, stride=2, padding=1)

        self.m1 = ResBlock(base * 8, base * 8, temb_dim)
        self.m2 = ResBlock(base * 8, base * 8, temb_dim)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 3, stride=2, padding=1)
        self.u3a = ResBlock(base * 8, base * 4, temb_dim)
        self.u3b = ResBlock(base * 4, base * 4, temb_dim)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.u2a = ResBlock(base * 4, base * 2, temb_dim)
        self.u2b = ResBlock(base * 2, base * 2, temb_dim)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.u1a = ResBlock(base * 2, base, temb_dim)
        self.u1b = ResBlock(base, base, temb_dim)

        self.out_norm = nn.GroupNorm(_gn_groups(base), base)
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):
        temb = timestep_embedding(t, self.temb_dim)
        temb = self.time_mlp(temb)

        h = self.in_conv(x)

        h = self.d1a(h, temb)
        h = self.d1b(h, temb)
        skip28 = h
        h = self.down1(h)

        h = self.d2a(h, temb)
        h = self.d2b(h, temb)
        skip14 = h
        h = self.down2(h)

        h = self.d3a(h, temb)
        h = self.d3b(h, temb)
        skip7 = h
        h = self.down3(h)

        h = self.m1(h, temb)
        h = self.m2(h, temb)

        h = self.up3(h)
        h = torch.cat([h, skip7], dim=1)
        h = self.u3a(h, temb)
        h = self.u3b(h, temb)

        h = self.up2(h)
        h = torch.cat([h, skip14], dim=1)
        h = self.u2a(h, temb)
        h = self.u2b(h, temb)

        h = self.up1(h)
        h = torch.cat([h, skip28], dim=1)
        h = self.u1a(h, temb)
        h = self.u1b(h, temb)

        return self.out_conv(F.silu(self.out_norm(h)))


def train_ddpm(model, train_loader, schedule, epochs=2, lr=2e-4, device=device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"[DDPM] Epoch {epoch}/{epochs}", leave=True)
        ema = None

        for x0, _ in pbar:
            x0 = x0.to(device, non_blocking=(device.type == "cuda"))

            t = torch.randint(0, schedule.T, (x0.size(0),), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt = schedule.q_sample(x0, t, noise)

            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            ema = loss.item() if ema is None else (0.95 * ema + 0.05 * loss.item())
            pbar.set_postfix(loss=f"{ema:.4f}")

        print(f"[DDPM] Epoch {epoch}: loss={ema:.4f}")

    return model


def _seeded_randn_like(x, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randn(x.shape, generator=gen, device="cpu", dtype=x.dtype).to(x.device)


@torch.no_grad()
def p_sample_loop(model, schedule, n_samples=25, device=device):
    model.eval()
    x = torch.randn(n_samples, 1, IMG_SIZE, IMG_SIZE, device=device)

    for i in tqdm(reversed(range(schedule.T)), total=schedule.T, desc="[DDPM] Sampling", leave=False):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        eps = model(x, t)

        beta_t = schedule.extract(schedule.betas, t)
        sqrt_1m_ab_t = schedule.extract(schedule.sqrt_one_minus_alpha_bar, t)
        sqrt_recip_a_t = schedule.extract(schedule.sqrt_recip_alphas, t)

        model_mean = sqrt_recip_a_t * (x - beta_t * eps / sqrt_1m_ab_t)

        if i == 0:
            x = model_mean
        else:
            var = schedule.extract(schedule.posterior_variance, t)
            x = model_mean + torch.sqrt(var) * torch.randn_like(x)

    return x


@torch.no_grad()
def reconstruct_from_xt(model, schedule, xt, t_start, reverse_seed=None):
    """
    Reverse starting from a given x_t rather than from pure Gaussian noise.
    """
    model.eval()
    x = xt
    t_start = int(t_start)

    for i in reversed(range(t_start + 1)):
        t = torch.full((x.size(0),), i, device=x.device, dtype=torch.long)
        eps = model(x, t)

        beta_t = schedule.extract(schedule.betas, t)
        sqrt_1m_ab_t = schedule.extract(schedule.sqrt_one_minus_alpha_bar, t)
        sqrt_recip_a_t = schedule.extract(schedule.sqrt_recip_alphas, t)

        model_mean = sqrt_recip_a_t * (x - beta_t * eps / sqrt_1m_ab_t)

        if i == 0:
            x = model_mean
        else:
            var = schedule.extract(schedule.posterior_variance, t)
            z = torch.randn_like(x) if reverse_seed is None else _seeded_randn_like(x, reverse_seed + i)
            x = model_mean + torch.sqrt(var) * z

    return x


@torch.no_grad()
def forward_backward_reconstruct(model, schedule, precond, x_raw_m11, t_start, forward_seed=None, reverse_seed=None):
    """
    Full experiment primitive:
        raw image x0 in [-1,1]
        -> map into DDPM model space
        -> forward to time t
        -> reverse back to time 0
        -> map back to raw image space [-1,1]
    """
    x0_model = precond.apply(x_raw_m11)
    t = torch.full((x0_model.size(0),), int(t_start), device=x0_model.device, dtype=torch.long)
    noise = torch.randn_like(x0_model) if forward_seed is None else _seeded_randn_like(x0_model, forward_seed)
    xt = schedule.q_sample(x0_model, t, noise)
    x_rec_model = reconstruct_from_xt(model, schedule, xt, t_start=int(t_start), reverse_seed=reverse_seed)
    x_rec_raw = precond.undo(x_rec_model).clamp(-1, 1)

    return {
        "x0_model": x0_model,
        "xt": xt,
        "x_rec_model": x_rec_model,
        "x_rec_raw": x_rec_raw,
        "forward_noise": noise,
    }


@torch.no_grad()
def sample_ddpm_raw(model, schedule, precond, n_samples=25, device=device):
    """
    Unconditional DDPM sampling, returned in original raw image space [-1,1] .
    """
    x_pre = p_sample_loop(model, schedule, n_samples=n_samples, device=device)
    x_raw = precond.undo(x_pre).clamp(-1, 1)
    return x_raw


def save_ddpm_checkpoint(path, model, precond, schedule, model_kwargs=None):
    ckpt = {
        "model_state": model.state_dict(),
        "model_kwargs": {"in_ch": 1, "base": 64, "temb_dim": 256} if model_kwargs is None else model_kwargs,
        "precond_state": precond.state_dict(),
        "schedule_state": schedule.state_dict(),
    }
    torch.save(ckpt, path)


def load_ddpm_checkpoint(path, map_location="cpu", device=device):
    ckpt = torch.load(path, map_location=map_location)
    model_kwargs = ckpt.get("model_kwargs", {"in_ch": 1, "base": 64, "temb_dim": 256})

    model = UNet28(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    precond = DDPMPreconditioner.from_state_dict(ckpt["precond_state"], device=device)
    schedule = DiffusionSchedule.from_state_dict(ckpt["schedule_state"], device=device)
    return model, precond, schedule, ckpt