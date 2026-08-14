"""Conditional set-diffusion model for synthetic corner kick generation.

Generates per-player (x, y, vx, vy) for a corner setup, conditioned on:
  - per-slot flags (teammate / keeper / actor / velocity-observed), taken from
    a real corner's roster template (the analog of TacticAI adjusting a fixed
    set of players),
  - the corner outcome (shot / no shot) via classifier-free guidance.

Denoiser: permutation-equivariant transformer over player slots (no positional
encoding), DDPM cosine schedule, epsilon prediction, EMA weights, DDIM sampling.

Run:  python diffusion_corners.py            # benchmark gate + train + save
      (aborts with a message if the projected training time is unreasonable)
"""
import math
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SP = os.path.dirname(os.path.abspath(__file__))


def pick_device():
    """Best available torch device: cluster GPU, Apple GPU, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


MAXP = 22
CH = 4                    # generated channels: x, y, vx, vy
FLAGS = 4                 # teammate, keeper, actor, has_vel
T_STEPS = 1000
TRAIN_STEPS = 60000
BATCH = 256
LR = 2e-4
EMA_DECAY = 0.9995
P_UNCOND = 0.15           # classifier-free guidance label dropout
JITTER_M = 0.25           # position augmentation (meters)
TIME_BUDGET_MIN = 90      # abort if projected training exceeds this

# Pitch projection bounds, in NORMALIZED units. normalize() maps the pitch
# rectangle [0,120] x [0,80] yd onto exactly [-1,1] x [-1,1], so the pitch
# boundary is +-1 and the existing +-2.5 guard is 2.5x the pitch in every
# direction -- a numerical safety net, not a physical constraint.
#
# Real freeze frames do put players slightly off the pitch (0.55% of players,
# median 0.32 yd past the line, max 3.2 yd): annotation noise plus players
# genuinely standing off the byline. Clamping hard at +-1 would therefore make
# the samples MORE constrained than reality. The margin below is the real
# out-of-bounds extent, so the projection removes only excursions the real
# data never shows.
#
# The margin is a knob, not a constant: at 3.25 it admits every excursion the
# real data shows (and so almost never binds), at 0.0 it is the strict pitch.
OOB_MARGIN_YD = 3.25
X0_GUARD = 2.5            # legacy numerical clamp, kept for velocities


def pitch_bounds(margin_yd=OOB_MARGIN_YD):
    """Pitch rectangle +- margin, in normalized units."""
    lo = np.array([-1.0 - margin_yd / 60.0, -1.0 - margin_yd / 40.0])
    hi = np.array([+1.0 + margin_yd / 60.0, +1.0 + margin_yd / 40.0])
    return lo, hi


def normalize(pos_vel):
    out = pos_vel.copy()
    out[..., 0] = (out[..., 0] - 60.0) / 60.0
    out[..., 1] = (out[..., 1] - 40.0) / 40.0
    out[..., 2] = out[..., 2] / 5.0
    out[..., 3] = out[..., 3] / 5.0
    return out


def denormalize(z):
    out = z.copy()
    out[..., 0] = out[..., 0] * 60.0 + 60.0
    out[..., 1] = out[..., 1] * 40.0 + 40.0
    out[..., 2] = out[..., 2] * 5.0
    out[..., 3] = out[..., 3] * 5.0
    return out


def load_dataset():
    dc = pd.read_parquet(f"{SP}/corners.parquet")
    dp = pd.read_parquet(f"{SP}/corner_players.parquet")
    ok = dc[(dc.n_att_visible >= 5) & (dc.n_def_visible >= 5)].reset_index(drop=True)
    dp = dp[dp.corner_id.isin(ok.corner_id)]
    shot = ok.set_index("corner_id")["shot"]

    ids, X, F, M, y = [], [], [], [], []
    for cid, g in dp.groupby("corner_id", sort=False):
        g = g.iloc[:MAXP]
        n = len(g)
        x = np.zeros((MAXP, CH), dtype=np.float32)
        f = np.zeros((MAXP, FLAGS), dtype=np.float32)
        m = np.zeros(MAXP, dtype=np.float32)
        x[:n, 0], x[:n, 1] = g.x, g.y
        x[:n, 2], x[:n, 3] = g.vx, g.vy
        f[:n, 0], f[:n, 1] = g.teammate, g.keeper
        f[:n, 2], f[:n, 3] = g.actor, g.has_vel
        m[:n] = 1.0
        ids.append(cid)
        X.append(normalize(x))
        F.append(f)
        M.append(m)
        y.append(int(shot.loc[cid]))
    return (ids, np.stack(X), np.stack(F), np.stack(M),
            np.array(y, dtype=np.int64))


# --------------------------- model ------------------------------------------
class TimeEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        half = self.d // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device) / (half - 1))
        ang = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))


class Denoiser(nn.Module):
    """Permutation-equivariant transformer over player slots."""

    def __init__(self, d=160, layers=5, heads=4):
        super().__init__()
        self.inp = nn.Linear(CH + FLAGS, d)
        self.time = TimeEmbed(d)
        self.label = nn.Embedding(3, d)     # 0 = no shot, 1 = shot, 2 = null
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=2 * d,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.out = nn.Linear(d, CH)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, flags, mask, t, label):
        h = self.inp(torch.cat([x, flags], dim=-1))
        cond = self.time(t) + self.label(label)
        h = h + cond[:, None, :]
        h = self.encoder(h, src_key_padding_mask=(mask < 0.5))
        return self.out(h)


# --------------------------- diffusion utils --------------------------------
def cosine_schedule(T):
    s = 0.008
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 1e-5, 0.999)
    return betas


class Diffusion:
    def __init__(self, device):
        self.device = device
        betas = cosine_schedule(T_STEPS).to(device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def add_noise(self, x0, t, noise):
        ab = self.alpha_bar[t][:, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def project_x0(self, x0, project, margin_yd=OOB_MARGIN_YD):
        """Clamp the predicted clean sample.

        `project=False` reproduces the original behaviour exactly: a symmetric
        +-2.5 guard on all four channels. `project=True` additionally pulls the
        two position channels onto the pitch plus `margin_yd`, leaving the
        velocity channels on the legacy guard.

        Applied at every DDIM step rather than only at the end: x0 is fed back
        into x_{t-1}, so constraining it steers the remaining trajectory
        instead of just truncating the final answer.
        """
        x0 = x0.clamp(-X0_GUARD, X0_GUARD)
        if not project:
            return x0
        lo_np, hi_np = pitch_bounds(margin_yd)
        lo = torch.tensor(lo_np, dtype=x0.dtype, device=x0.device)
        hi = torch.tensor(hi_np, dtype=x0.dtype, device=x0.device)
        pos = torch.max(torch.min(x0[..., :2], hi), lo)
        return torch.cat([pos, x0[..., 2:]], dim=-1)

    @torch.no_grad()
    def ddim_sample(self, model, flags, mask, label, steps=100, guidance=2.0,
                    seed=None, project=False, margin_yd=OOB_MARGIN_YD):
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)
        B = flags.shape[0]
        x = torch.randn(B, MAXP, CH, generator=g).to(self.device)
        null = torch.full((B,), 2, dtype=torch.long, device=self.device)
        ts = torch.linspace(T_STEPS - 1, 0, steps).long().to(self.device)
        for i in range(steps):
            t = ts[i].repeat(B)
            eps_c = model(x, flags, mask, t, label)
            eps_u = model(x, flags, mask, t, null)
            eps = eps_u + guidance * (eps_c - eps_u)
            ab_t = self.alpha_bar[ts[i]]
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0 = self.project_x0(x0, project, margin_yd)
            if i < steps - 1:
                ab_prev = self.alpha_bar[ts[i + 1]]
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
            else:
                x = x0
        return x


# --------------------------- training ---------------------------------------
def train(device=None, batch=BATCH, steps=TRAIN_STEPS, tag=""):
    device = device or pick_device()
    ids, X, F, M, y = load_dataset()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ids))
    n_hold = int(0.2 * len(ids))
    hold_idx, train_idx = perm[:n_hold], perm[n_hold:]
    split = pd.DataFrame({"corner_id": ids,
                          "split": ["train"] * len(ids)})
    split.loc[hold_idx, "split"] = "holdout"
    split.to_csv(f"{SP}/diffusion_split.csv", index=False)

    dev = torch.device(device)
    Xt = torch.tensor(X[train_idx]).to(dev)
    Ft = torch.tensor(F[train_idx]).to(dev)
    Mt = torch.tensor(M[train_idx]).to(dev)
    yt = torch.tensor(y[train_idx]).to(dev)
    N = len(train_idx)
    print(f"train corners: {N}  holdout: {n_hold}  device: {device}", flush=True)

    model = Denoiser().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"denoiser parameters: {n_par/1e6:.2f}M", flush=True)
    ema = {k: v.clone() for k, v in model.state_dict().items()}
    diff = Diffusion(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    jit = JITTER_M / 60.0  # jitter in normalized x units (y handled per-channel)

    def step_once(step_rng):
        idx = torch.randint(0, N, (batch,), device=dev)
        x0 = Xt[idx].clone()
        flags, mask, lab = Ft[idx], Mt[idx], yt[idx].clone()
        # position jitter augmentation
        x0[..., 0] += torch.randn_like(x0[..., 0]) * jit
        x0[..., 1] += torch.randn_like(x0[..., 1]) * (JITTER_M / 40.0)
        # classifier-free label dropout
        drop = torch.rand(batch, device=dev) < P_UNCOND
        lab[drop] = 2
        t = torch.randint(0, T_STEPS, (batch,), device=dev)
        noise = torch.randn_like(x0)
        xt = diff.add_noise(x0, t, noise)
        eps_hat = model(xt, flags, mask, t, lab)
        # loss mask: positions on real slots, velocities only where observed
        w = torch.zeros_like(x0)
        w[..., 0] = mask
        w[..., 1] = mask
        w[..., 2] = mask * flags[..., 3]
        w[..., 3] = mask * flags[..., 3]
        loss = ((eps_hat - noise) ** 2 * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(EMA_DECAY).add_(v, alpha=1 - EMA_DECAY)
                else:
                    ema[k] = v.clone()
        return loss.item()

    # ---- benchmark gate ----
    for _ in range(5):
        step_once(0)
    _sync(device)
    t0 = time.time()
    for _ in range(30):
        step_once(0)
    _sync(device)
    sps = 30 / (time.time() - t0)
    eta_min = steps / sps / 60
    print(f"benchmark: {sps:.1f} steps/s -> projected training "
          f"{eta_min:.1f} min for {steps} steps (batch {batch})", flush=True)
    if eta_min > TIME_BUDGET_MIN:
        print(f"ABORT: projected training time exceeds {TIME_BUDGET_MIN} min "
              f"budget on this machine. The script itself is the deliverable; "
              f"run it on a faster device.", flush=True)
        sys.exit(2)

    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        for gpm in opt.param_groups:
            gpm['lr'] = LR * 0.5 * (1 + math.cos(math.pi * step / steps))
        losses.append(step_once(step))
        if step % 2000 == 0:
            print(f"step {step}/{steps}  loss {np.mean(losses[-2000:]):.4f}  "
                  f"elapsed {(time.time()-t0)/60:.1f} min", flush=True)

    torch.save({"ema": ema, "model": model.state_dict(),
                "config": dict(d=160, layers=5, heads=4, T=T_STEPS)},
               f"{SP}/diffusion_model{tag}.pt")
    pd.DataFrame({"step": range(1, steps + 1), "loss": losses}) \
        .iloc[::50].to_csv(f"{SP}/diffusion_loss{tag}.csv", index=False)
    print(f"saved diffusion_model{tag}.pt", flush=True)


def load_model(device=None, tag=""):
    dev = torch.device(device or pick_device())
    ckpt = torch.load(f"{SP}/diffusion_model{tag}.pt", map_location=dev)
    model = Denoiser().to(dev)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    return model, Diffusion(dev)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    train(pick_device(), batch=a.batch, steps=a.steps, tag=a.tag)
