"""Delivery-conditioned set diffusion for corner setups.

`diffusion_corners.py` generates a corner setup from a roster template and the
shot label. It is never told the delivery -- not where the ball arrives, not
the swing, not the height -- so it can only produce the setup averaged over
every ball that roster might receive. `synthetic_trajectories.py` measured the
consequence: on the same deliveries, real setups put an attacker within 3 yd of
the arrival point on 73% of corners, generated ones on 34%. The 6-yd-box
shortfall (28.7% vs 35.1% of players) that capacity, steps, batch size and
pitch projection all failed to move is the same smearing seen from another
side.

This model adds the delivery as a conditioning variable:

    end location   (x, y) of pass.end_location, tacticai frame, normalized
                   exactly like player positions
    technique      Inswinging / Outswinging / Straight / none (short corners)
    height         High / Low / Ground Pass

It enters twice. Every player slot gets the arrival point concatenated to its
input, so "player minus ball" is linear in the first layer -- the quantity the
ball-consistency metric measures. And a small MLP embeds the whole delivery
into the global conditioning vector alongside time and label, which is where
technique and height act.

Label and delivery are dropped independently during training (15% each), so
one checkpoint serves four queries: (label, delivery), (label only),
(delivery only), (neither). Delivery-blind sampling from this same checkpoint
is the matched control for the delivery effect -- same weights, same training
budget, only the conditioning differs.

Leakage boundary, as for `vel+delivery` in train_eval.py: the delivery is
known only after the kick, so this is a retrospective generator. For the
attackers' flight-window velocities that is the point (they run to the ball).
For positions at the kick it means "setups that precede a delivery like this",
which mixes the taker choosing the ball to fit the setup with the setup being
arranged for the planned ball. Both are what a coach asking "show me a setup
for an inswinger to the near post" wants to see; neither is a causal claim.

Run:  python diffusion_delivery.py [--steps N] [--batch B] [--tag T]
      Resumable: a checkpoint is written every CKPT_EVERY steps and picked up
      again by a re-run with the same tag.
"""
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import outdirs  # noqa: F401 -- creates results/ figures/ models/
from diffusion_corners import (SP, MAXP, CH, FLAGS, T_STEPS, LR, EMA_DECAY,
                               P_UNCOND, JITTER_M, TimeEmbed, Diffusion,
                               load_dataset, pick_device, _sync)

TECHNIQUES = ['Inswinging', 'Outswinging', 'Straight', None]
HEIGHTS = ['High Pass', 'Low Pass', 'Ground Pass']
DELIV = 2 + len(TECHNIQUES) + len(HEIGHTS)   # 9
P_DROP_DELIV = 0.15
TRAIN_STEPS = 30000
BATCH = 256
CKPT_EVERY = 2500


def encode_delivery(end_x, end_y, technique, height):
    """One corner's delivery -> DELIV-dim vector (end point normalized like
    player positions: the pitch maps onto [-1, 1]^2)."""
    v = np.zeros(DELIV, dtype=np.float32)
    v[0] = (end_x - 60.0) / 60.0
    v[1] = (end_y - 40.0) / 40.0
    t = technique if technique in TECHNIQUES else None
    v[2 + TECHNIQUES.index(t)] = 1.0
    v[2 + len(TECHNIQUES) + HEIGHTS.index(height)] = 1.0
    return v


def load_deliveries(ids):
    """Delivery vectors aligned to `load_dataset()` ids.

    `corner_deliveries.parquet` is already in the tacticai frame (kick at
    (120, 0)), the same frame as `corner_players.parquet`.
    """
    dl = pd.read_parquet(f'{SP}/corner_deliveries.parquet').set_index('corner_id')
    D = np.stack([encode_delivery(r.end_x, r.end_y,
                                  None if pd.isna(r.technique) else r.technique,
                                  r.height_class)
                  for r in dl.loc[list(ids)].itertuples()])
    return D


class DeliveryDenoiser(nn.Module):
    """`diffusion_corners.Denoiser` plus the delivery, per slot and global.

    Per-slot input: generated channels, roster flags, arrival point, and a
    delivery-present bit (so a dropped delivery is distinguishable from a
    ball arriving at the pitch centre).
    """

    def __init__(self, d=160, layers=5, heads=4):
        super().__init__()
        self.inp = nn.Linear(CH + FLAGS + 2 + 1, d)
        self.time = TimeEmbed(d)
        self.label = nn.Embedding(3, d)     # 0 = no shot, 1 = shot, 2 = null
        self.deliv = nn.Sequential(nn.Linear(DELIV + 1, d), nn.SiLU(),
                                   nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=2 * d,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers,
                                             enable_nested_tensor=False)
        self.out = nn.Linear(d, CH)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, flags, mask, t, label, deliv, has_deliv):
        """deliv: (B, DELIV), zeroed where has_deliv == 0. has_deliv: (B,)."""
        deliv = deliv * has_deliv[:, None]
        B, P, _ = x.shape
        per_slot = torch.cat([deliv[:, None, :2].expand(B, P, 2),
                              has_deliv[:, None, None].expand(B, P, 1)], -1)
        h = self.inp(torch.cat([x, flags, per_slot], dim=-1))
        cond = (self.time(t) + self.label(label)
                + self.deliv(torch.cat([deliv, has_deliv[:, None]], -1)))
        h = h + cond[:, None, :]
        h = self.encoder(h, src_key_padding_mask=(mask < 0.5))
        return self.out(h)


class DeliveryDiffusion(Diffusion):
    @torch.no_grad()
    def ddim_sample(self, model, flags, mask, label, deliv=None, steps=200,
                    guidance=1.0, seed=None, project=False, margin_yd=3.25):
        """DDIM with classifier-free guidance on the LABEL only.

        With a delivery, both passes see it -- guidance pushes toward the
        label *given this ball*, it never extrapolates the delivery itself.
        `deliv=None` samples delivery-blind (the matched control).
        """
        dev = self.device
        g = torch.Generator(device='cpu')
        if seed is not None:
            g.manual_seed(seed)
        B = flags.shape[0]
        x = torch.randn(B, MAXP, CH, generator=g).to(dev)
        null = torch.full((B,), 2, dtype=torch.long, device=dev)
        if deliv is None:
            deliv = torch.zeros(B, DELIV, device=dev)
            has = torch.zeros(B, device=dev)
        else:
            has = torch.ones(B, device=dev)
        ts = torch.linspace(T_STEPS - 1, 0, steps).long().to(dev)
        for i in range(steps):
            t = ts[i].repeat(B)
            eps_c = model(x, flags, mask, t, label, deliv, has)
            if guidance == 1.0:
                eps = eps_c
            else:
                eps_u = model(x, flags, mask, t, null, deliv, has)
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


def ckpt_path(tag=''):
    return f'{SP}/models/diffusion_delivery{tag}.pt'


def train(device=None, batch=BATCH, steps=TRAIN_STEPS, tag=''):
    device = device or pick_device()
    torch.manual_seed(0)
    ids, X, F, M, y = load_dataset()
    D = load_deliveries(ids)
    # Reuse the committed split so every comparison with the delivery-blind
    # model is on the same 485 holdout templates.
    split = pd.read_csv(f'{SP}/results/diffusion_split.csv')
    assert list(split.corner_id) == list(ids), 'split/dataset order mismatch'
    train_idx = np.where(split.split.to_numpy() == 'train')[0]

    dev = torch.device(device)
    Xt, Ft, Mt, yt, Dt = (torch.tensor(a[train_idx]).to(dev)
                          for a in (X, F, M, y, D))
    N = len(train_idx)
    model = DeliveryDenoiser().to(dev)
    print(f'train corners: {N}  device: {device}  params: '
          f'{sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)
    ema = {k: v.clone() for k, v in model.state_dict().items()}
    diff = Diffusion(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    losses, start = [], 1
    path = ckpt_path(tag)
    if os.path.exists(path):
        ck = torch.load(path, map_location=dev)
        if ck.get('step', steps) < steps and ck.get('total_steps') == steps:
            model.load_state_dict(ck['model'])
            ema = ck['ema']
            opt.load_state_dict(ck['opt'])
            losses = list(ck['losses'])
            start = ck['step'] + 1
            torch.set_rng_state(ck['rng'])
            print(f'resuming from step {ck["step"]}', flush=True)
        elif ck.get('step') == steps:
            print(f'{os.path.basename(path)} is already trained to {steps} '
                  f'steps; delete it to retrain', flush=True)
            return

    def save(step):
        torch.save({'ema': ema, 'model': model.state_dict(),
                    'opt': opt.state_dict(), 'losses': losses, 'step': step,
                    'total_steps': steps, 'rng': torch.get_rng_state(),
                    'config': dict(d=160, layers=5, heads=4, T=T_STEPS,
                                   batch=batch, p_drop_deliv=P_DROP_DELIV)},
                   path)

    t0 = time.time()
    for step in range(start, steps + 1):
        for gpm in opt.param_groups:
            gpm['lr'] = LR * 0.5 * (1 + math.cos(math.pi * step / steps))
        idx = torch.randint(0, N, (batch,), device=dev)
        x0 = Xt[idx].clone()
        flags, mask, lab, dv = Ft[idx], Mt[idx], yt[idx].clone(), Dt[idx]
        x0[..., 0] += torch.randn_like(x0[..., 0]) * (JITTER_M / 60.0)
        x0[..., 1] += torch.randn_like(x0[..., 1]) * (JITTER_M / 40.0)
        lab[torch.rand(batch, device=dev) < P_UNCOND] = 2
        has = (torch.rand(batch, device=dev) >= P_DROP_DELIV).float()
        t = torch.randint(0, T_STEPS, (batch,), device=dev)
        noise = torch.randn_like(x0)
        eps_hat = model(diff.add_noise(x0, t, noise), flags, mask, t, lab,
                        dv, has)
        w = torch.zeros_like(x0)
        w[..., 0] = w[..., 1] = mask
        w[..., 2] = w[..., 3] = mask * flags[..., 3]
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
        losses.append(loss.item())
        if step % 500 == 0:
            _sync(device)
            print(f'step {step}/{steps}  loss {np.mean(losses[-500:]):.4f}  '
                  f'elapsed {(time.time()-t0)/60:.1f} min', flush=True)
        if step % CKPT_EVERY == 0 or step == steps:
            save(step)

    pd.DataFrame({'step': range(1, len(losses) + 1), 'loss': losses}) \
        .iloc[::50].to_csv(f'{SP}/results/diffusion_delivery_loss{tag}.csv',
                           index=False)
    print(f'saved models/diffusion_delivery{tag}.pt', flush=True)


def load_model(device=None, tag=''):
    dev = torch.device(device or pick_device())
    ck = torch.load(ckpt_path(tag), map_location=dev)
    model = DeliveryDenoiser().to(dev)
    model.load_state_dict(ck['ema'])
    model.eval()
    return model, DeliveryDiffusion(dev)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--steps', type=int, default=TRAIN_STEPS)
    ap.add_argument('--tag', type=str, default='')
    a = ap.parse_args()
    train(pick_device(), batch=a.batch, steps=a.steps, tag=a.tag)
