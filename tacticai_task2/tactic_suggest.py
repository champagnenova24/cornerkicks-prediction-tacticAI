"""Positioning suggestions via diffusion inpainting (paper Fig. 3 analog).

For every holdout corner:
  ATTACK suggestion : freeze defenders + the taker, re-denoise the outfield
                      attackers conditioned on shot=1 (guidance w=2)
  DEFENSE suggestion: freeze attackers + taker, re-denoise the defenders
                      conditioned on shot=0 (guidance w=2)

Every setup (original / attack-adjusted / defense-adjusted) is scored with the
independent HGB shot classifier trained on real training corners only.

Outputs: tactic_suggestions.csv (per-corner P values),
         tactic_suggestions.png (example illustrations)
"""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

from diffusion_corners import (SP, MAXP, T_STEPS, load_dataset, load_model,
                               denormalize)
from diffusion_eval import to_player_frame, real_player_frame, globals_for
from train_eval import CONFIGS

STEPS, W = 200, 2.0


@torch.no_grad()
def ddim_inpaint(diff, model, x0, known, flags, mask, label, seed=0):
    """DDIM sampling with the `known` player slots pinned to their real values
    (RePaint-style single pass)."""
    dev = diff.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = flags.shape[0]
    x = torch.randn(B, MAXP, 4, generator=g).to(dev)
    null = torch.full((B,), 2, dtype=torch.long, device=dev)
    ts = torch.linspace(T_STEPS - 1, 0, STEPS).long().to(dev)
    kn = known[..., None]
    for i in range(STEPS):
        ab_t = diff.alpha_bar[ts[i]]
        noise = torch.randn(B, MAXP, 4, generator=g).to(dev)
        x = torch.where(kn, ab_t.sqrt() * x0 + (1 - ab_t).sqrt() * noise, x)
        t = ts[i].repeat(B)
        eps_c = model(x, flags, mask, t, label)
        eps_u = model(x, flags, mask, t, null)
        eps = eps_u + W * (eps_c - eps_u)
        x0_hat = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        x0_hat = x0_hat.clamp(-2.5, 2.5)
        if i < STEPS - 1:
            ab_prev = diff.alpha_bar[ts[i + 1]]
            x = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps
        else:
            x = torch.where(kn, x0, x0_hat)
    return x


def suggest(model, diff, X, F, M, idx, mode, device):
    """mode='attack': move outfield attackers, label=1.
       mode='defense': move outfield defenders, label=0."""
    dev = torch.device(device)
    out = []
    lab_val = 1 if mode == "attack" else 0
    for s in range(0, len(idx), 128):
        sl = idx[s:s + 128]
        x0 = torch.tensor(X[sl]).to(dev)
        f = torch.tensor(F[sl]).to(dev)
        m = torch.tensor(M[sl]).to(dev)
        team = f[..., 0] > 0.5          # teammate flag
        actor = f[..., 2] > 0.5
        move = (team if mode == "attack" else ~team) & ~actor & (m > 0.5)
        known = ~move                    # includes padding: pinned to zeros
        lab = torch.full((len(sl),), lab_val, dtype=torch.long, device=dev)
        z = ddim_inpaint(diff, model, x0, known, f, m, lab, seed=9000 + s)
        out.append(z.cpu().numpy())
    return denormalize(np.concatenate(out))


def draw_panel(ax, dpf, title, moved_pairs=None):
    ax.add_patch(plt.Rectangle((90, 0), 30, 80, fill=False, lw=1, ec="gray"))
    ax.plot([102, 102], [18, 62], c="gray", lw=1)
    ax.plot([102, 120], [18, 18], c="gray", lw=1)
    ax.plot([102, 120], [62, 62], c="gray", lw=1)
    ax.add_patch(plt.Rectangle((114, 30), 6, 20, fill=False, lw=1, ec="gray"))
    ax.plot([120, 120], [36, 44], c="black", lw=3)
    if moved_pairs is not None:
        for (xo, yo, xn, yn) in moved_pairs:
            ax.plot([xo, xn], [yo, yn], c="dimgray", lw=0.9, alpha=0.8,
                    zorder=2)
            ax.scatter(xo, yo, facecolors="none", edgecolors="dimgray",
                       s=26, lw=0.8, zorder=2)
    for r in dpf.itertuples():
        c = "tab:red" if r.teammate else "tab:blue"
        mk = "*" if r.actor else ("s" if r.keeper else "o")
        ax.scatter(r.x, r.y, c=c, marker=mk, s=70 if r.actor else 40,
                   zorder=4 if (r.actor or r.keeper) else 3,
                   edgecolors="black", linewidths=0.4)
        if r.has_vel and (abs(r.vx) + abs(r.vy)) > 0.3:
            ax.arrow(r.x, r.y, r.vx, r.vy, head_width=0.8, color=c, alpha=0.5)
    ax.set_xlim(88, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids, X, F, M, y = load_dataset()
    split = pd.read_csv(f"{SP}/diffusion_split.csv")
    hold_idx = np.where(split.split.to_numpy() == "holdout")[0]
    train_idx = np.where(split.split.to_numpy() == "train")[0]
    model, diff = load_model(device)
    gcols = CONFIGS["vel"][1]

    # independent shot classifier (real training corners only)
    G_tr = globals_for(real_player_frame(ids, X, F, M, train_idx), gcols)
    shot_by = dict(zip(ids, y))
    clf = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=40, random_state=0).fit(
        G_tr.to_numpy(float),
        np.array([shot_by[c] for c in G_tr.index]))

    def scores(dpf):
        G = globals_for(dpf, gcols)
        p = clf.predict_proba(G.to_numpy(float))[:, 1]
        return dict(zip(G.index, p))

    print(f"generating suggestions for {len(hold_idx)} holdout corners "
          f"(2 x {STEPS} steps, w={W})", flush=True)
    Z_att = suggest(model, diff, X, F, M, hold_idx, "attack", device)
    print("attack suggestions done", flush=True)
    Z_def = suggest(model, diff, X, F, M, hold_idx, "defense", device)
    print("defense suggestions done", flush=True)

    dp_orig = real_player_frame(ids, X, F, M, hold_idx)
    dp_att = to_player_frame(ids, Z_att, F, M, hold_idx, "att")
    dp_def = to_player_frame(ids, Z_def, F, M, hold_idx, "def")
    p_orig = scores(dp_orig)
    p_att = scores(dp_att)
    p_def = scores(dp_def)

    rows = []
    for k, i in enumerate(hold_idx):
        cid = ids[i]
        rows.append(dict(
            corner_id=cid, shot_real=int(y[i]),
            p_original=p_orig.get(cid, np.nan),
            p_attack_adj=p_att.get(f"{cid}__att", np.nan),
            p_defense_adj=p_def.get(f"{cid}__def", np.nan)))
    df = pd.DataFrame(rows).dropna()
    df["att_gain"] = df.p_attack_adj - df.p_original
    df["def_gain"] = df.p_original - df.p_defense_adj
    df.to_csv(f"{SP}/tactic_suggestions.csv", index=False)

    print("\n=== aggregate effects (independent classifier) ===", flush=True)
    print(f"ATTACK  (shot=1 target): mean P {df.p_original.mean():.3f} -> "
          f"{df.p_attack_adj.mean():.3f}   improved on "
          f"{(df.att_gain > 0).mean():.0%} of corners", flush=True)
    print(f"DEFENSE (shot=0 target): mean P {df.p_original.mean():.3f} -> "
          f"{df.p_defense_adj.mean():.3f}   improved on "
          f"{(df.def_gain > 0).mean():.0%} of corners", flush=True)
    thr = df[df.shot_real == 1]
    print(f"DEFENSE on real shot-corners only: {thr.p_original.mean():.3f} -> "
          f"{thr.p_defense_adj.mean():.3f}", flush=True)

    # ---- illustration: one high-effect and one median-effect corner -------
    ok = df[(df.att_gain.notna()) & (df.def_gain.notna())]
    both = (ok.att_gain.rank(pct=True) + ok.def_gain.rank(pct=True))
    pick_hi = ok.loc[both.idxmax(), "corner_id"]
    med_idx = (both - both.median()).abs().idxmin()
    pick_md = ok.loc[med_idx, "corner_id"]

    fig, axes = plt.subplots(2, 3, figsize=(11, 9.5))
    for r, cid in enumerate([pick_hi, pick_md]):
        i = ids.index(cid)
        row = df[df.corner_id == cid].iloc[0]
        o = dp_orig[dp_orig.corner_id == cid]
        a = dp_att[dp_att.corner_id == f"{cid}__att"]
        d = dp_def[dp_def.corner_id == f"{cid}__def"]

        def pairs(orig, adj, team_flag):
            po = orig[(orig.teammate == team_flag) & (~orig.actor)]
            pa = adj[(adj.teammate == team_flag) & (~adj.actor)]
            return list(zip(po.x, po.y, pa.x, pa.y))

        draw_panel(axes[r, 0], o,
                   f"original (real shot={row.shot_real}, "
                   f"P={row.p_original:.2f})")
        draw_panel(axes[r, 1], a,
                   f"ATTACK suggestion, shot=1 (P={row.p_attack_adj:.2f})",
                   moved_pairs=pairs(o, a, True))
        draw_panel(axes[r, 2], d,
                   f"DEFENSE suggestion, shot=0 (P={row.p_defense_adj:.2f})",
                   moved_pairs=pairs(o, d, False))
    fig.suptitle(
        "Diffusion inpainting positioning suggestions (paper Fig. 3 analog)\n"
        "red = attack, blue = defense, * = taker (fixed), square = GK; "
        "gray circles/lines = original position -> suggested position;\n"
        "row 1: high-effect example, row 2: median example; "
        "P = independent classifier shot probability")
    fig.tight_layout()
    fig.savefig(f"{SP}/tactic_suggestions.png", dpi=150)
    print("saved tactic_suggestions.png", flush=True)


if __name__ == "__main__":
    main()
