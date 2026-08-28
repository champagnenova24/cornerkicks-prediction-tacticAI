"""Evaluate the corner-kick diffusion model.

1. Paper-style realism test: can an MLP (and a stronger HGB) tell held-out
   real corners from synthetic ones generated on the same roster templates?
   (TacticAI: MLP F1 = 0.53 +- 0.05, i.e. indistinguishable.)
2. Guidance check: does conditioning on shot=1 vs shot=0 shift the shot
   probability assigned by an independently trained shot classifier?
3. Feature distribution comparison + example figure.

Outputs: synthetic_corners.parquet, diffusion_eval.csv, real_vs_synthetic.png
"""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

from diffusion_corners import (pick_device, SP, MAXP, OOB_MARGIN_YD, load_dataset,
                               load_model, denormalize)
from train_eval import build_features, CONFIGS

GUIDANCE_REAL = 1.0   # realism sampling: plain conditional, no extrapolation
GUIDANCE_CTRL = 2.0   # control demo: push toward / away from shot setups
DDIM_STEPS = 200


def sample_variants(model, diff, F, M, y, idx, device, project=False,
                    margin_yd=OOB_MARGIN_YD):
    """Sample synthetic corners for the given templates under three labels:
    the real label, forced shot=1, forced shot=0.

    `project` forwards to the sampler's pitch projection; the default False
    reproduces every result already recorded in the README."""
    dev = torch.device(device)
    out = {}
    for name, labels, GUIDANCE in [
            ("real_label", y[idx], GUIDANCE_REAL),
            ("shot1", np.ones(len(idx), dtype=np.int64), GUIDANCE_CTRL),
            ("shot0", np.zeros(len(idx), dtype=np.int64), GUIDANCE_CTRL)]:
        chunks = []
        for s in range(0, len(idx), 128):
            sl = idx[s:s + 128]
            lab = torch.tensor(labels[s:s + 128]).to(dev)
            f = torch.tensor(F[sl]).to(dev)
            m = torch.tensor(M[sl]).to(dev)
            z = diff.ddim_sample(model, f, m, lab, steps=DDIM_STEPS,
                                 guidance=GUIDANCE, seed=1234 + s,
                                 project=project, margin_yd=margin_yd)
            chunks.append(z.cpu().numpy())
        out[name] = denormalize(np.concatenate(chunks))
    return out


def to_player_frame(ids, Z, F, M, idx, suffix):
    """Synthetic samples -> corner_players-style dataframe (velocities zeroed
    where the template had no observed velocity, for feature parity)."""
    rows = []
    for k, i in enumerate(idx):
        cid = f"{ids[i]}__{suffix}"
        n = int(M[i].sum())
        for j in range(n):
            has_vel = bool(F[i, j, 3])
            rows.append(dict(
                corner_id=cid, x=Z[k, j, 0], y=Z[k, j, 1],
                vx=Z[k, j, 2] if has_vel else 0.0,
                vy=Z[k, j, 3] if has_vel else 0.0,
                teammate=bool(F[i, j, 0]), keeper=bool(F[i, j, 1]),
                actor=bool(F[i, j, 2]), has_vel=has_vel, is_receiver=False))
    return pd.DataFrame(rows)


def real_player_frame(ids, X, F, M, idx):
    Z = denormalize(X)
    return to_player_frame(ids, Z[idx], F, M, idx, "real").assign(
        corner_id=lambda d: d.corner_id.str.replace("__real", "", regex=False))


def globals_for(dp_frame, gcols):
    _, dg = build_features(None, dp_frame)
    return dg[gcols]


def draw_corner(ax, g, title):
    ax.add_patch(plt.Rectangle((90, 0), 30, 80, fill=False, lw=1, ec="gray"))
    ax.plot([102, 102, 120], [18, 62, 62], c="gray", lw=1)
    ax.plot([102, 102], [18, 62], c="gray", lw=1)
    ax.add_patch(plt.Rectangle((114, 30), 6, 20, fill=False, lw=1, ec="gray"))
    ax.plot([120, 120], [36, 44], c="black", lw=3)
    for r in g.itertuples():
        c = "tab:red" if r.teammate else "tab:blue"
        mk = "*" if r.actor else ("s" if r.keeper else "o")
        ax.scatter(r.x, r.y, c=c, marker=mk, s=60 if r.actor else 35,
                   zorder=4 if (r.actor or r.keeper) else 3,
                   edgecolors="black", linewidths=0.4)
        if r.has_vel and (abs(r.vx) + abs(r.vy)) > 0.3:
            ax.arrow(r.x, r.y, r.vx, r.vy, head_width=0.8, color=c, alpha=0.6)
    ax.set_xlim(88, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def main(device=None):
    device = device or (pick_device())
    ids, X, F, M, y = load_dataset()
    split = pd.read_csv(f"{SP}/results/diffusion_split.csv")
    hold_mask = split.split.to_numpy() == "holdout"
    hold_idx = np.where(hold_mask)[0]
    train_idx = np.where(~hold_mask)[0]
    model, diff = load_model(device)
    print(f"sampling {len(hold_idx)} holdout templates x 3 variants "
          f"({DDIM_STEPS} DDIM steps, guidance {GUIDANCE_REAL}/{GUIDANCE_CTRL})",
          flush=True)
    variants = sample_variants(model, diff, F, M, y, hold_idx, device)

    # ---------- feature frames ----------
    gcols = CONFIGS["vel"][1]
    dp_real_hold = real_player_frame(ids, X, F, M, hold_idx)
    dp_real_train = real_player_frame(ids, X, F, M, train_idx)
    dp_syn = {k: to_player_frame(ids, v, F, M, hold_idx, k)
              for k, v in variants.items()}

    G_real_hold = globals_for(dp_real_hold, gcols)
    G_real_train = globals_for(dp_real_train, gcols)
    G_syn = {k: globals_for(v, gcols) for k, v in dp_syn.items()}

    # ---------- 1. realism: real vs synthetic discriminators ----------
    Xd = np.vstack([G_real_hold.to_numpy(float),
                    G_syn["real_label"].to_numpy(float)])
    yd = np.concatenate([np.ones(len(G_real_hold)),
                         np.zeros(len(G_syn["real_label"]))])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xd))
    n_tr = int(0.7 * len(Xd))
    tr, te = perm[:n_tr], perm[n_tr:]
    sc = StandardScaler().fit(Xd[tr])
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                        random_state=0).fit(sc.transform(Xd[tr]), yd[tr])
    p_mlp = mlp.predict_proba(sc.transform(Xd[te]))[:, 1]
    hgb = HistGradientBoostingClassifier(random_state=0).fit(Xd[tr], yd[tr])
    p_hgb = hgb.predict_proba(Xd[te])[:, 1]
    disc = dict(
        disc_mlp_f1=f1_score(yd[te], p_mlp >= 0.5),
        disc_mlp_auc=roc_auc_score(yd[te], p_mlp),
        disc_hgb_f1=f1_score(yd[te], p_hgb >= 0.5),
        disc_hgb_auc=roc_auc_score(yd[te], p_hgb),
    )
    print("discriminator:", {k: round(v, 3) for k, v in disc.items()}, flush=True)

    # ---------- 2. guidance check with an independent shot classifier ------
    shot_by_cid = dict(zip([ids[i] for i in range(len(ids))], y))
    y_tr = np.array([shot_by_cid[c] for c in G_real_train.index])
    clf = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=40, random_state=0).fit(
        G_real_train.to_numpy(float), y_tr)

    def mean_p(G):
        return float(clf.predict_proba(G.to_numpy(float))[:, 1].mean())

    y_hold = np.array([shot_by_cid[c] for c in G_real_hold.index])
    guid = dict(
        p_shot_real_shot=float(clf.predict_proba(
            G_real_hold.to_numpy(float))[y_hold == 1, 1].mean()),
        p_shot_real_noshot=float(clf.predict_proba(
            G_real_hold.to_numpy(float))[y_hold == 0, 1].mean()),
        p_shot_gen_shot1=mean_p(G_syn["shot1"]),
        p_shot_gen_shot0=mean_p(G_syn["shot0"]),
        p_shot_gen_reallabel=mean_p(G_syn["real_label"]),
    )
    print("guidance:", {k: round(v, 3) for k, v in guid.items()}, flush=True)

    # ---------- 3. distribution table ----------
    dist = pd.DataFrame({
        "real_holdout_mean": G_real_hold.mean(),
        "synthetic_mean": G_syn["real_label"].mean(),
        "real_holdout_std": G_real_hold.std(),
        "synthetic_std": G_syn["real_label"].std(),
    }).round(3)
    dist.to_csv(f"{SP}/results/diffusion_distributions.csv")

    # ---------- 4. figure ----------
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    pick = hold_idx[:3]
    for col, i in enumerate(pick):
        cid = ids[i]
        draw_corner(axes[0, col], dp_real_hold[dp_real_hold.corner_id == cid],
                    f"real (shot={y[i]})")
        draw_corner(axes[1, col],
                    dp_syn["real_label"][dp_syn["real_label"].corner_id
                                         == f"{cid}__real_label"],
                    f"synthetic (label={y[i]})")
    fig.suptitle("Real vs diffusion-generated corners (red = attack, "
                 "* = taker, square = GK, arrows = velocity)")
    fig.tight_layout()
    fig.savefig(f"{SP}/figures/real_vs_synthetic.png", dpi=150)

    # ---------- save ----------
    syn_all = pd.concat([v.assign(variant=k) for k, v in dp_syn.items()])
    syn_all.to_parquet(f"{SP}/synthetic_corners.parquet")
    res = pd.Series({**disc, **guid,
                     "n_holdout": len(hold_idx),
                     "guidance_real": GUIDANCE_REAL,
                     "guidance_ctrl": GUIDANCE_CTRL,
                     "ddim_steps": DDIM_STEPS})
    res.to_csv(f"{SP}/results/diffusion_eval.csv")
    print(res.round(3), flush=True)


if __name__ == "__main__":
    main()
