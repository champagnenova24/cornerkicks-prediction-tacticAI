"""Positioning suggestions, round 2: best-of-k candidate selection.

Mirrors the paper's deployment loop: the generative model PROPOSES k candidate
adjustments per corner, the shot predictor EVALUATES them, and the best one is
kept (max P for attack/shot=1 suggestions, min P for defense/shot=0).

Outputs: tactic_suggestions.csv (vanilla + best-of-k columns),
         tactic_suggestions.png, tactic_best_players.parquet
"""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

from diffusion_corners import pick_device, SP, load_dataset, load_model, denormalize
from diffusion_eval import to_player_frame, real_player_frame, globals_for
from train_eval import CONFIGS
from tactic_suggest import ddim_inpaint, draw_panel
import tactic_suggest

K = 6                # candidates per corner per mode
tactic_suggest.STEPS = 150   # candidate sampling steps (speed/quality tradeoff)


def gen_candidates(model, diff, X, F, M, idx, mode, device, seed0):
    dev = torch.device(device)
    lab_val = 1 if mode == "attack" else 0
    out = []
    for s in range(0, len(idx), 128):
        sl = idx[s:s + 128]
        x0 = torch.tensor(X[sl]).to(dev)
        f = torch.tensor(F[sl]).to(dev)
        m = torch.tensor(M[sl]).to(dev)
        team = f[..., 0] > 0.5
        actor = f[..., 2] > 0.5
        move = (team if mode == "attack" else ~team) & ~actor & (m > 0.5)
        lab = torch.full((len(sl),), lab_val, dtype=torch.long, device=dev)
        z = ddim_inpaint(diff, model, x0, ~move, f, m, lab, seed=seed0 + s)
        out.append(z.cpu().numpy())
    return denormalize(np.concatenate(out))


def main():
    device = pick_device()
    ids, X, F, M, y = load_dataset()
    split = pd.read_csv(f"{SP}/results/diffusion_split.csv")
    hold_idx = np.where(split.split.to_numpy() == "holdout")[0]
    train_idx = np.where(split.split.to_numpy() == "train")[0]
    model, diff = load_model(device)
    gcols = CONFIGS["vel"][1]

    G_tr = globals_for(real_player_frame(ids, X, F, M, train_idx), gcols)
    shot_by = dict(zip(ids, y))
    clf = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=40, random_state=0).fit(
        G_tr.to_numpy(float),
        np.array([shot_by[c] for c in G_tr.index]))

    def score_frame(dpf):
        G = globals_for(dpf, gcols)
        return pd.Series(clf.predict_proba(G.to_numpy(float))[:, 1],
                         index=G.index)

    dp_orig = real_player_frame(ids, X, F, M, hold_idx)
    p_orig = score_frame(dp_orig)

    best = {}
    for mode in ("attack", "defense"):
        allP, allZ, allF = [], [], []
        for k in range(K):
            Z = gen_candidates(model, diff, X, F, M, hold_idx, mode,
                               device, seed0=20000 + 1000 * k)
            dpf = to_player_frame(ids, Z, F, M, hold_idx, f"{mode}{k}")
            p = score_frame(dpf)
            p.index = p.index.str.replace(f"__{mode}{k}", "", regex=False)
            allP.append(p)
            allZ.append(Z)
            allF.append(dpf)
            print(f"{mode} candidate {k + 1}/{K} scored "
                  f"(mean P {p.mean():.3f})", flush=True)
        P = pd.concat(allP, axis=1)              # corners x K
        pick = P.to_numpy().argmax(1) if mode == "attack" \
            else P.to_numpy().argmin(1)
        chosen_p = P.to_numpy()[np.arange(len(P)), pick]
        Zbest = np.stack([allZ[pick[j]][j] for j in range(len(P))])
        best[mode] = dict(p=pd.Series(chosen_p, index=P.index),
                          p_first=P.iloc[:, 0], Z=Zbest)

    # ---------- results table ----------
    rows = []
    for j, i in enumerate(hold_idx):
        cid = ids[i]
        rows.append(dict(
            corner_id=cid, shot_real=int(y[i]),
            p_original=p_orig.get(cid, np.nan),
            p_attack_1=best["attack"]["p_first"].get(cid, np.nan),
            p_attack_bestk=best["attack"]["p"].get(cid, np.nan),
            p_defense_1=best["defense"]["p_first"].get(cid, np.nan),
            p_defense_bestk=best["defense"]["p"].get(cid, np.nan)))
    df = pd.DataFrame(rows).dropna()
    df.to_csv(f"{SP}/results/tactic_suggestions.csv", index=False)

    m = df.mean(numeric_only=True)
    print("\n=== aggregate P(shot), independent classifier ===", flush=True)
    print(f"original                : {m.p_original:.3f}", flush=True)
    print(f"attack  single / best-{K}: {m.p_attack_1:.3f} / "
          f"{m.p_attack_bestk:.3f}  (target: higher)", flush=True)
    print(f"defense single / best-{K}: {m.p_defense_1:.3f} / "
          f"{m.p_defense_bestk:.3f}  (target: lower)", flush=True)
    thr = df[df.shot_real == 1]
    print(f"defense best-{K} on real shot-corners: "
          f"{thr.p_original.mean():.3f} -> {thr.p_defense_bestk.mean():.3f}",
          flush=True)
    print(f"attack improved: {(df.p_attack_bestk > df.p_original).mean():.0%}"
          f"   defense improved: "
          f"{(df.p_defense_bestk < df.p_original).mean():.0%}", flush=True)

    # ---------- figure: high-effect + median example ----------
    dp_att = to_player_frame(ids, best["attack"]["Z"], F, M, hold_idx, "att")
    dp_def = to_player_frame(ids, best["defense"]["Z"], F, M, hold_idx, "def")
    df["att_gain"] = df.p_attack_bestk - df.p_original
    df["def_gain"] = df.p_original - df.p_defense_bestk
    both = df.att_gain.rank(pct=True) + df.def_gain.rank(pct=True)
    pick_hi = df.loc[both.idxmax(), "corner_id"]
    pick_md = df.loc[(both - both.median()).abs().idxmin(), "corner_id"]

    fig, axes = plt.subplots(2, 3, figsize=(11, 9.5))
    for r, cid in enumerate([pick_hi, pick_md]):
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
                   f"ATTACK suggestion (P={row.p_attack_bestk:.2f})",
                   moved_pairs=pairs(o, a, True))
        draw_panel(axes[r, 2], d,
                   f"DEFENSE suggestion (P={row.p_defense_bestk:.2f})",
                   moved_pairs=pairs(o, d, False))
    fig.suptitle(
        "Positioning suggestions: diffusion inpainting + best-of-%d selection "
        "by the shot predictor (paper Fig. 3 analog)\n"
        "red = attack, blue = defense, * = taker (fixed), square = GK; "
        "gray circles/lines = original -> suggested position;\n"
        "row 1: high-effect example, row 2: median example" % K)
    fig.tight_layout()
    fig.savefig(f"{SP}/figures/tactic_suggestions.png", dpi=150)
    dp_best = pd.concat([dp_att.assign(mode="attack"),
                         dp_def.assign(mode="defense")])
    dp_best.to_parquet(f"{SP}/tactic_best_players.parquet")
    print("saved figures/tactic_suggestions.png / results/tactic_suggestions.csv"
          " / tactic_best_players.parquet",
          flush=True)


if __name__ == "__main__":
    main()
