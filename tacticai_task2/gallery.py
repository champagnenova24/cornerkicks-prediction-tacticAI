"""Gallery of synthetic corners: real vs shot0/shot1-guided + sample diversity.

Each generated panel is annotated with the shot probability assigned by the
independent HGB classifier trained on real corners.
"""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

from diffusion_corners import pick_device, SP, load_dataset, load_model, denormalize
from diffusion_eval import (draw_corner, to_player_frame, real_player_frame,
                            globals_for)
from train_eval import CONFIGS

DDIM_STEPS, W = 200, 2.0


def main():
    device = pick_device()
    dev = torch.device(device)
    ids, X, F, M, y = load_dataset()
    split = pd.read_csv(f"{SP}/results/diffusion_split.csv")
    hold = np.where(split.split.to_numpy() == "holdout")[0]
    train_idx = np.where(split.split.to_numpy() == "train")[0]
    model, diff = load_model(device)
    gcols = CONFIGS["vel"][1]

    # independent shot scorer trained on real training corners
    dp_tr = real_player_frame(ids, X, F, M, train_idx)
    G_tr = globals_for(dp_tr, gcols)
    shot_by = dict(zip(ids, y))
    clf = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=40, random_state=0).fit(
        G_tr.to_numpy(float),
        np.array([shot_by[c] for c in G_tr.index]))

    def score(dpf):
        return float(clf.predict_proba(
            globals_for(dpf, gcols).to_numpy(float))[0, 1])

    # pick 3 representative holdout templates: typical visibility, velocity
    # observed for most players, mixed real labels
    rng = np.random.default_rng(3)
    def eligible(i):
        n = M[i].sum()
        return 12 <= n <= 18 and F[i, :, 3].sum() >= 0.5 * n
    pool = [i for i in hold if eligible(i)]
    shots = [i for i in pool if y[i] == 1]
    noshots = [i for i in pool if y[i] == 0]
    picks = (list(rng.choice(shots, 2, replace=False)) +
             list(rng.choice(noshots, 1, replace=False)))

    def gen(i, label, seed):
        f = torch.tensor(F[i:i + 1]).to(dev)
        m = torch.tensor(M[i:i + 1]).to(dev)
        lab = torch.tensor([label]).to(dev)
        z = diff.ddim_sample(model, f, m, lab, steps=DDIM_STEPS,
                             guidance=W, seed=seed).cpu().numpy()
        return to_player_frame(ids, denormalize(z), F, M, np.array([i]), "g")

    fig, axes = plt.subplots(4, 3, figsize=(11, 14))
    for r, i in enumerate(picks):
        cid = ids[i]
        dp_real = real_player_frame(ids, X, F, M, np.array([i]))
        draw_corner(axes[r, 0], dp_real,
                    f"real (shot={y[i]}, P={score(dp_real):.2f})")
        for c, lab in ((1, 0), (2, 1)):
            dpg = gen(i, lab, seed=100 + r)
            draw_corner(axes[r, c], dpg,
                        f"generated, guided shot={lab} (P={score(dpg):.2f})")
    # diversity row: same template, three seeds, real label
    i = picks[0]
    for c, seed in enumerate((7, 42, 99)):
        dpg = gen(i, int(y[i]), seed=seed)
        draw_corner(axes[3, c], dpg,
                    f"same template, seed {seed} (P={score(dpg):.2f})")
    fig.suptitle("Diffusion-generated corners — rows 1-3: real vs guided "
                 "generations; row 4: sample diversity\n(red = attack, "
                 "blue = defense, * = taker, square = GK, arrows = velocity, "
                 "P = independent classifier's shot prob)")
    fig.tight_layout()
    fig.savefig(f"{SP}/figures/synthetic_gallery.png", dpi=150)
    print("saved figures/synthetic_gallery.png", flush=True)


if __name__ == "__main__":
    main()
