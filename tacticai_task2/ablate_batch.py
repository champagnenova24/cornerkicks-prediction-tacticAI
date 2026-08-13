"""Batch-size ablation for the corner diffusion model.

Trains the same architecture at batch sizes 64/128/256/512 (fixed 30k steps,
identical split, LR schedule rescaled to the step budget), then evaluates every
checkpoint with the same protocol:
  - realism: MLP + HGB discriminators, real vs synthetic (w=1, 200 DDIM steps)
  - guidance: mean P(shot) under shot=0 / shot=1 guidance (w=2), scored by an
    independent HGB shot classifier trained on real corners
Output: ablation_batch.csv
"""
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

from diffusion_corners import SP, load_dataset, load_model
from diffusion_eval import (sample_variants, to_player_frame,
                            real_player_frame, globals_for)
from train_eval import CONFIGS

BATCHES = [64, 128, 256, 512]
STEPS = 30000


def evaluate(tag, ids, X, F, M, y, hold_idx, train_idx, device):
    model, diff = load_model(device, tag=tag)
    variants = sample_variants(model, diff, F, M, y, hold_idx, device)
    gcols = CONFIGS["vel"][1]

    dp_real_hold = real_player_frame(ids, X, F, M, hold_idx)
    dp_real_train = real_player_frame(ids, X, F, M, train_idx)
    G_real_hold = globals_for(dp_real_hold, gcols)
    G_real_train = globals_for(dp_real_train, gcols)
    G_syn = {k: globals_for(to_player_frame(ids, v, F, M, hold_idx, k), gcols)
             for k, v in variants.items()}

    # --- discriminators ---
    Xd = np.vstack([G_real_hold.to_numpy(float),
                    G_syn["real_label"].to_numpy(float)])
    yd = np.concatenate([np.ones(len(G_real_hold)),
                         np.zeros(len(G_syn["real_label"]))])
    perm = np.random.default_rng(0).permutation(len(Xd))
    tr, te = perm[:int(0.7 * len(Xd))], perm[int(0.7 * len(Xd)):]
    sc = StandardScaler().fit(Xd[tr])
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                        random_state=0).fit(sc.transform(Xd[tr]), yd[tr])
    p_mlp = mlp.predict_proba(sc.transform(Xd[te]))[:, 1]
    hgb = HistGradientBoostingClassifier(random_state=0).fit(Xd[tr], yd[tr])
    p_hgb = hgb.predict_proba(Xd[te])[:, 1]

    # --- guidance separation ---
    shot_by = dict(zip(ids, y))
    clf = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=40, random_state=0).fit(
        G_real_train.to_numpy(float),
        np.array([shot_by[c] for c in G_real_train.index]))

    def mp(G):
        return float(clf.predict_proba(G.to_numpy(float))[:, 1].mean())

    return dict(
        disc_mlp_f1=f1_score(yd[te], p_mlp >= 0.5),
        disc_mlp_auc=roc_auc_score(yd[te], p_mlp),
        disc_hgb_f1=f1_score(yd[te], p_hgb >= 0.5),
        disc_hgb_auc=roc_auc_score(yd[te], p_hgb),
        p_shot_gen_shot0=mp(G_syn["shot0"]),
        p_shot_gen_shot1=mp(G_syn["shot1"]),
        p_shot_gen_reallabel=mp(G_syn["real_label"]),
    )


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []
    for b in BATCHES:
        tag = f"_b{b}"
        print(f"=== training batch={b} ===", flush=True)
        r = subprocess.run(
            [sys.executable, f"{SP}/diffusion_corners.py",
             "--batch", str(b), "--steps", str(STEPS), "--tag", tag],
            capture_output=True, text=True)
        print(r.stdout[-500:], flush=True)
        if r.returncode != 0:
            print(f"TRAINING FAILED for batch {b}:\n{r.stderr[-800:]}", flush=True)
            continue
        ids, X, F, M, y = load_dataset()
        split = pd.read_csv(f"{SP}/diffusion_split.csv")
        hold_idx = np.where(split.split.to_numpy() == "holdout")[0]
        train_idx = np.where(split.split.to_numpy() == "train")[0]
        print(f"=== evaluating batch={b} ===", flush=True)
        res = evaluate(tag, ids, X, F, M, y, hold_idx, train_idx, device)
        res["batch"] = b
        res["steps"] = STEPS
        rows.append(res)
        print({k: round(v, 3) if isinstance(v, float) else v
               for k, v in res.items()}, flush=True)
        pd.DataFrame(rows).to_csv(f"{SP}/ablation_batch.csv", index=False)
    print("done -> ablation_batch.csv", flush=True)


if __name__ == "__main__":
    main()
