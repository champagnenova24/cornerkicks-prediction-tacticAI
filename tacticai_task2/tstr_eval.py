"""TSTR: train-on-synthetic, test-on-real for the corner shot classifier.

Protocol
  1. For every real TRAINING template (1,940 corners), generate one synthetic
     corner with the diffusion model (w=1, 200 DDIM steps), conditioned on the
     template's real shot label -> synthetic training set with known labels.
  2. Train the Task-2 direct shot classifier (same HGB family/params as the
     guidance check) on: real train (TRTR), synthetic train (TSTR), and
     real+synthetic (augmentation).
  3. Evaluate all on the same REAL holdout corners (485).
F1 thresholds are tuned on a validation slice carved from each model's OWN
training source (so TSTR never touches real data before test time).
Output: tstr_results.csv
"""
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

from diffusion_corners import SP, load_dataset, load_model, denormalize
from diffusion_eval import to_player_frame, real_player_frame, globals_for
from train_eval import CONFIGS

DDIM_STEPS, W = 200, 1.0


def sample_training_set(model, diff, F, M, y, train_idx, device):
    dev = torch.device(device)
    chunks = []
    for s in range(0, len(train_idx), 128):
        sl = train_idx[s:s + 128]
        lab = torch.tensor(y[sl]).to(dev)
        f = torch.tensor(F[sl]).to(dev)
        m = torch.tensor(M[sl]).to(dev)
        z = diff.ddim_sample(model, f, m, lab, steps=DDIM_STEPS,
                             guidance=W, seed=5000 + s)
        chunks.append(z.cpu().numpy())
        if s % 512 == 0:
            print(f"sampled {s + len(sl)}/{len(train_idx)}", flush=True)
    return denormalize(np.concatenate(chunks))


def fit_eval(Xtr, ytr, Xte, yte, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    n_val = max(int(0.15 * len(Xtr)), 50)
    va, tr = perm[:n_val], perm[n_val:]
    clf = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=40, random_state=seed).fit(Xtr[tr], ytr[tr])
    p_va = clf.predict_proba(Xtr[va])[:, 1]
    best_t, best_f = 0.5, -1
    for t in np.linspace(0.05, 0.6, 56):
        f = f1_score(ytr[va], p_va >= t)
        if f > best_f:
            best_t, best_f = t, f
    p = clf.predict_proba(Xte)[:, 1]
    return dict(auc=roc_auc_score(yte, p), f1=f1_score(yte, p >= best_t))


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids, X, F, M, y = load_dataset()
    split = pd.read_csv(f"{SP}/diffusion_split.csv")
    hold_idx = np.where(split.split.to_numpy() == "holdout")[0]
    train_idx = np.where(split.split.to_numpy() == "train")[0]
    model, diff = load_model(device)
    gcols = CONFIGS["vel"][1]

    print(f"generating synthetic training set: {len(train_idx)} corners",
          flush=True)
    Z = sample_training_set(model, diff, F, M, y, train_idx, device)
    dp_syn = to_player_frame(ids, Z, F, M, train_idx, "tstr")
    G_syn = globals_for(dp_syn, gcols)
    y_syn = y[train_idx]  # conditioning labels = the synthetic ground truth

    G_real_tr = globals_for(real_player_frame(ids, X, F, M, train_idx), gcols)
    G_real_te = globals_for(real_player_frame(ids, X, F, M, hold_idx), gcols)
    y_tr, y_te = y[train_idx], y[hold_idx]

    Xr, Xs = G_real_tr.to_numpy(float), G_syn.to_numpy(float)
    Xe = G_real_te.to_numpy(float)

    rows = []
    for name, Xt, yt in [
            ("TRTR (real -> real)", Xr, y_tr),
            ("TSTR (synthetic -> real)", Xs, y_syn),
            ("AUG (real+synthetic -> real)",
             np.vstack([Xr, Xs]), np.concatenate([y_tr, y_syn]))]:
        # 3 seeds to average out classifier/threshold noise
        res = [fit_eval(Xt, yt, Xe, y_te, seed=s) for s in range(3)]
        rows.append(dict(
            setting=name, n_train=len(Xt),
            auc_mean=np.mean([r["auc"] for r in res]),
            auc_std=np.std([r["auc"] for r in res]),
            f1_mean=np.mean([r["f1"] for r in res]),
            f1_std=np.std([r["f1"] for r in res])))
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows).round(4)
    df.to_csv(f"{SP}/tstr_results.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
