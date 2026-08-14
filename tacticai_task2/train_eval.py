"""TacticAI Task 2 on StatsBomb open data.

Reproduces the paper's comparison:
  direct model      P(shot|corner)
  decomposed model  P(shot|corner) = sum_i P(shot|recv=i, corner) P(recv=i|corner)   [Eq. 1]

Feature configurations:
  base     static freeze-frame geometry only
  vel      + per-player velocities estimated from consecutive 360 frames
           (corner frame -> first-touch frame, Hungarian-matched; note these
           are BALL-FLIGHT average velocities, i.e. mildly post-kick info)
  vel+bio  + player height/weight aggregates from Wikidata (team means/max,
           defending GK height; anonymous freeze-frame dots cannot carry
           individual bio, so per-player bio only enters the ground-truth
           receiver diagnostic)

Protocol mirrors the paper: random 80:20 corner split, F1 on the shot class,
mean +- std over 5 seeds.
"""
import numpy as np
import pandas as pd
from paths import PROCESSED
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

RNG = np.random.default_rng
GOAL = np.array([120.0, 40.0])
CORNER = np.array([120.0, 0.0])
PEN_SPOT = np.array([108.0, 40.0])
NEAR_POST, FAR_POST = np.array([120.0, 36.0]), np.array([120.0, 44.0])

CAND_BASE = ["x", "y", "dist_goal", "dist_corner", "dist_pen", "in_box",
             "in_6yd", "teammate", "keeper", "near_opp_dist", "n_opp_3m",
             "n_all_3m", "goal_rank", "dist_np", "dist_fp", "angle_corner"]
CAND_VEL = CAND_BASE + ["vx", "vy", "speed", "v_goal", "has_vel"]

GCOLS_BASE = ["n_att", "n_def", "n_att_box", "n_def_box", "box_diff",
              "n_att_6yd", "n_def_6yd", "short_opt", "min_att_dg",
              "mean_att_dg", "min_def_dg", "att_cx", "att_cy", "att_spread",
              "def_cx", "gk_x", "gk_y", "mean_att_press", "min_att_press"] + \
             [f"a{k}_{s}" for k in range(3) for s in ("dg", "y", "press")]
GVEL = ["att_mean_speed", "att_max_speed", "att_mean_vgoal", "n_att_fast",
        "def_mean_speed", "def_mean_vgoal", "vel_avail"] + \
       [f"a{k}_{s}" for k in range(3) for s in ("speed", "vgoal")]
GBIO = ["att_h_mean", "att_h_max", "def_h_mean", "def_h_max", "def_gk_h",
        "att_w_mean", "def_w_mean"]

# delivery attributes off the corner pass event.
# NOTE: technique and height describe the delivery AFTER it is struck, so
# configs using GDELIV are retrospective, not pre-kick predictors.
# body_part (foot) is knowable before the kick -> GPREKICK is leakage-safe.
GDELIV = ["tech_in", "tech_out", "tech_straight", "tech_missing",
          "h_high", "h_ground", "h_low", "foot_right"]
GPREKICK = ["foot_right"]

CONFIGS = {
    "base": (CAND_BASE, GCOLS_BASE, False),
    "vel": (CAND_VEL, GCOLS_BASE + GVEL, False),
    "vel+bio": (CAND_VEL, GCOLS_BASE + GVEL + GBIO, True),
    "vel+prekick": (CAND_VEL, GCOLS_BASE + GVEL + GPREKICK, False),
    "vel+delivery": (CAND_VEL, GCOLS_BASE + GVEL + GDELIV, False),
    "vel+bio+delivery": (CAND_VEL, GCOLS_BASE + GVEL + GBIO + GDELIV, True),
}


def build_features(dc, dp):
    """Per-candidate features and per-corner global features."""
    dp = dp.copy()
    xy = dp[["x", "y"]].to_numpy()
    dp["dist_goal"] = np.linalg.norm(xy - GOAL, axis=1)
    dp["dist_corner"] = np.linalg.norm(xy - CORNER, axis=1)
    dp["dist_pen"] = np.linalg.norm(xy - PEN_SPOT, axis=1)
    dp["in_box"] = ((dp.x >= 102) & (dp.y >= 18) & (dp.y <= 62)).astype(float)
    dp["in_6yd"] = ((dp.x >= 114) & (dp.y >= 30) & (dp.y <= 50)).astype(float)
    dp["dist_np"] = np.linalg.norm(xy - NEAR_POST, axis=1)
    dp["dist_fp"] = np.linalg.norm(xy - FAR_POST, axis=1)
    dp["angle_corner"] = np.arctan2(dp.y - 0.0, 120.0 - dp.x)
    dp["speed"] = np.hypot(dp.vx, dp.vy)
    to_goal = GOAL - xy
    norm = np.linalg.norm(to_goal, axis=1)
    norm[norm < 1e-6] = 1.0
    dp["v_goal"] = (to_goal[:, 0] * dp.vx + to_goal[:, 1] * dp.vy) / norm
    dp["has_vel"] = dp.has_vel.astype(float)

    near_opp, n_opp3, n_all3, grank = [], [], [], []
    for cid, g in dp.groupby("corner_id", sort=False):
        P = g[["x", "y"]].to_numpy()
        team = g["teammate"].to_numpy()
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        opp = team[:, None] != team[None, :]
        d_opp = np.where(opp, D, np.inf)
        near_opp.append(d_opp.min(axis=1))
        n_opp3.append((d_opp < 3.0).sum(axis=1))
        n_all3.append((D < 3.0).sum(axis=1))
        dg = g["dist_goal"].to_numpy()
        r = np.empty(len(g))
        for side in (True, False):
            m = team == side
            r[m] = dg[m].argsort().argsort() / max(m.sum() - 1, 1)
        grank.append(r)
    dp["near_opp_dist"] = np.clip(np.concatenate(near_opp), 0, 30)
    dp["n_opp_3m"] = np.concatenate(n_opp3)
    dp["n_all_3m"] = np.concatenate(n_all3)
    dp["goal_rank"] = np.concatenate(grank)

    # ---- global per-corner features ----
    rows = []
    for cid, g in dp.groupby("corner_id", sort=False):
        att = g[g.teammate & ~g.actor]
        dfd = g[~g.teammate]
        dfd_field = dfd[~dfd.keeper]
        gk = dfd[dfd.keeper]
        row = dict(corner_id=cid)
        row["n_att"] = len(att)
        row["n_def"] = len(dfd_field)
        row["n_att_box"] = att.in_box.sum()
        row["n_def_box"] = dfd_field.in_box.sum()
        row["box_diff"] = row["n_att_box"] - row["n_def_box"]
        row["n_att_6yd"] = att.in_6yd.sum()
        row["n_def_6yd"] = dfd_field.in_6yd.sum()
        row["short_opt"] = (att.dist_corner < 15).sum()
        row["min_att_dg"] = att.dist_goal.min() if len(att) else 60.0
        row["mean_att_dg"] = att.dist_goal.mean() if len(att) else 60.0
        row["min_def_dg"] = dfd_field.dist_goal.min() if len(dfd_field) else 60.0
        row["att_cx"] = att.x.mean() if len(att) else 90.0
        row["att_cy"] = att.y.mean() if len(att) else 40.0
        row["att_spread"] = att[["x", "y"]].std().mean() if len(att) > 1 else 0.0
        row["def_cx"] = dfd_field.x.mean() if len(dfd_field) else 100.0
        row["gk_x"] = gk.x.iloc[0] if len(gk) else 119.0
        row["gk_y"] = gk.y.iloc[0] if len(gk) else 40.0
        row["mean_att_press"] = att.near_opp_dist.mean() if len(att) else 15.0
        row["min_att_press"] = att.near_opp_dist.min() if len(att) else 15.0
        # velocity aggregates (flight-average velocities)
        att_v = att[att.has_vel > 0]
        def_v = dfd_field[dfd_field.has_vel > 0]
        row["vel_avail"] = float(len(att_v) + len(def_v) > 0)
        row["att_mean_speed"] = att_v.speed.mean() if len(att_v) else 0.0
        row["att_max_speed"] = att_v.speed.max() if len(att_v) else 0.0
        row["att_mean_vgoal"] = att_v.v_goal.mean() if len(att_v) else 0.0
        row["n_att_fast"] = (att_v.speed > 3.0).sum()
        row["def_mean_speed"] = def_v.speed.mean() if len(def_v) else 0.0
        row["def_mean_vgoal"] = def_v.v_goal.mean() if len(def_v) else 0.0
        # pooled "pseudo-attention": 3 attackers nearest to goal
        srt = att.sort_values("dist_goal")
        for k in range(3):
            if k < len(srt):
                r = srt.iloc[k]
                row[f"a{k}_dg"] = r.dist_goal
                row[f"a{k}_y"] = r.y
                row[f"a{k}_press"] = r.near_opp_dist
                row[f"a{k}_speed"] = r.speed if r.has_vel else 0.0
                row[f"a{k}_vgoal"] = r.v_goal if r.has_vel else 0.0
            else:
                row[f"a{k}_dg"], row[f"a{k}_y"], row[f"a{k}_press"] = 60.0, 40.0, 15.0
                row[f"a{k}_speed"], row[f"a{k}_vgoal"] = 0.0, 0.0
        rows.append(row)
    dg_global = pd.DataFrame(rows).set_index("corner_id")
    return dp, dg_global


def add_bio_features(dg_global, dc, sp):
    """Team height/weight aggregates + defending GK height per corner."""
    bio = pd.read_parquet(f"{sp}/player_bio.parquet").set_index("player_id")
    onp = pd.read_parquet(f"{sp}/onpitch.parquet").set_index("corner_id")
    matches = pd.read_parquet(f"{PROCESSED}/matches.parquet")
    gender = matches.set_index("match_id")["gender"]
    mid_gender = dc.set_index("corner_id")["match_id"].map(
        lambda m: gender.get(int(m), "male"))

    h_all, w_all = bio.height_cm, bio.weight_kg

    def agg(cids, id_lists, gk_ids, genders):
        out = {c: {} for c in cids}
        for cid, ids, gkid, gen in zip(cids, id_lists, gk_ids, genders):
            hs = h_all.reindex(ids).dropna()
            ws = w_all.reindex(ids).dropna()
            d = out[cid]
            d["h_mean"] = hs.mean() if len(hs) else np.nan
            d["h_max"] = hs.max() if len(hs) else np.nan
            d["w_mean"] = ws.mean() if len(ws) else np.nan
            d["gk_h"] = h_all.get(gkid, np.nan) if gkid is not None else np.nan
        return out

    cids = [c for c in dg_global.index if c in onp.index]
    sub = onp.loc[cids]
    gens = mid_gender.reindex(cids)
    att = agg(cids, sub.att_players, sub.att_gk, gens)
    dfd = agg(cids, sub.def_players, sub.def_gk, gens)
    for col, src, key in [("att_h_mean", att, "h_mean"), ("att_h_max", att, "h_max"),
                          ("att_w_mean", att, "w_mean"),
                          ("def_h_mean", dfd, "h_mean"), ("def_h_max", dfd, "h_max"),
                          ("def_w_mean", dfd, "w_mean"), ("def_gk_h", dfd, "gk_h")]:
        s = pd.Series({c: src[c][key] for c in cids})
        dg_global[col] = s.reindex(dg_global.index)
    # impute remaining NaNs per gender (height) / overall (weight)
    gen_full = mid_gender.reindex(dg_global.index)
    for col in GBIO:
        for g in ("male", "female"):
            m = (gen_full == g) & dg_global[col].isna()
            fill = dg_global.loc[gen_full == g, col].mean()
            dg_global.loc[m, col] = fill
        dg_global[col] = dg_global[col].fillna(dg_global[col].mean())

    # receiver bio (for the GT-receiver diagnostic only)
    rb = dc.set_index("corner_id")["receiver_player_id"]
    dg_global["recv_h"] = h_all.reindex(rb).to_numpy()
    dg_global["recv_w"] = w_all.reindex(rb).to_numpy()
    for col in ("recv_h", "recv_w"):
        for g in ("male", "female"):
            m = (gen_full == g) & dg_global[col].isna()
            dg_global.loc[m, col] = dg_global.loc[gen_full == g, col].mean()
        dg_global[col] = dg_global[col].fillna(dg_global[col].mean())
    return dg_global


def add_delivery_features(dg_global, sp):
    """One-hot technique / height / body_part from the corner pass event."""
    d = pd.read_parquet(f"{sp}/corner_delivery.parquet").set_index("corner_id")
    d = d.reindex(dg_global.index)
    dg_global["tech_in"] = (d.technique == "Inswinging").astype(float)
    dg_global["tech_out"] = (d.technique == "Outswinging").astype(float)
    dg_global["tech_straight"] = (d.technique == "Straight").astype(float)
    dg_global["tech_missing"] = (~d.technique.isin(
        ["Inswinging", "Outswinging", "Straight"])).astype(float)
    dg_global["h_high"] = (d.height == "High Pass").astype(float)
    dg_global["h_ground"] = (d.height == "Ground Pass").astype(float)
    dg_global["h_low"] = (d.height == "Low Pass").astype(float)
    dg_global["foot_right"] = (d.body_part == "Right Foot").astype(float)
    return dg_global


# ---------------- receiver model: softmax choice over candidates -------------
class ReceiverNet:
    def __init__(self, n_feat, hidden=16, l2=1e-3, seed=0):
        r = RNG(seed)
        self.h = hidden
        self.l2 = l2
        if hidden:
            self.W1 = r.normal(0, 1 / np.sqrt(n_feat), (n_feat, hidden))
            self.b1 = np.zeros(hidden)
            self.W2 = r.normal(0, 1 / np.sqrt(hidden), (hidden, 1))
        else:
            self.W1 = r.normal(0, 1 / np.sqrt(n_feat), (n_feat, 1))

    def scores(self, X):
        if self.h:
            Z = np.tanh(X @ self.W1 + self.b1)
            return (Z @ self.W2).squeeze(-1), Z
        return (X @ self.W1).squeeze(-1), None

    def fit(self, X, mask, target, epochs=400, lr=0.05):
        params = ["W1", "b1", "W2"] if self.h else ["W1"]
        m_t = {p: 0.0 for p in params}
        v_t = {p: 0.0 for p in params}
        N = len(X)
        for t in range(1, epochs + 1):
            s, Z = self.scores(X)
            s = np.where(mask, s, -1e9)
            s -= s.max(axis=1, keepdims=True)
            p = np.exp(s) * mask
            p /= p.sum(axis=1, keepdims=True)
            g = p.copy()
            g[np.arange(N), target] -= 1.0
            g /= N
            grads = {}
            if self.h:
                dZ = g[..., None] * self.W2[None, None, :, 0]
                dpre = dZ * (1 - Z ** 2)
                grads["W2"] = np.einsum("nch,nc->h", Z, g)[:, None] + self.l2 * self.W2
                grads["b1"] = dpre.sum((0, 1))
                grads["W1"] = np.einsum("ncf,nch->fh", X, dpre) + self.l2 * self.W1
            else:
                grads["W1"] = np.einsum("ncf,nc->f", X, g)[:, None] + self.l2 * self.W1
            for pname in params:
                m_t[pname] = 0.9 * m_t[pname] + 0.1 * grads[pname]
                v_t[pname] = 0.999 * v_t[pname] + 0.001 * grads[pname] ** 2
                mh = m_t[pname] / (1 - 0.9 ** t)
                vh = v_t[pname] / (1 - 0.999 ** t)
                setattr(self, pname, getattr(self, pname) - lr * mh / (np.sqrt(vh) + 1e-8))
        return self

    def predict_proba(self, X, mask):
        s, _ = self.scores(X)
        s = np.where(mask, s, -1e9)
        s -= s.max(axis=1, keepdims=True)
        p = np.exp(s) * mask
        return p / p.sum(axis=1, keepdims=True)


def pad_candidates(groups, scaler):
    C = max(len(g[0]) for g in groups)
    F = groups[0][0].shape[1]
    X = np.zeros((len(groups), C, F))
    mask = np.zeros((len(groups), C), dtype=bool)
    tgt = np.full(len(groups), -1)
    for i, (feats, ridx) in enumerate(groups):
        n = len(feats)
        X[i, :n] = scaler.transform(feats)
        mask[i, :n] = True
        tgt[i] = ridx
    return X, mask, tgt


def make_groups(dp_f, corner_ids, cand_feats):
    out, order = [], []
    by_c = dict(tuple(dp_f[~dp_f.actor].groupby("corner_id", sort=False)))
    for cid in corner_ids:
        g = by_c.get(cid)
        if g is None or len(g) < 2:
            continue
        feats = g[cand_feats].to_numpy(dtype=float)
        ridx = int(np.argmax(g.is_receiver.to_numpy())) if g.is_receiver.any() else -1
        out.append((feats, ridx))
        order.append(cid)
    return out, order


def run_seed(seed, config, dc, dp_f, dg_global, results):
    cand_feats, gcols, use_recv_bio = CONFIGS[config]
    r = RNG(seed)
    ids = dc.corner_id.to_numpy()
    perm = r.permutation(len(ids))
    n_test = int(0.2 * len(ids))
    n_val = int(0.1 * len(ids))
    test_ids = set(ids[perm[:n_test]])
    val_ids = set(ids[perm[n_test:n_test + n_val]])
    train_ids = set(ids[perm[n_test + n_val:]])

    dc_i = dc.set_index("corner_id")
    valid_recv = set(dc[(dc.receiver_found) & (dc.receiver_match_dist <= 10)].corner_id)

    # ---------- receiver model ----------
    gtr, _ = make_groups(dp_f, [c for c in train_ids if c in valid_recv], cand_feats)
    gte, _ = make_groups(dp_f, [c for c in test_ids if c in valid_recv], cand_feats)
    scaler = StandardScaler().fit(np.vstack([g[0] for g in gtr]))
    Xtr, mtr, ttr = pad_candidates(gtr, scaler)
    Xte, mte, tte = pad_candidates(gte, scaler)
    nets = [ReceiverNet(len(cand_feats), hidden=16, seed=seed * 10 + k)
            .fit(Xtr, mtr, ttr) for k in range(3)]

    def recv_proba(X, mask):
        return np.mean([n.predict_proba(X, mask) for n in nets], axis=0)

    pte = recv_proba(Xte, mte)
    top1 = (pte.argmax(1) == tte).mean()
    top3 = np.mean([t in np.argsort(-p)[:3] for p, t in zip(pte, tte)])

    # ---------- conditional shot model ----------
    def cond_rows(corner_ids, with_bio=False):
        rows, ys = [], []
        recv = dp_f[dp_f.is_receiver & ~dp_f.actor].set_index("corner_id")
        cols = gcols + (["recv_h", "recv_w"] if with_bio else [])
        for cid in corner_ids:
            if cid not in recv.index or cid not in valid_recv:
                continue
            rrow = recv.loc[cid]
            if isinstance(rrow, pd.DataFrame):
                rrow = rrow.iloc[0]
            rows.append(np.concatenate([
                dg_global.loc[cid, cols].to_numpy(dtype=float),
                rrow[cand_feats].to_numpy(dtype=float)]))
            ys.append(dc_i.loc[cid, "shot"])
        return np.array(rows), np.array(ys)

    def make_hgb():
        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
            min_samples_leaf=40, early_stopping=True, validation_fraction=0.15,
            random_state=seed)

    Xc_tr, yc_tr = cond_rows(train_ids)
    cond = make_hgb().fit(Xc_tr, yc_tr)

    # ---------- decomposed P(shot|corner) via Eq. 1 ----------
    def eq1_probs(corner_ids, uniform=False):
        groups, order = make_groups(dp_f, corner_ids, cand_feats)
        X, mask, _ = pad_candidates(groups, scaler)
        p_recv = recv_proba(X, mask)
        out = {}
        for i, cid in enumerate(order):
            feats = groups[i][0]
            gvec = dg_global.loc[cid, gcols].to_numpy(dtype=float)
            Z = np.hstack([np.tile(gvec, (len(feats), 1)), feats])
            p_shot_given = cond.predict_proba(Z)[:, 1]
            w = (np.full(len(feats), 1.0 / len(feats)) if uniform
                 else p_recv[i, :len(feats)])
            out[cid] = float((w * p_shot_given).sum())
        return out

    # ---------- direct P(shot|corner) ----------
    def direct_matrix(corner_ids):
        cids = [c for c in corner_ids if c in dg_global.index]
        X = dg_global.loc[cids, gcols].to_numpy(dtype=float)
        y = dc_i.loc[cids, "shot"].to_numpy()
        return cids, X, y

    tr_c, Xd_tr, yd_tr = direct_matrix(train_ids)
    direct = make_hgb().fit(Xd_tr, yd_tr)

    def tune_threshold(y, p):
        best_t, best_f = 0.5, -1
        for t in np.linspace(0.05, 0.6, 56):
            f = f1_score(y, p >= t)
            if f > best_f:
                best_t, best_f = t, f
        return best_t

    va_c, Xd_va, yd_va = direct_matrix(val_ids)
    te_c, Xd_te, yd_te = direct_matrix(test_ids)

    p_dir_va = direct.predict_proba(Xd_va)[:, 1]
    p_dir_te = direct.predict_proba(Xd_te)[:, 1]
    eq1_va = eq1_probs(val_ids)
    eq1_te = eq1_probs(test_ids)
    eq1_te_unif = eq1_probs(test_ids, uniform=True)

    # diagnostic upper bound: conditional model fed the ground-truth receiver
    Xc_te, yc_te = cond_rows(test_ids)
    Xc_va, yc_va = cond_rows(val_ids)
    p_gt_te = cond.predict_proba(Xc_te)[:, 1]
    p_gt_va = cond.predict_proba(Xc_va)[:, 1]

    res = dict(config=config, seed=seed,
               recv_top1=top1, recv_top3=top3, n_test_recv=len(tte))

    if use_recv_bio:
        cond_b = make_hgb().fit(*cond_rows(train_ids, with_bio=True))
        Xb_te, yb_te = cond_rows(test_ids, with_bio=True)
        res["auc_gt_recv_bio"] = roc_auc_score(
            yb_te, cond_b.predict_proba(Xb_te)[:, 1])

    p_eq1_va = np.array([eq1_va[c] for c in va_c if c in eq1_va])
    y_eq1_va = dc_i.loc[[c for c in va_c if c in eq1_va], "shot"].to_numpy()
    p_eq1_te = np.array([eq1_te[c] for c in te_c if c in eq1_te])
    y_eq1_te = dc_i.loc[[c for c in te_c if c in eq1_te], "shot"].to_numpy()

    t_dir = tune_threshold(yd_va, p_dir_va)
    t_eq1 = tune_threshold(y_eq1_va, p_eq1_va)

    res.update(
        f1_direct=f1_score(yd_te, p_dir_te >= t_dir),
        f1_eq1=f1_score(y_eq1_te, p_eq1_te >= t_eq1),
        auc_direct=roc_auc_score(yd_te, p_dir_te),
        auc_eq1=roc_auc_score(y_eq1_te, p_eq1_te),
        auc_eq1_uniform=roc_auc_score(
            y_eq1_te, [eq1_te_unif[c] for c in te_c if c in eq1_te_unif]),
        auc_gt_receiver=roc_auc_score(yc_te, p_gt_te),
        f1_gt_receiver=f1_score(yc_te, p_gt_te >= tune_threshold(yc_va, p_gt_va)),
        f1_allpos=f1_score(yd_te, np.ones_like(yd_te)),
        prec_eq1=precision_score(y_eq1_te, p_eq1_te >= t_eq1),
        rec_eq1=recall_score(y_eq1_te, p_eq1_te >= t_eq1),
        n_test=len(yd_te), shot_rate_test=float(np.mean(yd_te)),
    )
    results.append(res)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()},
          flush=True)


def main():
    import os
    sp = os.path.dirname(os.path.abspath(__file__))
    dc = pd.read_parquet(f"{sp}/corners.parquet")
    dp = pd.read_parquet(f"{sp}/corner_players.parquet")
    ok = dc[(dc.n_att_visible >= 5) & (dc.n_def_visible >= 5)]
    print(f"corners kept after visibility filter: {len(ok)}/{len(dc)} "
          f"(shot rate {ok.shot.mean():.3f})")
    dp = dp[dp.corner_id.isin(ok.corner_id)]
    dp_f, dg_global = build_features(ok, dp)

    configs = ["base", "vel"]
    if os.path.exists(f"{sp}/player_bio.parquet"):
        dg_global = add_bio_features(dg_global, ok, sp)
        configs.append("vel+bio")
    else:
        print("player_bio.parquet not found -> skipping vel+bio config")
    if os.path.exists(f"{sp}/corner_delivery.parquet"):
        dg_global = add_delivery_features(dg_global, sp)
        configs += ["vel+prekick", "vel+delivery"]
        if "vel+bio" in configs:
            configs.append("vel+bio+delivery")
    else:
        print("corner_delivery.parquet not found -> skipping delivery configs")

    results = []
    for config in configs:
        for seed in range(5):
            run_seed(seed, config, ok, dp_f, dg_global, results)
    df = pd.DataFrame(results)
    df.to_csv(f"{sp}/results_seeds.csv", index=False)
    print("\n=== summary per config (mean +- std over seeds) ===")
    summary = df.drop(columns="seed").groupby("config").agg(["mean", "std"]).T
    print(summary.round(3))
    summary.to_csv(f"{sp}/results_summary.csv")


if __name__ == "__main__":
    main()
