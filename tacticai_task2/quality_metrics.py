"""Extended quality metrics for synthetic corner kicks.

Beyond the discriminator test, this evaluates the generator along five axes
that fail independently:

  1. Distribution match  : MMD^2 (RBF) on pairwise-distance descriptors with a
                           permutation p-value, overall and per pair type;
                           plus a 1-NN two-sample test (0.5 = indistinguishable)
  2. Fidelity vs diversity: precision / recall (Kynkaanniemi et al.) and
                           density / coverage (Naeem et al.), k = 5
  3. Memorization        : Chamfer distance from each corner to its nearest
                           TRAINING corner; synthetic vs real-holdout baseline
                           (much smaller => the model is copying)
  4. Physical plausibility: constraint violation rates, scored identically on
                           real corners so the base rate is visible
  5. Spatial structure   : pair-correlation g(r) curves per pair type (figure)

Uses the already-generated `synthetic_corners.parquet` (variant real_label,
sampled at guidance w = 1).

Outputs: quality_metrics.csv, quality_pcf.csv, quality_metrics.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = "/private/tmp/claude-501/-Users-daniel-Desktop-open-data-master/149301ad-df98-430c-b299-a436270c69f7/scratchpad"
RNG = np.random.default_rng(0)
BINS = np.linspace(0, 40, 17)          # 16 bins for distance histograms
RMAX, NR = 30.0, 30                    # pair-correlation range / resolution
K_NN = 5


# ------------------------------------------------------------------ loading
def corner_sets(dp):
    """{corner_id: (attacker xy array, defender xy array, all xy, teams)}"""
    out = {}
    for cid, g in dp.groupby("corner_id", sort=False):
        att = g[g.teammate].loc[:, ["x", "y"]].to_numpy(float)
        dfd = g[~g.teammate].loc[:, ["x", "y"]].to_numpy(float)
        out[cid] = (att, dfd, g[["x", "y"]].to_numpy(float),
                    g["teammate"].to_numpy(bool), g)
    return out


def pair_dists(A, B=None):
    if B is None:
        if len(A) < 2:
            return np.array([])
        D = np.linalg.norm(A[:, None] - A[None], axis=2)
        return D[np.triu_indices(len(A), 1)]
    if len(A) == 0 or len(B) == 0:
        return np.array([])
    return np.linalg.norm(A[:, None] - B[None], axis=2).ravel()


def descriptor(att, dfd):
    """Permutation-invariant descriptor: normalized histograms of att-att,
    def-def and att-def pairwise distances (48 dims)."""
    parts = []
    for d in (pair_dists(att), pair_dists(dfd), pair_dists(att, dfd)):
        h, _ = np.histogram(d, bins=BINS)
        s = h.sum()
        parts.append(h / s if s else np.zeros(len(BINS) - 1))
    return np.concatenate(parts)


def build(dp):
    sets = corner_sets(dp)
    ids = list(sets)
    Dsc = np.stack([descriptor(sets[c][0], sets[c][1]) for c in ids])
    return ids, Dsc, sets


# ------------------------------------------------------------------ metrics
def mmd2_and_p(Xa, Xb, n_perm=1000, seed=0):
    """Unbiased MMD^2 with RBF kernel (median heuristic) + permutation p."""
    Z = np.vstack([Xa, Xb])
    d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / med if med > 0 else 1.0
    Kz = np.exp(-gamma * d2)
    na = len(Xa)

    def stat(idx):
        ia, ib = idx[:na], idx[na:]
        Kaa = Kz[np.ix_(ia, ia)]
        Kbb = Kz[np.ix_(ib, ib)]
        Kab = Kz[np.ix_(ia, ib)]
        n, m = len(ia), len(ib)
        return ((Kaa.sum() - np.trace(Kaa)) / (n * (n - 1))
                + (Kbb.sum() - np.trace(Kbb)) / (m * (m - 1))
                - 2 * Kab.mean())

    base = np.arange(len(Z))
    obs = stat(base)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(base)) for _ in range(n_perm)])
    return obs, float((null >= obs).mean() + 1 / (n_perm + 1))


def knn_radii(X, k):
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, k - 1]


def prdc(Xr, Xf, k=K_NN):
    """Precision, recall, density, coverage (standardized space)."""
    mu, sd = Xr.mean(0), Xr.std(0) + 1e-9
    Xr, Xf = (Xr - mu) / sd, (Xf - mu) / sd
    rr, rf = knn_radii(Xr, k), knn_radii(Xf, k)
    Drf = np.linalg.norm(Xr[:, None] - Xf[None], axis=2)   # real x fake
    precision = float((Drf <= rr[:, None]).any(0).mean())
    recall = float((Drf <= rf[None, :]).any(1).mean())
    density = float((Drf <= rr[:, None]).sum(0).mean() / k)
    coverage = float((Drf.min(1) <= rr).mean())
    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)


def one_nn_accuracy(Xa, Xb):
    """Leave-one-out 1-NN two-sample test; 0.5 = indistinguishable."""
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    Z = np.vstack([(Xa - mu) / sd, (Xb - mu) / sd])
    lab = np.concatenate([np.ones(len(Xa)), np.zeros(len(Xb))])
    D = np.linalg.norm(Z[:, None] - Z[None], axis=2)
    np.fill_diagonal(D, np.inf)
    return float((lab[D.argmin(1)] == lab).mean())


def chamfer(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.inf
    D = np.linalg.norm(a[:, None] - b[None], axis=2)
    return 0.5 * (D.min(1).mean() + D.min(0).mean())


def nearest_train_distance(sets_q, ids_q, sets_tr, ids_tr):
    """Team-aware Chamfer distance to the closest training corner."""
    tr = [(sets_tr[c][0], sets_tr[c][1]) for c in ids_tr]
    out = []
    for c in ids_q:
        a, d = sets_q[c][0], sets_q[c][1]
        best = min(chamfer(a, ta) + chamfer(d, td) for ta, td in tr)
        out.append(best / 2)
    return np.array(out)


def violations(sets, ids):
    """Physical / rule plausibility checks (rates per corner)."""
    rows = []
    for c in ids:
        att, dfd, allxy, team, g = sets[c]
        gk = g[g.keeper]
        sp = np.hypot(g.vx.to_numpy(float), g.vy.to_numpy(float))
        D = np.linalg.norm(allxy[:, None] - allxy[None], axis=2)
        np.fill_diagonal(D, np.inf)
        rows.append(dict(
            out_of_bounds=float(((allxy[:, 0] < 0) | (allxy[:, 0] > 120) |
                                 (allxy[:, 1] < 0) | (allxy[:, 1] > 80)).any()),
            behind_goal=float((allxy[:, 0] > 120).any()),
            gk_far_from_goal=float(
                (np.linalg.norm(gk[["x", "y"]].to_numpy(float)
                                - np.array([120.0, 40.0]), axis=1) > 12).any()
                if len(gk) else 0.0),
            overlap_lt_0p5m=float((D < 0.5).any()),
            speed_gt_9m_s=float((sp > 9.0).any()),
            outside_final_third=float((allxy[:, 0] < 80).mean()),
        ))
    return pd.DataFrame(rows).mean()


def pcf(sets, ids, kind):
    """Pair-correlation-style radial density g(r), averaged over corners."""
    edges = np.linspace(0, RMAX, NR + 1)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    acc = np.zeros(NR)
    n = 0
    for c in ids:
        att, dfd, _, _, _ = sets[c]
        d = (pair_dists(att) if kind == "att-att" else
             pair_dists(dfd) if kind == "def-def" else pair_dists(att, dfd))
        if len(d) == 0:
            continue
        h, _ = np.histogram(d, bins=edges)
        acc += h / (area * len(d))
        n += 1
    return edges[:-1] + np.diff(edges) / 2, acc / max(n, 1)


# ------------------------------------------------------------------ main
def main():
    dp_all = pd.read_parquet(f"{SP}/corner_players.parquet")
    dc = pd.read_parquet(f"{SP}/corners.parquet")
    ok = dc[(dc.n_att_visible >= 5) & (dc.n_def_visible >= 5)].corner_id
    dp_all = dp_all[dp_all.corner_id.isin(ok)]
    split = pd.read_csv(f"{SP}/diffusion_split.csv")
    hold_ids = set(split[split.split == "holdout"].corner_id)
    train_ids = set(split[split.split == "train"].corner_id)

    syn = pd.read_parquet(f"{SP}/synthetic_corners.parquet")
    syn = syn[syn.variant == "real_label"].copy()

    ids_r, Xr, sets_r = build(dp_all[dp_all.corner_id.isin(hold_ids)])
    ids_t, Xt, sets_t = build(dp_all[dp_all.corner_id.isin(train_ids)])
    ids_s, Xs, sets_s = build(syn)
    print(f"real holdout {len(ids_r)} | train {len(ids_t)} | "
          f"synthetic {len(ids_s)}", flush=True)

    res = {}

    # 1. distribution match -------------------------------------------------
    mmd, p = mmd2_and_p(Xr, Xs)
    res["mmd2_all"], res["mmd2_all_p"] = mmd, p
    for j, name in enumerate(["att-att", "def-def", "att-def"]):
        sl = slice(j * (len(BINS) - 1), (j + 1) * (len(BINS) - 1))
        m, pp = mmd2_and_p(Xr[:, sl], Xs[:, sl])
        res[f"mmd2_{name}"], res[f"mmd2_{name}_p"] = m, pp
    # real-vs-real control: split the holdout in half
    half = RNG.permutation(len(Xr))
    m0, p0 = mmd2_and_p(Xr[half[:len(Xr) // 2]], Xr[half[len(Xr) // 2:]])
    res["mmd2_real_vs_real"], res["mmd2_real_vs_real_p"] = m0, p0
    res["nn1_real_vs_syn"] = one_nn_accuracy(Xr, Xs)
    res["nn1_real_vs_real"] = one_nn_accuracy(Xr[half[:len(Xr) // 2]],
                                              Xr[half[len(Xr) // 2:]])

    # 2. fidelity vs diversity ---------------------------------------------
    res.update({f"{k}": v for k, v in prdc(Xr, Xs).items()})
    ctrl = prdc(Xr[half[:len(Xr) // 2]], Xr[half[len(Xr) // 2:]])
    res.update({f"ctrl_{k}": v for k, v in ctrl.items()})

    # 3. memorization -------------------------------------------------------
    print("computing nearest-training distances...", flush=True)
    d_syn = nearest_train_distance(sets_s, ids_s, sets_t, ids_t)
    d_real = nearest_train_distance(sets_r, ids_r, sets_t, ids_t)
    res["nn_train_dist_syn_median"] = float(np.median(d_syn))
    res["nn_train_dist_real_median"] = float(np.median(d_real))
    res["nn_train_dist_ratio"] = float(np.median(d_syn) / np.median(d_real))

    # 4. plausibility -------------------------------------------------------
    v_r, v_s = violations(sets_r, ids_r), violations(sets_s, ids_s)
    for k in v_r.index:
        res[f"viol_{k}_real"] = float(v_r[k])
        res[f"viol_{k}_syn"] = float(v_s[k])

    out = pd.Series(res).round(4)
    out.to_csv(f"{SP}/quality_metrics.csv")
    print("\n=== quality metrics ===")
    print(out.to_string(), flush=True)

    # 5. spatial structure figure ------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    pcf_rows = []
    for ax, kind in zip(axes[:3], ["att-att", "def-def", "att-def"]):
        r, gr = pcf(sets_r, ids_r, kind)
        _, gs = pcf(sets_s, ids_s, kind)
        ax.plot(r, gr, label="real holdout", lw=2)
        ax.plot(r, gs, label="synthetic", lw=2, ls="--")
        ax.set_title(f"pair-correlation g(r): {kind}", fontsize=10)
        ax.set_xlabel("distance r (m)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        pcf_rows.append(pd.DataFrame({"r": r, "kind": kind,
                                      "real": gr, "synthetic": gs}))
    axes[3].hist(d_real, bins=30, alpha=0.6, label="real holdout", density=True)
    axes[3].hist(d_syn, bins=30, alpha=0.6, label="synthetic", density=True)
    axes[3].set_title("distance to nearest TRAINING corner", fontsize=10)
    axes[3].set_xlabel("Chamfer distance (m)")
    axes[3].legend(fontsize=8)
    fig.suptitle("Synthetic corner quality: spatial structure and memorization"
                 " check")
    fig.tight_layout()
    fig.savefig(f"{SP}/quality_metrics.png", dpi=150)
    pd.concat(pcf_rows).to_csv(f"{SP}/quality_pcf.csv", index=False)
    print("saved quality_metrics.png / quality_pcf.csv", flush=True)


if __name__ == "__main__":
    main()
