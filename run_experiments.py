"""ML441 A4: SOM classifier on the network traffic data we have been working with, compared against our A2 KNN and CART."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent.parent  # project root (scripts live in code/ - in my full repo on local)
SIDES = [10, 20, 30, 40, 50, 60, 70]  # map side length L, M = L * L units
SIGMA_F = [0.5, 1.0, 2.0]  # final neighbourhood width
EPOCHS = [10, 30]  # batch map epochs T
KS = [2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000]  # numbers of clusters
RULES = ["majority", "normalised"]
C = 10

# data (from A2)
df = pd.read_csv(HERE / "data/networkTraffic.csv", na_values="?")
X = df.drop(columns=["attack_cat", "id"])  # target removed; id is a row number
X["service"] = X["service"].fillna("none")  # '?' means no service was identified
y = df["attack_cat"].to_numpy()
nominal = ["proto", "state", "service"]
numeric = [c for c in X.columns if c not in nominal and c not in ["sloss", "dloss", "ct_ftp_cmd"]]


def distance_prep():
    # same preprocessing as the A2 KNN (reused for the SOM). correlated cols already dropped from numeric
    return ColumnTransformer([
        ("cat", OneHotEncoder(min_frequency=100, handle_unknown="infrequent_if_exist", sparse_output=False), nominal),
        ("num", QuantileTransformer(output_distribution="uniform", random_state=1), numeric)])


def tree_prep():
    return ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), nominal)],
                             remainder="passthrough")


def prepare(tr, te, prep):
    prep = prep.fit(X.iloc[tr])
    return prep.transform(X.iloc[tr]).astype(np.float32), prep.transform(X.iloc[te]).astype(np.float32)


# SOM + Ward + labelling, own implementation
def sqdist(P, Q):
    return np.maximum((P * P).sum(1)[:, None] - 2 * P @ Q.T + (Q * Q).sum(1)[None, :], 0)


def lattice(L):
    r = np.array([(i, j) for i in range(L) for j in range(L)], dtype=np.float32)
    return sqdist(r, r)


def bmus(A, W, chunk=8192):
    # best matching unit, distance to it, and second-best unit for every row of A
    first, dist, second = [], [], []
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk]
        rows = np.arange(len(a))
        d = (W * W).sum(1)[None, :] - 2 * a @ W.T  # squared distance minus |a|^2
        b = d.argmin(1)
        first.append(b)
        dist.append(np.sqrt(np.maximum(d[rows, b] + (a * a).sum(1), 0)))
        d[rows, b] = np.inf
        second.append(d.argmin(1))
    return np.concatenate(first), np.concatenate(dist), np.concatenate(second)


def train_som(A, L, sigma_f, epochs):
    # linear initialisation on the two leading principal components, then the batch map
    vals, vecs = np.linalg.eigh(np.cov(A, rowvar=False))
    p1, p2 = vecs[:, -1] * np.sqrt(vals[-1]), vecs[:, -2] * np.sqrt(vals[-2])
    u = np.linspace(-1, 1, L)
    W = np.array([A.mean(0) + a * p1 + b * p2 for a in u for b in u], dtype=np.float32)
    G = lattice(L)
    for t in range(epochs):
        sigma = L / 2 * (sigma_f / (L / 2)) ** (t / max(epochs - 1, 1))  # from L/2 down to sigma_f
        b = bmus(A, W)[0]
        H = np.exp(-G / (2 * sigma ** 2))
        sums = np.stack([np.bincount(b, A[:, j], L * L) for j in range(A.shape[1])], 1)
        num, den = H @ sums, H @ np.bincount(b, minlength=L * L)
        keep = den > 1e-12  # units with no weighted data stay put
        W[keep] = num[keep] / den[keep, None]
    return W


def quality(A, W, L):
    # quantisation error and topographic error (best and second-best units not lattice neighbours)
    b1, d, b2 = bmus(A, W)
    return d.mean(), (lattice(L)[b1, b2] > 1).mean()


def ward(W):
    # Ward clustering of the codebook (Lance-Williams update); returns the merge sequence
    W = W.astype(np.float64)
    D = sqdist(W, W)
    np.fill_diagonal(D, np.inf)
    size = np.ones(len(W))
    merges = []
    for _ in range(len(W) - 1):
        i, j = np.unravel_index(D.argmin(), D.shape)
        new = ((size[i] + size) * D[i] + (size[j] + size) * D[j] - size * D[i, j]) / (size[i] + size[j] + size)
        D[i, :], D[:, i] = new, new
        D[i, i] = np.inf
        D[j, :], D[:, j] = np.inf, np.inf  # cluster j merged into cluster i
        size[i] += size[j]
        merges.append((i, j))
    return merges


def cut(merges, M, K):
    # cluster index (0..K-1) of every unit after the first M - K merges
    owner = np.arange(M)
    for i, j in merges[:M - K]:
        owner[owner == j] = i
    return np.unique(owner, return_inverse=True)[1]


def centroids(W, assign, K):
    cent = np.zeros((K, W.shape[1]))
    np.add.at(cent, assign, W)
    return cent / np.bincount(assign, minlength=K)[:, None]


def davies_bouldin(W, assign, K):
    cent = centroids(W, assign, K)
    s = np.bincount(assign, np.sqrt(((W - cent[assign]) ** 2).sum(1)), K) / np.bincount(assign, minlength=K)
    R = (s[:, None] + s[None, :]) / np.sqrt(sqdist(cent, cent) + np.eye(K))
    np.fill_diagonal(R, -np.inf)
    return R.max(1).mean()


def label_units(W, assign, K, hits, rule):
    # label each cluster from the training hits (units x classes) of its units; one label per unit
    n = np.zeros((K, C))
    np.add.at(n, assign, hits)
    lab = (n / hits.sum(0) if rule == "normalised" else n).argmax(1)
    empty = n.sum(1) == 0  # no hits: label of the nearest labelled cluster
    if empty.any():
        cent = centroids(W, assign, K)
        lab[empty] = lab[~empty][sqdist(cent[empty], cent[~empty]).argmin(1)]
    return lab[assign]


def fit_map(A, y_tr, L, sigma_f, T):
    W = train_som(A, L, sigma_f, T)
    hits = np.zeros((L * L, C))
    np.add.at(hits, (bmus(A, W)[0], y_tr), 1)
    return W, hits


def score(yt, yp):
    return dict(macro_f1=f1_score(yt, yp, average="macro", labels=range(C), zero_division=0),
                bal_acc=balanced_accuracy_score(yt, yp), acc=accuracy_score(yt, yp))


# parallel jobs
def stage1_task(fold, A, V, y_tr, y_va, L, sigma_f, T):
    # one map on one tuning fold: errors and unit-level (K = M) majority-rule scores
    W, hits = fit_map(A, y_tr, L, sigma_f, T)
    lab = label_units(W, np.arange(L * L), L * L, hits, "majority")
    qe, te = quality(V, W, L)
    return dict(fold=fold, L=L, sigma_f=sigma_f, T=T, qe=qe, te=te, **score(y_va, lab[bmus(V, W)[0]]))


def stage2_task(fold, A, V, y_tr, y_va, cfg):
    # the chosen map on one tuning fold: every K and rule, plus the Davies-Bouldin index
    M = cfg["L"] ** 2
    W, hits = fit_map(A, y_tr, cfg["L"], cfg["sigma_f"], cfg["T"])
    bv, merges, rows = bmus(V, W)[0], ward(W), []
    for K in [k for k in KS if k < M] + [M]:
        assign = cut(merges, M, K)
        db = davies_bouldin(W, assign, K) if K < M else np.nan
        for rule in RULES:
            rows.append(dict(fold=fold, K=K, rule=rule, db=db, **score(y_va, label_units(W, assign, K, hits, rule)[bv])))
    return rows


def som_fold(fold, A, V, y_tr, cfg):
    # final SOM on one evaluation fold, with the comparison variants
    M = cfg["L"] ** 2
    W, hits = fit_map(A, y_tr, cfg["L"], cfg["sigma_f"], cfg["T"])
    bv, merges = bmus(V, W)[0], ward(W)
    other = [r for r in RULES if r != cfg["rule"]][0]
    variants = {"som": (cfg["K"], cfg["rule"]), "som_units": (M, cfg["rule"]),
                "som_other_rule": (cfg["K"], other), "som_db": (cfg["K_db"], cfg["rule"])}
    preds = {name: label_units(W, cut(merges, M, K), K, hits, rule)[bv] for name, (K, rule) in variants.items()}
    qe, te = quality(V, W, L=cfg["L"])
    return fold, preds, dict(qe=qe, te=te), dict(W=W, hits=hits, assign=cut(merges, M, cfg["K"]))


def knn_fold(fold, A, V, y_tr):
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="brute", n_jobs=1).fit(A, y_tr)
    return fold, {"knn": clf.predict(V)}, {}, None


def tree_fold(fold, A, V, y_tr):
    clf = DecisionTreeClassifier(min_samples_leaf=5, random_state=1).fit(A, y_tr)
    tests = clf.decision_path(V).sum(1).mean() - 1  # threshold tests per query
    return fold, {"tree": clf.predict(V)}, dict(tests=tests, depth=clf.get_depth()), None


# model selection + stats
def pick_map(s1):
    # take the best mean unit-level F1, then the smallest map within 1 SE of it
    g = s1.groupby(["L", "sigma_f", "T"]).macro_f1.agg(["mean", "std", "count"]).reset_index()
    best = g.loc[g["mean"].idxmax()]
    near = g[g["mean"] >= best["mean"] - best["std"] / np.sqrt(best["count"])]
    row = near[near.L == near.L.min()].sort_values("mean").iloc[-1]
    return dict(L=int(row["L"]), sigma_f=float(row["sigma_f"]), T=int(row["T"]))


def pick_clusters(s2, M):
    # pick the rule with best mean F1 over K < M, then the smallest K within 1 SE of the best.
    # DB only searched up to sqrt(M) since it keeps dropping as clusters approach single units
    g = s2[s2.K < M].groupby(["rule", "K"]).macro_f1.agg(["mean", "std", "count"]).reset_index()
    rule = g.loc[g["mean"].idxmax(), "rule"]
    g = g[g.rule == rule]
    best = g.loc[g["mean"].idxmax()]
    K = int(g[g["mean"] >= best["mean"] - best["std"] / np.sqrt(best["count"])].K.min())
    db = s2[(s2.rule == rule) & (s2.K <= np.sqrt(M))].groupby("K").db.mean()
    return dict(rule=rule, K=K, K_db=int(db.idxmin()))


def holm_print(piv, pairs):
    # paired two-sided Wilcoxon signed-rank tests, Holm step-down adjusted p-values
    ps = sorted((wilcoxon(piv[a], piv[b]).pvalue if (piv[a] != piv[b]).any() else 1.0, a, b) for a, b in pairs)
    running = 0.0
    for k, (p, a, b) in enumerate(ps):
        running = max(running, min(1.0, p * (len(ps) - k)))
        print(f"{a} vs {b}: mean diff {(piv[a] - piv[b]).mean():+.4f}, p={p:.4f}, holm={running:.4f}")


def run_checks():
    # own Ward against scipy, own BMU search against brute force
    from scipy.cluster.hierarchy import fcluster, linkage
    rng = np.random.default_rng(7)
    P, W = rng.normal(size=(300, 5)).astype(np.float32), rng.normal(size=(40, 5)).astype(np.float32)
    merges, Z = ward(P), linkage(P.astype(np.float64), method="ward")
    for K in [2, 5, 10, 50, 150]:
        ct = pd.crosstab(cut(merges, len(P), K), fcluster(Z, K, criterion="maxclust")) > 0
        assert (ct.sum(0) == 1).all() and (ct.sum(1) == 1).all()
    b, d, b2 = bmus(P, W)
    full = np.sqrt(sqdist(P.astype(np.float64), W.astype(np.float64)))
    assert (b == full.argmin(1)).all() and np.allclose(d, full.min(1), atol=1e-4)
    assert (b2 == np.argsort(full, 1)[:, 1]).all()
    print("checks passed")


if __name__ == "__main__":
    run_checks()
    idx = np.arange(len(y))

    # tuning: the assignment 2 subsample (25 percent) and its five stratified folds
    sub = train_test_split(idx, train_size=0.25, stratify=y, random_state=1)[0]
    tune = [(f, *prepare(sub[a], sub[b], distance_prep()), y[sub[a]], y[sub[b]])
            for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=1).split(sub, y[sub]))]
    grid = sorted([(L, s, T) for L in SIDES for s in SIGMA_F for T in EPOCHS], key=lambda g: -g[0] ** 2 * g[2])
    s1 = pd.DataFrame(Parallel(n_jobs=8, verbose=5)(delayed(stage1_task)(*t, *g) for g in grid for t in tune))
    s1.to_csv(HERE / "stage1.csv", index=False)
    cfg = pick_map(s1)
    print("stage 1 choice:", cfg)
    s2 = pd.DataFrame([r for rows in Parallel(n_jobs=8)(delayed(stage2_task)(*t, cfg) for t in tune) for r in rows])
    s2.to_csv(HERE / "stage2.csv", index=False)
    cfg.update(pick_clusters(s2, cfg["L"] ** 2))
    print("stage 2 choice:", cfg)
    json.dump(cfg, open(HERE / "chosen.json", "w"), indent=1)

    # evaluation: stratified ten-fold cross-validation with the assignment 2 folds
    folds = list(StratifiedKFold(10, shuffle=True, random_state=1).split(idx, y))
    jobs = []
    for f, (tr, te) in enumerate(folds):  # heaviest tasks (KNN) first
        A, V = prepare(tr, te, distance_prep())
        jobs += [delayed(knn_fold)(f, A, V, y[tr]), delayed(som_fold)(f, A, V, y[tr], cfg),
                 delayed(tree_fold)(f, *prepare(tr, te, tree_prep()), y[tr])]
    pred, rows = {}, []
    for f, preds, info, extra in Parallel(n_jobs=8, verbose=5)(jobs):
        te = folds[f][1]
        for name, p in preds.items():
            pred.setdefault(name, np.full(len(y), -1))[te] = p
            rows.append(dict(fold=f, model=name, **score(y[te], p), **info))
        if extra is not None and f == 0:
            np.savez(HERE / "map_fold1.npz", L=cfg["L"], **extra)
    res = pd.DataFrame(rows).sort_values(["model", "fold"])
    res.to_csv(HERE / "folds.csv", index=False)
    np.savez(HERE / "predictions.npz", y=y, **pred)

    print(res.groupby("model").agg(["mean", "std"]).drop(columns="fold").round(4).to_string())
    holm_print(res.pivot(index="fold", columns="model", values="macro_f1"),
               [("som", "knn"), ("som", "tree"), ("som", "som_units"), ("som", "som_other_rule")])
