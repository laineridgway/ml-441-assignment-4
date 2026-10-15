"""Report figures (one pdf per panel) + numbers quoted in the text"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent.parent  # project root (same as run_experiments.py)
OUT = HERE / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "mathtext.fontset": "stix",
                     "font.size": 8, "legend.fontsize": 6, "lines.linewidth": 1.0, "axes.linewidth": 0.6}) # just fonts etc. from my research, otherwise figures look out of place with IEEE
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUES = matplotlib.colors.ListedColormap(plt.cm.Blues(np.linspace(0.3, 1, 256)))  # empty units stay white

names = pd.read_csv(HERE / "data/attack_category_map.csv").sort_values("Mapping")["Attack Category Name"].tolist()
cfg = json.load(open(HERE / "chosen.json"))
M = cfg["L"] ** 2
g1 = pd.read_csv(HERE / "stage1.csv").groupby(["sigma_f", "T", "L"])
s1, e1 = g1.mean().reset_index(), g1.sem().reset_index()  # mean and standard error over five folds
g2 = pd.read_csv(HERE / "stage2.csv").groupby(["rule", "K"])
s2, e2 = g2.mean(numeric_only=True).reset_index(), g2.sem(numeric_only=True).reset_index()


def panel(name, xlabel, ylabel, series, logx=False, legend=False):
    fig, ax = plt.subplots(figsize=(1.72, 1.05))
    for colour, (label, x, yv, err) in zip(COLOURS, series):
        ax.errorbar(x, yv, yerr=err, fmt="-o", color=colour, ms=2.5, capsize=1.5, elinewidth=0.6, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, xscale="log" if logx else "linear")
    if not logx:
        ax.set_xticks([10, 30, 50, 70])
    ax.tick_params(length=2)
    if legend:
        ax.legend(frameon=False)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


# stage 1 plots (unit-level F1 and TE vs L)
part, err = s1[s1["T"] == cfg["T"]], e1[e1["T"] == cfg["T"]]
for col, ylabel in [("macro_f1", "Validation macro F1"), ("te", "Topographic error")]:
    panel(f"stage1_{col}", "Side length $L$", ylabel,
          [(rf"$\sigma_f = {sf:g}$", g.L, g[col], err[err.sigma_f == sf][col]) for sf, g in part.groupby("sigma_f")],
          legend=col == "macro_f1")

# stage 2 plots
rules = [("Majority", "majority"), ("Class-normalised", "normalised")]
panel("stage2_f1", "Number of clusters $K$", "Validation macro F1",
      [(lab, s2[s2.rule == r].K, s2[s2.rule == r].macro_f1, e2[e2.rule == r].macro_f1) for lab, r in rules],
      logx=True, legend=True)
g = s2[(s2.rule == "majority") & (s2.K < M)]
panel("stage2_db", "Number of clusters $K$", "Davies–Bouldin", [("", g.K, g.db, None)], logx=True)

# fold 1 map: share of each class per unit, biggest classes first
mp = np.load(HERE / "map_fold1.npz")
share = mp["hits"] / mp["hits"].sum(0)
fig, axes = plt.subplots(1, 10, figsize=(7.1, 0.95))
for ax, c in zip(axes, np.argsort(-mp["hits"].sum(0))):
    im = ax.imshow(np.ma.masked_equal(share[:, c].reshape(cfg["L"], cfg["L"]), 0), cmap=BLUES,
                   norm=LogNorm(1e-4, share.max()), interpolation="none")
    ax.set_title(names[c], fontsize=7, pad=2)
    ax.set(xticks=[], yticks=[])
fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01).set_label("Share of class", fontsize=6)
fig.savefig(OUT / "map_classes.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)

# numbers for the text
res = pd.read_csv(HERE / "folds.csv")
pred = np.load(HERE / "predictions.npz")
print(cfg)
print(s1.round(4).to_string())
print(s2.round(4).to_string())
print(res.groupby("model").agg(["mean", "std"]).drop(columns="fold").round(4).to_string())
print(pd.DataFrame({m: f1_score(pred["y"], pred[m], average=None) for m in pred.files if m != "y"},
                   index=names).round(3).to_string())
own = mp["hits"].argmax(1)[:, None] == np.arange(10)  # units where the class is the majority
print(pd.DataFrame({"units_hit": (mp["hits"] > 0).sum(0), "units_owned": (own & (mp["hits"].sum(1) > 0)[:, None]).sum(0),
                    "share_in_owned_units": (mp["hits"] * own).sum(0) / mp["hits"].sum(0)}, index=names).round(3))

# what ends up in the K = 2 (DB) clusters on the fold 1 map
from run_experiments import cut, ward
two = cut(ward(mp["W"]), cfg["L"] ** 2, 2)
print("instances per class in the K = 2 clusters:\n",
      pd.DataFrame([mp["hits"][two == a].sum(0) for a in range(2)], columns=names).astype(int).to_string())
