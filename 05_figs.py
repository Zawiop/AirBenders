"""Step 5: slide-ready charts (PNG, 16:9-friendly)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
OUT, FIG = ROOT / "out", ROOT / "figs"
FIG.mkdir(exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 15,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left"})
d = pd.read_csv(OUT / "clean.csv")


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def money(x):
    return f"${x/1e6:.2f}M" if x >= 1e6 else f"${x/1e3:.0f}K"


# 1. where the money goes: share of events vs share of cost ----------------------------
g = d.groupby("fault").agg(n=("field_cost", "size"), cost=("field_cost", "sum")).sort_values("cost")
fig, ax = plt.subplots(figsize=(10, 4.6))
y = np.arange(len(g))
ax.barh(y + 0.2, g.n / g.n.sum() * 100, 0.38, color=GRAY, label="Share of field failures")
ax.barh(y - 0.2, g.cost / g.cost.sum() * 100, 0.38, color=BLUE, label="Share of field repair cost")
for i, (n, c) in enumerate(zip(g.n / g.n.sum() * 100, g.cost / g.cost.sum() * 100)):
    ax.text(n + 0.8, i + 0.2, f"{n:.0f}%", va="center", color=INK2, fontsize=11)
    ax.text(c + 0.8, i - 0.2, f"{c:.0f}%  ({money(g.cost.iloc[i])})", va="center", color=INK, fontsize=11)
ax.set_yticks(y, g.index)
ax.set_xlim(0, 95)
ax.set_xlabel("% of total")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Leaks are 26% of failures but 76% of the $3.83M field repair bill")
save(fig, "01_cost_vs_count.png")

# 2. leak share by facility x supplier heatmap ----------------------------------------
t = pd.crosstab(d.facility, d.supplier, values=d.is_leak, aggfunc="mean")
n = pd.crosstab(d.facility, d.supplier)
fig, ax = plt.subplots(figsize=(8, 4.8))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec"] + SEQ[2:])
ax.imshow(t.values, cmap=cmap, vmin=0, vmax=0.65, aspect="auto")
for i in range(t.shape[0]):
    for j in range(t.shape[1]):
        v = t.values[i, j]
        ax.text(j, i, f"{v:.0%}\nn={n.values[i, j]}", ha="center", va="center",
                color="white" if v > 0.35 else INK, fontsize=12, fontweight="bold" if v > 0.35 else "normal")
ax.set_xticks(range(3), t.columns)
ax.set_yticks(range(4), t.index)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Leak share of failures: only Facility 2 or Supplier B coils leak", pad=28)
ax.text(0, 1.025, "0 leaks in 1,570 failures outside these two sources (Fisher p < 1e-140)",
        transform=ax.transAxes, color=INK2, fontsize=11)
save(fig, "02_leak_heatmap.png")

# 3. three root causes in one figure ---------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
panels = [("Facility / supplier", np.where(d.facility.eq("Facility 2") | d.supplier.eq("Supplier B"),
                                           "Fac 2 or\nSup B", "All other"), "Leak", "Leak share"),
          ("Controls revision", d.rev.values, "Controls", "Controls share"),
          ("Shift", d["shift"].values, "Fan Rotation", "Fan rotation share")]
for ax, (title, grp, fault, lab) in zip(axes, panels):
    s = pd.Series(d.fault.eq(fault).values).groupby(grp).agg(["mean", "size", "sum"])
    p, k, m = s["mean"].values, s["sum"].values, s["size"].values
    z = 1.96
    den = 1 + z**2 / m
    half = z * np.sqrt(p * (1 - p) / m + z**2 / (4 * m**2)) / den
    cols = [ORANGE if v > 0.05 else GRAY for v in p]
    ax.bar(range(len(s)), p * 100, 0.6, color=cols, yerr=half * 100, ecolor=INK2, capsize=4)
    for i, v in enumerate(p):
        ax.text(i, v * 100 + 4.5, f"{v:.0%}", ha="center", color=INK, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(s)), s.index)
    ax.set_title(f"{lab} by {title.lower()}", fontsize=13)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
axes[0].set_ylabel("% of failures (95% CI)")
axes[0].set_ylim(0, 72)
fig.suptitle("Each fault type traces to one build condition", x=0.01, ha="left", fontsize=16, fontweight="bold", color=INK)
fig.tight_layout()
save(fig, "03_root_causes.png")

# 4. LegacyX cost premium ----------------------------------------------------------------
c = d.assign(fam=np.where(d.family.eq("LegacyX"), "LegacyX", "DC100/200/300")).groupby(["fault", "fam"]).field_cost.mean().unstack()
c = c.loc[["Leak", "Controls", "Fan Rotation", "Power Wiring"]]
fig, ax = plt.subplots(figsize=(10, 4.4))
x = np.arange(len(c))
ax.bar(x - 0.2, c["DC100/200/300"], 0.38, color=GRAY, label="DC100 / DC200 / DC300")
ax.bar(x + 0.2, c["LegacyX"], 0.38, color=BLUE, label="LegacyX")
for i in range(len(c)):
    ax.text(i + 0.2, c["LegacyX"].iloc[i] + 80, f"${c['LegacyX'].iloc[i]:,.0f}", ha="center", fontsize=11, color=INK)
    ax.text(i - 0.2, c["DC100/200/300"].iloc[i] + 80, f"${c['DC100/200/300'].iloc[i]:,.0f}", ha="center", fontsize=11, color=INK2)
ax.set_xticks(x, c.index)
ax.set_ylabel("Mean field repair cost ($)")
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)
ax.legend(frameon=False)
ax.set_title("Same fault, 1.52x the cost on LegacyX (95% CI 1.45 to 1.60)")
save(fig, "04_legacyx_premium.png")

# 5. model comparison ------------------------------------------------------------------
r = pd.read_csv(OUT / "q1_classifier_results.csv")
r = r[r.model != "Majority class (baseline)"].sort_values("log_loss_skill_vs_prior")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
cols = [GRAY if "baseline" in m else BLUE for m in r.model]
axes[0].barh(r.model, r.log_loss_skill_vs_prior * 100, color=cols, height=0.6)
for i, v in enumerate(r.log_loss_skill_vs_prior * 100):
    axes[0].text(v + 1, i, f"{v:.0f}%", va="center", fontsize=11)
axes[0].set_title("Log-loss skill vs class priors", fontsize=13)
axes[0].set_xlabel("% reduction in test log loss")
axes[0].set_xlim(0, 52)
axes[1].barh(r.model, r.leak_auc, color=cols, height=0.6)
for i, v in enumerate(r.leak_auc):
    axes[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=11)
axes[1].axvline(0.5, color=INK2, lw=1, ls="--")
axes[1].set_xlim(0.4, 1.0)
axes[1].set_title("Leak ROC AUC (0.5 = coin flip)", fontsize=13)
for ax in axes:
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
fig.suptitle("Fault-type models on held-out Oct 2025 to Jan 2026 failures (n=748)", x=0.01, ha="left",
             fontsize=16, fontweight="bold", color=INK)
fig.tight_layout()
save(fig, "05_model_comparison.png")

# 6. regression comparison ------------------------------------------------------------
q = pd.read_csv(OUT / "q2_regression_results.csv")
fig, ax = plt.subplots(figsize=(10, 4.2))
cols = [GRAY if "baseline" in m else BLUE for m in q.model]
ax.barh(q.model[::-1], q.MAE[::-1], color=cols[::-1], height=0.6)
for i, (m, r2) in enumerate(zip(q.MAE[::-1], q.R2[::-1])):
    ax.text(m + 15, i, f"${m:,.0f}   R² {r2:.2f}", va="center", fontsize=11)
ax.set_xlim(0, 1700)
ax.set_xlabel("Mean absolute error per failed unit ($), lower is better")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.set_title("Expected repair cost from build data: MAE down 33% vs baseline")
save(fig, "06_cost_model.png")

# 7. leak capture curve ----------------------------------------------------------------
cc = pd.read_csv(OUT / "p_leak_capture_curve.csv")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([0, 100], [0, 100], color=GRAY, lw=2, ls="--", label="Random inspection")
ax.plot(np.r_[0, cc.cum_units * 100], np.r_[0, cc.cum_leak_cost * 100], color=BLUE, lw=2.5, label="Model leak-risk ranking")
k = (cc.cum_leak_cost >= 0.999).idxmax()
ax.scatter([cc.cum_units[k] * 100], [100], s=60, color=BLUE, zorder=3)
ax.annotate(f"{cc.cum_units[k]:.0%} of units hold\n100% of leak cost", (cc.cum_units[k] * 100, 100),
            (cc.cum_units[k] * 100 + 6, 70), color=INK, fontsize=11, arrowprops=dict(arrowstyle="-", color=INK2))
ax.set_xlabel("% of units flagged (highest risk first)")
ax.set_ylabel("% of leak repair cost captured")
ax.grid(color=GRID)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Targeting leak checks with the model (test window)")
save(fig, "07_leak_capture.png")

# 8. standalone investments + chosen portfolio -------------------------------------------
s = pd.read_csv(OUT / "p_standalone.csv").sort_values("roi_3yr")
best = pd.read_csv(OUT / "p_all_portfolios.csv").iloc[0]
chosen = set(eval(best.port))
fig, ax = plt.subplots(figsize=(11, 5.2))
cols = [BLUE if i in chosen else GRAY for i in s.id]
ax.barh(s.name + "  (" + (s.cost / 1e3).astype(int).astype(str) + "K)", s.roi_3yr * 100, color=cols, height=0.6)
for i, v in enumerate(s.roi_3yr * 100):
    ax.text(v + 10, i, f"{v:.0f}%", va="center", fontsize=11)
ax.set_xlabel("3-year ROI if bought alone (net savings / cost)")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.set_title(f"Recommended portfolio (blue): \\${best.cost/1e3:.0f}K of \\$750K", pad=28)
ax.text(0, 1.025, f"Year 1 savings {money(best.y1)}, run rate {money(best.steady)}/yr, 3-year net {money(best.net3)}".replace("$", "\\$"),
        transform=ax.transAxes, color=INK2, fontsize=11)
ax.text(0, 1.025, "", transform=ax.transAxes)
save(fig, "08_investments.png")

# 9. Monte Carlo ------------------------------------------------------------------------
mc = pd.read_csv(OUT / "p_monte_carlo.csv")
draws = np.load(OUT / "p_mc_best_draws.npy")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
axes[0].hist(draws / 1e6, bins=40, color=BLUE, edgecolor=SURF)
p5 = np.percentile(draws, 5) / 1e6
axes[0].axvline(p5, color=INK2, ls="--")
axes[0].text(p5, axes[0].get_ylim()[1] * 0.9, f"  5th pct ${p5:.2f}M", color=INK, fontsize=11)
axes[0].set_xlabel("3-year net savings ($M)")
axes[0].set_ylabel("Simulations")
axes[0].set_title("Recommended portfolio, 2,000 simulations", fontsize=13)
top = mc.head(5).iloc[::-1]
short = {"Smart Torque Tools": "Torque", "Wiring Poka-Yoke": "Poka-Yoke", "Predictive Quality Dashboard": "Dashboard",
         "Automated Leak Detection": "Leak Detect", "Vision Inspection": "Vision", "Supplier Quality Program": "Supplier",
         "Braze Process Training": "Braze", "Enhanced Design Validation": "Design Val", "Additional EOLT Station": "EOLT",
         "LegacyX Product Refresh": "LegacyX"}
labels = [" + ".join(short[x] for x in n.split(" + ")) for n in top.names]
cols = [BLUE if i == len(top) - 1 else GRAY for i in range(len(top))]
axes[1].barh(labels, top.p_best * 100, color=cols, height=0.6)
for i, v in enumerate(top.p_best * 100):
    axes[1].text(v + 1, i, f"{v:.0f}%", va="center", fontsize=11)
axes[1].set_xlim(0, 100)
axes[1].set_xlabel("% of simulations where portfolio is best")
axes[1].set_title("Which portfolio wins under uncertainty", fontsize=13)
for ax in axes:
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
fig.suptitle("Benefits varied by risk level (Low ±25%, Medium ±40%, High ±60%) plus 0-3 month delays",
             x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.tight_layout()
save(fig, "09_monte_carlo.png")
print("figs:", sorted(p.name for p in FIG.glob("*.png")))
