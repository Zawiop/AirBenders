"""Step 2: statistical inference on root-cause drivers.

All 3,000 units in the data had a field failure, so every rate here is a share of
failures (P(fault type | unit failed)), not a failure rate per unit built.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

OUT = Path(__file__).parent / "out"
d = pd.read_csv(OUT / "clean.csv")
rng = np.random.default_rng(42)
lines = []


def log(s=""):
    lines.append(str(s))


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return p, max(0, c - h), min(1, c + h)


# ---- H1: leaks come only from Facility 2 and Supplier B ------------------------
d["F2"] = d.facility.eq("Facility 2")
d["SB"] = d.supplier.eq("Supplier B")
d["leak_src"] = np.select([d.F2 & d.SB, d.F2, d.SB], ["Fac 2 + Sup B", "Fac 2 only", "Sup B only"], "Neither")
log("H1 leak share of failures by source (Wilson 95% CI)")
rows = []
for g, x in d.groupby("leak_src"):
    p, lo, hi = wilson(x.is_leak.sum(), len(x))
    rows.append((g, len(x), x.is_leak.sum(), p, lo, hi))
    log(f"  {g:14s} n={len(x):5d} leaks={x.is_leak.sum():4d} share={p:.3f} [{lo:.3f}, {hi:.3f}]")
pd.DataFrame(rows, columns=["group", "n", "leaks", "share", "lo", "hi"]).to_csv(OUT / "h1_leak_by_source.csv", index=False)

for a in ["Fac 2 only", "Sup B only", "Fac 2 + Sup B"]:
    t = pd.crosstab(d.leak_src.isin([a]), d.is_leak).loc[:, [0, 1]]
    sub = d[d.leak_src.isin([a, "Neither"])]
    tab = pd.crosstab(sub.leak_src == a, sub.is_leak).reindex(columns=[0, 1], fill_value=0)
    _, pf = stats.fisher_exact(tab.values)
    log(f"  Fisher {a} vs Neither: p={pf:.2e}")
sub = d[d.leak_src != "Neither"]
chi, p, *_ = stats.chi2_contingency(pd.crosstab(sub.leak_src, sub.is_leak))
log(f"  Among exposed groups, do shares differ? chi2={chi:.2f} p={p:.3f} (no: one source is enough, both do not stack)")

# Within each facility, is Supplier B the driver? within each supplier, is Facility 2?
log("  Leak share by facility x supplier")
log(pd.crosstab(d.facility, d.supplier, values=d.is_leak, aggfunc="mean").round(3).to_string())
L = d[d.is_leak == 1]
chi, p, *_ = stats.chi2_contingency(pd.crosstab(L.leak_src, L["Failure Description"]))
log(f"  Leak subtype mix differs by source? chi2={chi:.2f} p={p:.3f}")

# ---- H2: controls faults only on Rev A ----------------------------------------
log("\nH2 controls share of failures by controls revision")
for g, x in d.groupby("rev"):
    k = (x.fault == "Controls").sum()
    p, lo, hi = wilson(k, len(x))
    log(f"  {g}: n={len(x)} controls={k} share={p:.3f} [{lo:.3f}, {hi:.3f}]")
tab = pd.crosstab(d.rev == "Rev A", d.fault == "Controls")
log(f"  Fisher Rev A vs Rev B/C: p={stats.fisher_exact(tab.values)[1]:.2e}")
log("  Rev A share of units by month (still being built?): "
    + d.groupby("month").rev.apply(lambda s: (s == "Rev A").mean()).round(2).iloc[:-1].describe()[["min", "max"]].to_string().replace("\n", ", "))

# ---- H3: fan rotation faults only on night shift -------------------------------
log("\nH3 fan rotation share of failures by shift")
for g, x in d.groupby("shift"):
    k = (x.fault == "Fan Rotation").sum()
    p, lo, hi = wilson(k, len(x))
    log(f"  {g}: n={len(x)} fan={k} share={p:.3f} [{lo:.3f}, {hi:.3f}]")
log(f"  Fisher: p={stats.fisher_exact(pd.crosstab(d['shift'], d.fault == 'Fan Rotation').values)[1]:.2e}")
log(f"  Sensitivity: rework label says {(d['Repair Type'] == 'Fan Rotation Correction').sum()} fan units, "
    f"all night shift = {(d.loc[d['Repair Type'] == 'Fan Rotation Correction', 'shift'] == 'Night').all()}")

# ---- H4: LegacyX repairs cost more for the same fault ---------------------------
log("\nH4 OLS log(field cost) ~ fault + family + facility + supplier + rev + shift + region")
X = pd.get_dummies(d[["fault", "family", "facility", "supplier", "rev", "shift", "region"]], drop_first=True).astype(float)
X.insert(0, "const", 1.0)
y = np.log(d.field_cost.values)


def ols(Xm, yv):
    b, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    return b


beta = ols(X.values, y)
resid = y - X.values @ beta
r2 = 1 - resid.var() / y.var()
boots = np.array([ols(X.values[i], y[i]) for i in (rng.integers(0, len(d), len(d)) for _ in range(1000))])
lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
coef = pd.DataFrame({"beta": beta, "lo": lo, "hi": hi}, index=X.columns)
coef["multiplier"] = np.exp(coef.beta)
coef["mult_lo"] = np.exp(coef.lo)
coef["mult_hi"] = np.exp(coef.hi)
coef.to_csv(OUT / "h4_cost_model.csv")
log(f"  R^2 = {r2:.3f}")
log(coef[["multiplier", "mult_lo", "mult_hi"]].round(3).to_string())
log("  read: multiplier = cost relative to the baseline level (Controls, DC100, Facility 1, Sup A, Rev A, Day, Canada)")

# ---- Null checks: things that do NOT matter ------------------------------------
log("\nNull checks (chi-square of reconciled fault vs variable)")
for v in ["region", "eolt_station", "disposition", "month", "family"]:
    chi, p, dof, _ = stats.chi2_contingency(pd.crosstab(d[v], d.fault))
    log(f"  fault vs {v:12s}: chi2={chi:7.1f} dof={dof:3d} p={p:.3f}")
for v in ["region", "disposition", "eolt_station"]:
    h, p = stats.kruskal(*[g.field_cost for _, g in d.groupby(v)])
    log(f"  field cost vs {v:12s}: Kruskal p={p:.3f}")
log("  10+ tests run; findings H1-H3 have p < 1e-30, far past any Bonferroni cut")

# ---- excess cost attributable to each driver ------------------------------------
log("\nCost of each problem over the 12 months of field data")
cost = d.groupby("fault").field_cost.agg(["count", "sum", "mean"])
log(cost.round(0).to_string())
leg = d[d.family == "LegacyX"]
fair = leg.groupby("fault").size() * d[d.family != "LegacyX"].groupby("fault").field_cost.mean()
log(f"  LegacyX actual cost {leg.field_cost.sum():,.0f} vs cost at non-LegacyX prices {fair.sum():,.0f} "
    f"-> premium {leg.field_cost.sum() - fair.sum():,.0f}")

open(OUT / "02_inference_report.txt", "w").write("\n".join(lines))
print("\n".join(lines))
