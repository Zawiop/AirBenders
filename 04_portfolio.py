"""Step 4: price each catalog investment against the cost pools found in the data,
pick the best portfolio under $750K, and test how robust it is.

Cost pools (12 months of data treated as one year):
  field  = field repair cost (proxy for warranty cost: Dataset 2 not provided and every
           field event is warranty eligible)
  rework = in-plant rework labor x rate + material + EOLT rework hours x rate
Stacking: benefits act multiplicatively on each unit's cost, so a second investment on the
same problem acts on what is left after the first.
Timing: an investment saves nothing until its implementation months have passed.
"""
import itertools
import json
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent / "out"
d = pd.read_csv(OUT / "clean.csv")
LABOR = 45.0          # $/hr loaded labor rate (assumption, tested below)
BUDGET = 750_000
HORIZON = 36          # months
LEGACY_MULT = pd.read_csv(OUT / "h4_cost_model.csv", index_col=0).loc["family_LegacyX", "multiplier"]

d["rework_cost"] = d.rework_labor_hrs * LABOR + d.rework_material + d.eolt_rework_hrs * LABOR
d["legacy"] = d.family.eq("LegacyX")
d["F2"] = d.facility.eq("Facility 2")
d["SB"] = d.supplier.eq("Supplier B")
d["leak_src"] = np.select([d.F2 & d.SB, d.F2, d.SB], ["both", "F2", "SB"], "none")
# split LegacyX field cost into base and the model-estimated LegacyX premium
d["field_prem"] = np.where(d.legacy, d.field_cost * (1 - 1 / LEGACY_MULT), 0)
d["field_base"] = d.field_cost - d.field_prem

cells = d.groupby(["fault", "legacy", "leak_src"]).agg(
    n=("field_cost", "size"), field_base=("field_base", "sum"),
    field_prem=("field_prem", "sum"), rework=("rework_cost", "sum")).reset_index()

# ---- catalog --------------------------------------------------------------------
CAT = pd.DataFrame([
    (1, "Braze Process Training", 50_000, 2, "Low"),
    (2, "Smart Torque Tools", 100_000, 3, "Low"),
    (3, "Wiring Poka-Yoke", 125_000, 4, "Low"),
    (4, "Predictive Quality Dashboard", 150_000, 6, "Medium"),
    (5, "Enhanced Design Validation", 200_000, 6, "Medium"),
    (6, "Vision Inspection", 275_000, 6, "Medium"),
    (7, "Automated Leak Detection", 350_000, 8, "Low"),
    (8, "Additional EOLT Station", 450_000, 8, "Medium"),
    (9, "Supplier Quality Program", 175_000, 4, "Medium"),
    (10, "LegacyX Product Refresh", 500_000, 12, "High"),
], columns=["id", "name", "cost", "months", "risk"]).set_index("id")


def effects(c):
    """Return {inv_id: (field_base_red, field_prem_red, rework_red)} for one cell c, at catalog values.
    Mapping from catalog target problems to the reconciled fault classes:
      leak = Refrigerant/Suction/Service Valve leaks, braze defects
      power wiring = loose power wiring / loose terminal (electrical)
      fan rotation + controls = miswiring (EOLT calls controls defects Sensor Miswire)
    Where the catalog gives one number we apply it to field and rework alike."""
    f, leg, src = c.fault, c.legacy, c.leak_src
    e = {}
    if f == "Leak":
        e[1] = (0.05, 0.05, 0.10)
        e[7] = (0.40, 0.40, 0.0)
        if src == "SB":                        # supplier-driven leaks: only Sup B outside Facility 2
            e[9] = (0.25, 0.25, 0.25)
    if f == "Power Wiring":
        e[2] = (0.15, 0.15, 0.30)
    if f in ("Fan Rotation", "Controls"):
        e[3] = (0.40, 0.40, 0.40)
    if f in ("Power Wiring", "Fan Rotation", "Controls"):
        e[6] = (0.35, 0.35, 0.35)
    if f == "Controls":
        e[5] = (0.20, 0.20, 0.0)
    if leg:
        e[5] = (e.get(5, (0, 0, 0))[0], 0.20, 0.0)     # design validation also hits the LegacyX premium
        e[10] = (0.35, 0.35, 0.0)
    e[4] = (0.15, 0.15, 0.0)                    # quality escapes = all field cost
    e[8] = (0.15, 0.15, 0.0)
    return e


EFF = [effects(c) for c in cells.itertuples()]
POOLS = cells[["field_base", "field_prem", "rework"]].values      # (cells, 3)
base_annual = POOLS.sum()


def annual_cost(active, scale=None):
    """annual cost of all cells with a set of active investments; scale = {id: benefit multiplier}"""
    m = np.ones_like(POOLS)
    for ci, e in enumerate(EFF):
        for i in active:
            if i in e:
                s = 1.0 if scale is None else scale[i]
                m[ci] *= 1 - np.minimum(np.array(e[i]) * s, 0.95)
    return (POOLS * m).sum()


def evaluate(port, scale=None, months=None):
    months = months or CAT.months.to_dict()
    out = {"y1": 0.0, "h": 0.0}
    for t in range(1, HORIZON + 1):
        active = [i for i in port if months[i] < t]
        s = (base_annual - annual_cost(active, scale)) / 12
        out["h"] += s
        if t <= 12:
            out["y1"] += s
    out["steady"] = base_annual - annual_cost(list(port), scale)
    return out


# ---- standalone value of each investment ------------------------------------------
lines = [f"annual cost base: field {d.field_cost.sum():,.0f} + rework {d.rework_cost.sum():,.0f} "
         f"= {base_annual:,.0f}  (labor ${LABOR}/hr)"]
lines.append("pool by fault:\n" + d.groupby("fault")[["field_cost", "rework_cost"]].sum().round(0).to_string())
solo = []
for i, r in CAT.iterrows():
    ev = evaluate([i])
    solo.append(dict(id=i, name=r["name"], cost=r.cost, months=r.months, risk=r.risk,
                     steady_annual=ev["steady"], year1=ev["y1"], three_yr=ev["h"],
                     roi_3yr=(ev["h"] - r.cost) / r.cost, payback_months=r.months + r.cost / (ev["steady"] / 12)))
solo = pd.DataFrame(solo)
solo.to_csv(OUT / "p_standalone.csv", index=False)
lines.append("\nstandalone value of each investment\n" + solo.round(2).to_string(index=False))

# ---- enumerate every portfolio under budget ---------------------------------------
ports = []
for k in range(1, 11):
    for combo in itertools.combinations(CAT.index, k):
        cost = CAT.loc[list(combo), "cost"].sum()
        if cost <= BUDGET:
            ev = evaluate(combo)
            ports.append(dict(port=combo, cost=cost, y1=ev["y1"], three_yr=ev["h"], steady=ev["steady"],
                              net3=ev["h"] - cost, roi3=(ev["h"] - cost) / cost, net1=ev["y1"] - cost))
P = pd.DataFrame(ports).sort_values("net3", ascending=False).reset_index(drop=True)
P["names"] = P.port.apply(lambda p: " + ".join(CAT.loc[list(p), "name"]))
P.to_csv(OUT / "p_all_portfolios.csv", index=False)
lines.append(f"\n{len(P)} feasible portfolios. top 10 by 3-year net savings")
lines.append(P.head(10)[["names", "cost", "y1", "steady", "net3", "roi3"]].round(2).to_string())
best = P.port.iloc[0]
lines.append(f"\nbest by year-1 net: {P.sort_values('net1', ascending=False).names.iloc[0]}")

# ---- data-adjusted scenario: detection-only investments at half benefit -------------
# every field leak in the data had already failed EOLT as a leak and passed after repair,
# so more detection may help less than the catalog says; prevention is unaffected.
adj = {i: 1.0 for i in CAT.index}
adj[7] = adj[8] = 0.5
alt = sorted(((evaluate(p, adj)["h"] - CAT.loc[list(p), "cost"].sum(), p) for p in P.port), reverse=True)[:5]
lines.append("\nscenario: Leak Detection and EOLT Station at 50% of catalog benefit, top 5")
for v, p in alt:
    lines.append(f"  {v:12,.0f}  {' + '.join(CAT.loc[list(p), 'name'])}  (${CAT.loc[list(p), 'cost'].sum():,})")
pd.DataFrame([(v, " + ".join(CAT.loc[list(p), "name"])) for v, p in alt], columns=["net3", "names"]).to_csv(
    OUT / "p_scenario_detection_half.csv", index=False)

# ---- Dashboard dependency: the Dashboard is credited with 15% of ALL field cost --------
# (every unit in the data is a quality escape). It is also a detection tool, so test the
# pick with its benefit cut, alone and together with the detection-half scenario above.
DASH = [("Dashboard 50%", {4: 0.5}), ("Dashboard 0%", {4: 0.0}),
        ("Dashboard 50% + Leak Detection/EOLT 50%", {4: 0.5, 7: 0.5, 8: 0.5})]
dash_rows = []
lines.append("\nscenario: Predictive Quality Dashboard benefit cut, top 3")
for label, cut in DASH:
    sc = {i: cut.get(i, 1.0) for i in CAT.index}
    res = sorted(((evaluate(p, sc)["h"] - CAT.loc[list(p), "cost"].sum(), p) for p in P.port), reverse=True)[:3]
    lines.append(f" {label}")
    for v, p in res:
        nm = " + ".join(CAT.loc[list(p), "name"])
        lines.append(f"  {v:12,.0f}  {nm}  (${CAT.loc[list(p), 'cost'].sum():,})")
        dash_rows.append((label, v, nm, CAT.loc[list(p), "cost"].sum()))
pd.DataFrame(dash_rows, columns=["scenario", "net3", "names", "cost"]).to_csv(
    OUT / "p_scenario_dashboard.csv", index=False)

# ---- Monte Carlo: uncertain benefits and delays ------------------------------------
rng = np.random.default_rng(7)
SPREAD = {"Low": 0.25, "Medium": 0.40, "High": 0.60}
N = 2000
top = P.head(40).port.tolist()          # candidates: the 40 best at catalog values
wins = {p: 0 for p in top}
draws = {p: [] for p in top}
for _ in range(N):
    scale = {i: rng.triangular(1 - SPREAD[r], 1 - SPREAD[r] / 4, 1 + SPREAD[r] / 2) for i, r in CAT.risk.items()}
    months = {i: m + (rng.integers(0, 4) if CAT.risk[i] != "Low" else rng.integers(0, 2)) for i, m in CAT.months.items()}
    vals = {p: evaluate(p, scale, months)["h"] - CAT.loc[list(p), "cost"].sum() for p in top}
    w = max(vals, key=vals.get)
    wins[w] += 1
    for p in top:
        draws[p].append(vals[p])
mc = pd.DataFrame([dict(names=" + ".join(CAT.loc[list(p), "name"]), port=p,
                        p_best=wins[p] / N, net3_mean=np.mean(draws[p]), net3_p5=np.percentile(draws[p], 5),
                        net3_p95=np.percentile(draws[p], 95), p_loss=np.mean(np.array(draws[p]) < 0))
                   for p in top]).sort_values("p_best", ascending=False)
mc.to_csv(OUT / "p_monte_carlo.csv", index=False)
np.save(OUT / "p_mc_best_draws.npy", np.array(draws[mc.port.iloc[0]]))
lines.append(f"\nMonte Carlo ({N} draws; benefit scale triangular by risk, extra delay 0-3 months for Medium/High)")
lines.append(mc.head(8).drop(columns="port").round(3).to_string(index=False))

# ---- sensitivity: labor rate and catalog benefit haircut ---------------------------
lines.append("\nsensitivity of the top portfolio choice")
for lab in [30, 45, 70]:
    for hair in [1.0, 0.75, 0.5]:
        sc = {i: hair for i in CAT.index}
        dd = d.assign(rc=d.rework_labor_hrs * lab + d.rework_material + d.eolt_rework_hrs * lab)
        POOLS[:, 2] = dd.groupby(["fault", "legacy", "leak_src"]).rc.sum().values
        base_annual = POOLS.sum()
        res = [(evaluate(p, sc)["h"] - CAT.loc[list(p), "cost"].sum(), p) for p in top]
        v, p = max(res)
        lines.append(f"  labor ${lab}/hr, benefits x{hair}: best = {' + '.join(CAT.loc[list(p), 'name'])}  net3 {v:,.0f}")
POOLS[:, 2] = cells.rework.values
base_annual = POOLS.sum()

# ---- field failures avoided per year (estimate) ------------------------------------
# Apply each item's field-side reduction to the failure COUNT in each cell instead of
# its cost. The catalog states some benefits as cost reductions, not defect
# reductions, so this is an approximation.
def failures_avoided(port):
    by = {}
    for c, e in zip(cells.itertuples(), EFF):
        m = 1.0
        for i in port:
            if i in e:
                m *= 1 - min(e[i][0], 0.95)
        by[c.fault] = by.get(c.fault, 0.0) + c.n * (1 - m)
    return by

fa_rows = []
lines.append("\nfield failures avoided per year at steady state (estimate: catalog reductions applied to counts)")
for label, port in [("recommended", best), ("Dashboard-half fallback", (2, 6, 7))]:
    by = failures_avoided(port)
    tot = sum(by.values())
    lines.append(f"  {label}: {tot:,.0f} of {len(d):,}  " + ", ".join(f"{k} {v:,.0f}" for k, v in sorted(by.items())))
    fa_rows.append(dict(portfolio=label, names=" + ".join(CAT.loc[list(port), "name"]), total=tot, **by))
pd.DataFrame(fa_rows).to_csv(OUT / "p_failures_avoided.csv", index=False)

# ---- model-informed targeting: leak risk score on the held-out window -------------
t = pd.read_csv(OUT / "q1_test_predictions.csv").sort_values("p_leak", ascending=False)
t["is_leak"] = t.fault.eq("Leak")
t["cum_units"] = np.arange(1, len(t) + 1) / len(t)
t["cum_leak_cost"] = (t.field_cost * t.is_leak).cumsum() / (t.field_cost * t.is_leak).sum()
t[["cum_units", "cum_leak_cost", "p_leak"]].to_csv(OUT / "p_leak_capture_curve.csv", index=False)
for q in [0.2, 0.3, 0.4, 0.5]:
    lines.append(f"  flag top {q:.0%} of units by model leak risk -> {t.cum_leak_cost.iloc[int(q * len(t)) - 1]:.1%} of leak cost captured (random: {q:.0%})")

open(OUT / "04_portfolio_report.txt", "w").write("\n".join(lines))
print("\n".join(lines))
