"""Step 6: checks added after the first write-up.

1. Three-flag rule model: P(fault | leak-exposed source, Rev A, night shift), an 8-cell
   lookup fit on the training months only. Is the classifier's signal just these flags?
2. Do the three rules hold in the training months alone (so the flags are not learned
   from the test window)?
3. Bootstrap 95% CIs for the headline numbers: classifier skill vs class priors, and the
   two-stage cost model's MAE reduction vs the training-mean baseline.
4. A loose lower bound on log loss for any build-cell model: each test cell's own fault
   mix, fit on the test set itself (in-sample, so optimistic).
5. Calibration of the leak probability on the test window.
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent / "out"
CLASSES = ["Controls", "Fan Rotation", "Leak", "Power Wiring"]
KEY = ["facility", "supplier", "rev", "shift"]

d = pd.read_csv(OUT / "clean.csv")
t = pd.read_csv(OUT / "q1_test_predictions.csv")          # best classifier, test window
test_ids = set(t["Serial Number"])
tr = d[~d["Serial Number"].isin(test_ids)].copy()
te = d[d["Serial Number"].isin(test_ids)].copy()
lines = [f"train n {len(tr)}, test n {len(te)}"]


def flags(x):
    x = x.copy()
    x["L"] = x.facility.eq("Facility 2") | x.supplier.eq("Supplier B")
    x["A"] = x.rev.eq("Rev A")
    x["N"] = x["shift"].eq("Night")
    return x


def logloss(P, y):
    return -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1)))


tr, te = flags(tr), flags(te)
y = te.fault.map({c: i for i, c in enumerate(CLASSES)}).values

# 2. rules on training months only
lines.append("\nrules in training months only")
lines.append(f"  leaks outside Facility 2 / Supplier B: {(tr.fault.eq('Leak') & ~tr.L).sum()} of {(~tr.L).sum()}; "
             f"inside: {(tr.fault.eq('Leak') & tr.L).sum()} of {tr.L.sum()}")
lines.append(f"  controls faults off Rev A: {(tr.fault.eq('Controls') & ~tr.A).sum()} of {(~tr.A).sum()}; "
             f"on Rev A: {(tr.fault.eq('Controls') & tr.A).sum()} of {tr.A.sum()}")
lines.append(f"  fan faults on day shift: {(tr.fault.eq('Fan Rotation') & ~tr.N).sum()} of {(~tr.N).sum()}; "
             f"on night shift: {(tr.fault.eq('Fan Rotation') & tr.N).sum()} of {tr.N.sum()}")

# 1. three-flag rule model
counts = tr.groupby(["L", "A", "N"]).fault.value_counts().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
mix = (counts + 0.5).div((counts + 0.5).sum(axis=1), axis=0)         # light smoothing
P_rule = te[["L", "A", "N"]].merge(mix, left_on=["L", "A", "N"], right_index=True, how="left")[CLASSES].values
prior = tr.fault.value_counts(normalize=True).reindex(CLASSES).values
P_prior = np.tile(prior, (len(y), 1))
P_best = t.set_index("Serial Number").loc[te["Serial Number"], [f"p_{c}" for c in CLASSES]].values
ll_prior, ll_rule, ll_best = logloss(P_prior, y), logloss(P_rule, y), logloss(P_best, y)
lines.append("\ntest log loss")
lines.append(f"  class priors          {ll_prior:.3f}")
lines.append(f"  three-flag rule model {ll_rule:.3f}  ({len(mix)} cells, skill {1 - ll_rule / ll_prior:.1%})")
lines.append(f"  best classifier       {ll_best:.3f}  (skill {1 - ll_best / ll_prior:.1%})")

# 4. optimistic bound
tt = te.copy()
cell = tt.groupby(KEY).fault.value_counts(normalize=True).unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
P_or = tt[KEY].merge(cell, left_on=KEY, right_index=True, how="left")[CLASSES].values
lines.append(f"  in-sample bound (each test cell's own mix, fit on test): {logloss(P_or, y):.3f}")

# 3. bootstrap CIs
rng = np.random.default_rng(0)
cost = t.set_index("Serial Number").loc[te["Serial Number"]]
fc, ec = cost.field_cost.values, cost.exp_cost_two_stage.values
base_mae_pred = tr.field_cost.mean()
skill, skill_rule, mae_red = [], [], []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    lp = logloss(P_prior[i], y[i])
    skill.append(1 - logloss(P_best[i], y[i]) / lp)
    skill_rule.append(1 - logloss(P_rule[i], y[i]) / lp)
    mae_red.append(1 - np.abs(fc[i] - ec[i]).mean() / np.abs(fc[i] - base_mae_pred).mean())
ci = lambda v: f"{np.mean(v):.1%} [{np.percentile(v, 2.5):.1%}, {np.percentile(v, 97.5):.1%}]"
lines.append("\nbootstrap 95% CIs on the test window (2,000 resamples)")
lines.append(f"  classifier skill vs priors: {ci(skill)}")
lines.append(f"  rule model skill vs priors: {ci(skill_rule)}")
lines.append(f"  two-stage MAE ${np.abs(fc - ec).mean():,.0f} vs training mean ${np.abs(fc - base_mae_pred).mean():,.0f}; "
             f"reduction {ci(mae_red)}")

# 5. calibration of the leak probability
cal = t.assign(is_leak=t.fault.eq("Leak")).groupby(pd.cut(t.p_leak, [-0.01, 0.05, 0.3, 0.5, 0.6, 1.0]), observed=True).agg(
    n=("is_leak", "size"), predicted=("p_leak", "mean"), observed=("is_leak", "mean"))
lines.append("\nleak probability calibration (test window)\n" + cal.round(3).to_string())
lines.append(f"share of failed test units built at Facility 2 or with Supplier B: "
             f"{(te.L).mean():.1%}")

open(OUT / "06_rule_check_report.txt", "w").write("\n".join(lines))
print("\n".join(lines))
