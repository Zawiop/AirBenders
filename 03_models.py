"""Step 3: predictive models from build-time features only.

Q1 (classification): when a unit fails in the field, which fault will it be?
Q2 (regression): what field repair cost should we expect for a unit when it fails?
Q3 (clustering): which build configurations behave alike?

Features = what is known when the unit ships: product family, facility, coil supplier,
controls revision, shift, customer region. Rework and EOLT labels are excluded since
they share the target's defect label (leakage risk).

Validation = time split on service date: train Jan-Sep 2025, test Oct 2025-Jan 2026.
Hyperparameters tuned by 5-fold stratified CV inside the training window only.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (log_loss, accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score,
                             average_precision_score, brier_score_loss, mean_absolute_error,
                             mean_squared_error, r2_score, confusion_matrix, silhouette_score)
from sklearn.cluster import KMeans, AgglomerativeClustering

OUT = Path(__file__).parent / "out"
d = pd.read_csv(OUT / "clean.csv")
FEATS = ["family", "facility", "supplier", "rev", "shift", "region"]
CLASSES = ["Controls", "Fan Rotation", "Leak", "Power Wiring"]
d["Date"] = pd.to_datetime(d["Date"])
train = d[d.Date < "2025-10-01"].copy()
test = d[d.Date >= "2025-10-01"].copy()
res = {"n_train": len(train), "n_test": len(test)}
lines = [f"train n={len(train)} (Jan-Sep 2025), test n={len(test)} (Oct 2025-Jan 2026)"]
log = lines.append
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def ohe():
    return make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATS))


# ============================== Q1 classification ===============================
cands = {
    "Majority class (baseline)": (DummyClassifier(strategy="most_frequent"), {}),
    "Class priors (baseline)": (DummyClassifier(strategy="prior"), {}),
    "Multinomial logistic": (LogisticRegression(max_iter=2000), {"logisticregression__C": [0.01, 0.1, 1, 10]}),
    "Logistic + pairwise interactions": (
        LogisticRegression(max_iter=4000), {"logisticregression__C": [0.01, 0.1, 1]}),
    "Decision tree": (DecisionTreeClassifier(random_state=0),
                      {"decisiontreeclassifier__max_depth": [3, 4, 5, 6, 8],
                       "decisiontreeclassifier__min_samples_leaf": [10, 30]}),
    "Random forest": (RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
                      {"randomforestclassifier__min_samples_leaf": [5, 20, 50]}),
    "Gradient boosting": (HistGradientBoostingClassifier(random_state=0),
                          {"histgradientboostingclassifier__max_depth": [2, 3, 4],
                           "histgradientboostingclassifier__learning_rate": [0.05, 0.1]}),
}
from sklearn.preprocessing import PolynomialFeatures

clf_rows, fitted = [], {}
for name, (est, grid) in cands.items():
    steps = [ohe()]
    if "interactions" in name:
        steps.append(PolynomialFeatures(2, interaction_only=True, include_bias=False))
    pipe = make_pipeline(*steps, est)
    if grid:
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_log_loss", n_jobs=-1).fit(train[FEATS], train.fault)
        model, best = gs.best_estimator_, gs.best_params_
    else:
        model, best = pipe.fit(train[FEATS], train.fault), {}
    P = model.predict_proba(test[FEATS])
    P = pd.DataFrame(P, columns=model.classes_)[CLASSES].values
    pred = np.array(CLASSES)[P.argmax(1)]
    y = test.fault.values
    yl = (y == "Leak").astype(int)
    row = dict(model=name, log_loss=log_loss(y, np.clip(P, 1e-6, 1), labels=CLASSES),
               accuracy=accuracy_score(y, pred), balanced_acc=balanced_accuracy_score(y, pred),
               macro_f1=f1_score(y, pred, average="macro"),
               leak_auc=roc_auc_score(yl, P[:, 2]) if P[:, 2].std() > 0 else 0.5,
               leak_ap=average_precision_score(yl, P[:, 2]), leak_brier=brier_score_loss(yl, P[:, 2]),
               params=json.dumps(best))
    clf_rows.append(row)
    fitted[name] = (model, P)
clf = pd.DataFrame(clf_rows)
base_ll = clf.loc[clf.model == "Class priors (baseline)", "log_loss"].iloc[0]
clf["log_loss_skill_vs_prior"] = 1 - clf.log_loss / base_ll
clf.to_csv(OUT / "q1_classifier_results.csv", index=False)
log("\nQ1 fault-type classifier, held-out test window")
log(clf.drop(columns="params").round(3).to_string(index=False))

best_name = clf[~clf.model.str.contains("baseline")].sort_values("log_loss").model.iloc[0]
best_model, Pbest = fitted[best_name]
log(f"\nbest by test log loss: {best_name}  params {clf.set_index('model').loc[best_name, 'params']}")
pred = np.array(CLASSES)[Pbest.argmax(1)]
cm = pd.DataFrame(confusion_matrix(test.fault, pred, labels=CLASSES), index=CLASSES, columns=CLASSES)
cm.to_csv(OUT / "q1_confusion.csv")
log("confusion (rows=true, cols=predicted)\n" + cm.to_string())
log("per-class recall " + str(dict(zip(CLASSES, (np.diag(cm) / cm.sum(1)).round(3)))))
for i, c in enumerate(CLASSES):
    log(f"  one-vs-rest AUC {c}: {roc_auc_score(test.fault == c, Pbest[:, i]):.3f}")

# theoretical ceiling: predict the empirical fault mix of each exact build cell (fit on train)
key = ["facility", "supplier", "rev", "shift"]
cell = train.groupby(key).fault.value_counts(normalize=True).unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
Pc = test[key].merge(cell, left_on=key, right_index=True, how="left")[CLASSES].fillna(0).values + 1e-3
Pc = Pc / Pc.sum(1, keepdims=True)
log(f"lookup-table of build-cell fault mix (train): test log loss {log_loss(test.fault, Pc, labels=CLASSES):.3f}")

# permutation importance of each feature for the best model (log loss increase)
rng = np.random.default_rng(0)
imp = {}
base = log_loss(test.fault, np.clip(Pbest, 1e-6, 1), labels=CLASSES)
for f in FEATS:
    vals = []
    for _ in range(30):
        t2 = test[FEATS].copy()
        t2[f] = rng.permutation(t2[f].values)
        P2 = pd.DataFrame(best_model.predict_proba(t2), columns=best_model.classes_)[CLASSES].values
        vals.append(log_loss(test.fault, np.clip(P2, 1e-6, 1), labels=CLASSES) - base)
    imp[f] = (np.mean(vals), np.std(vals))
impdf = pd.DataFrame(imp, index=["mean_logloss_increase", "sd"]).T.sort_values("mean_logloss_increase", ascending=False)
impdf.to_csv(OUT / "q1_permutation_importance.csv")
log("permutation importance (test log loss increase)\n" + impdf.round(4).to_string())

# save test-set leak probability for the flagging decision
test_out = test[["Serial Number", "fault", "field_cost", "family"] + FEATS[1:]].copy()
test_out["p_leak"] = Pbest[:, 2]
for i, c in enumerate(CLASSES):
    test_out[f"p_{c}"] = Pbest[:, i]
test_out.to_csv(OUT / "q1_test_predictions.csv", index=False)

# leak model sensitivity to contested labels: refit with rework-label fan definition
alt = d.copy()
alt.loc[alt["Repair Type"] == "Fan Rotation Correction", "fault"] = "Fan Rotation"
tr2, te2 = alt[alt.Date < "2025-10-01"], alt[alt.Date >= "2025-10-01"]
m2 = make_pipeline(ohe(), LogisticRegression(max_iter=2000, C=1)).fit(tr2[FEATS], tr2.fault)
P2 = pd.DataFrame(m2.predict_proba(te2[FEATS]), columns=m2.classes_)[CLASSES].values
pri = tr2.fault.value_counts(normalize=True).reindex(CLASSES).values
log(f"sensitivity (rework-defined fan labels): logistic test log loss {log_loss(te2.fault, np.clip(P2, 1e-6, 1), labels=CLASSES):.3f}"
    f" vs prior {log_loss(te2.fault, np.tile(pri, (len(te2), 1)), labels=CLASSES):.3f}")

# ============================== Q2 cost regression ===============================
log("\nQ2 expected field repair cost per failed unit, held-out test window")
reg_rows = []
y_tr, y_te = train.field_cost.values, test.field_cost.values


def reg_metrics(name, pred):
    reg_rows.append(dict(model=name, MAE=mean_absolute_error(y_te, pred),
                         RMSE=mean_squared_error(y_te, pred) ** 0.5, R2=r2_score(y_te, pred)))


reg_metrics("Global mean (baseline)", DummyRegressor().fit(train[FEATS], y_tr).predict(test[FEATS]))
reg_metrics("Mean by product family (baseline)",
            test.family.map(train.groupby("family").field_cost.mean()).values)
ridge = GridSearchCV(make_pipeline(ohe(), Ridge()), {"ridge__alpha": [0.1, 1, 10, 100]},
                     cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_squared_error").fit(train[FEATS], y_tr)
reg_metrics("Ridge regression", ridge.predict(test[FEATS]))
gbr = GridSearchCV(make_pipeline(ohe(), HistGradientBoostingRegressor(random_state=0)),
                   {"histgradientboostingregressor__max_depth": [2, 3, 4],
                    "histgradientboostingregressor__learning_rate": [0.05, 0.1]},
                   cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_squared_error").fit(train[FEATS], y_tr)
reg_metrics("Gradient boosting", gbr.predict(test[FEATS]))
# two-stage: P(fault | build) x mean cost(fault, LegacyX or not) from train
train["legacy"] = train.family.eq("LegacyX")
test["legacy"] = test.family.eq("LegacyX")
cmean = train.groupby(["legacy", "fault"]).field_cost.mean()
C = np.array([[cmean[(lg, c)] for c in CLASSES] for lg in test.legacy])
two = (Pbest * C).sum(1)
reg_metrics("Two-stage: P(fault) x cost(fault, family)", two)
reg = pd.DataFrame(reg_rows)
reg.to_csv(OUT / "q2_regression_results.csv", index=False)
log(reg.round(3).to_string(index=False))
test_out["exp_cost_two_stage"] = two
test_out.to_csv(OUT / "q1_test_predictions.csv", index=False)

# ============================== Q3 clustering ====================================
log("\nQ3 clustering build configurations (facility x supplier x rev x shift) by fault mix")
prof = d.groupby(key).fault.value_counts(normalize=True).unstack(fill_value=0)[CLASSES]
prof["n"] = d.groupby(key).size()
prof["mean_cost"] = d.groupby(key).field_cost.mean()
Xc = prof[CLASSES].values
sil = {}
for k in range(2, 9):
    km = KMeans(k, n_init=20, random_state=0).fit(Xc, sample_weight=prof.n)
    sil[k] = silhouette_score(Xc, km.labels_)
log("silhouette by k: " + json.dumps({k: round(v, 3) for k, v in sil.items()}))
kbest = max(sil, key=sil.get)
km = KMeans(kbest, n_init=20, random_state=0).fit(Xc, sample_weight=prof.n)
prof["cluster"] = km.labels_
agg = AgglomerativeClustering(kbest).fit(Xc)
from sklearn.metrics import adjusted_rand_score
log(f"k={kbest}; agreement with hierarchical clustering (ARI) = {adjusted_rand_score(km.labels_, agg.labels_):.3f}")
summ = prof.reset_index().groupby("cluster").apply(lambda g: pd.Series({
    "cells": len(g), "units": g.n.sum(),
    **{c: np.average(g[c], weights=g.n) for c in CLASSES},
    "mean_cost": np.average(g.mean_cost, weights=g.n),
    "facilities": ",".join(sorted(g.facility.str[-1].unique())),
    "suppliers": ",".join(sorted(g.supplier.str[-1].unique())),
    "revs": ",".join(sorted(g.rev.str[-1].unique())),
    "shifts": ",".join(sorted(g["shift"].unique()))}), include_groups=False)
summ.to_csv(OUT / "q3_clusters.csv")
prof.to_csv(OUT / "q3_cell_profiles.csv")
log(summ.round(3).to_string())

open(OUT / "03_models_report.txt", "w").write("\n".join(lines))
print("\n".join(lines))
