# Airedale Data Science Challenge: findings and method

Everything here rebuilds from the four raw files with `./run_all.sh` (about 5 minutes). Python 3, pandas, scikit-learn, scipy, matplotlib. Numbers below come from `out/*.txt` and `out/*.csv`.

## Problem formulation

The business question (which units fail how, what it costs, and where $750K does the most good) splits into five questions:

| Question | Target | Unit | Method | Scored by |
|---|---|---|---|---|
| Which build variables go with each fault? | Fault class | Serial (3,000) | Fisher exact tests, Wilson 95% CIs | p-value, CI |
| Which fault will a shipped unit have if it fails? | 4-class fault | Serial | Logistic, tree, forest, boosting vs class priors | Log loss, AUC, balanced accuracy |
| What will its field repair cost? | Repair $ | Serial | Ridge, boosting, two-stage vs mean | MAE, RMSE, R² |
| Which build cells behave alike? | Fault mix | Build cell (72) | k-means, hierarchical | Silhouette, ARI |
| Which investments within $750K? | 3-year net savings | Portfolio (178) | Enumeration, Monte Carlo | Net $, P(best) |

Not modelled: whether a unit fails. Every serial in the data failed, so there are no negatives.

## Data notes to state on a slide

- Dataset 2 (Warranty) was not in our files. Every field event is warranty eligible, so field repair cost stands in for warranty cost.
- All 3,000 serials appear exactly once in every file. Every unit failed EOLT, got reworked, and then failed in the field. There are no good units, so we can't model whether a unit fails, and FPY or DPMO can't be computed honestly. Every rate below is a share of failures.
- Service dates run Jan 2025 to Jan 2026, with no drift by month (chi-square p = 0.32). Build date is not given, so we split on service date.

## 1. Label reconciliation (Validity)

The three sources name the same defect differently. We mapped each one to 4 fault classes (Leak, Power Wiring, Fan Rotation, Controls) and took a 2-of-3 majority vote.

- All three sources agree on 91% of units. Field agrees with the vote 97.2% of the time, EOLT 97.1%, rework 96.7%.
- The disagreement is almost all fan rotation. Rework logged 366 fan corrections, all on night shift. Field calls 181 of them Loose Power Wiring, and EOLT calls 185 of them Loose Terminal.
- 98 night-shift units stay contested (rework says fan, field and EOLT say wiring). We kept the vote and reran the model with rework's definition. The conclusions held (log loss 0.707 vs prior 1.137).
- EOLT test station is unrelated to the fault found (chi-square p = 0.80). The leak test flags sensor miswires as often as leaks. Treat station as noise.

## 2. Inference: what each fault is associated with (Insight)

| Fault | Build condition | Share of failures in that group | Elsewhere | Test |
|---|---|---|---|---|
| Leak | Facility 2 OR Supplier B coils | 788 of 1,430 (55%; 54 to 57% by cell) | 0 of 1,570 | Fisher p < 1e-140 |
| Controls | Controls Rev A | 283 of 894, 31.7% [28.7, 34.8] | 0 of 2,106 on Rev B/C | Fisher p = 2e-165 |
| Fan Rotation | Night shift | 268 of 1,199, 22.4% [20.1, 24.8] | 0 of 1,801 on day shift | Fisher p = 4e-116 |

- Groups differ in size, so every comparison is a rate within its group, with Wilson 95% CIs and Fisher's exact test.
- These are associations from observational data, not proven causes. To confirm before spending: run Rev A beside Rev B on one line and compare controls faults; tear down failed Supplier B coils from Facilities 1, 3 and 4; audit night-shift against day-shift fan wiring.
- The two leak sources don't stack. Fac 2 + Sup B is 57%, and either one alone is 54 to 55% (p = 0.71). Fixing only one source leaves the 248 both-source units exposed.
- Leak subtype (refrigerant, suction, service valve) mix is the same for both sources (p = 0.22), so subtype can't tell a braze leak from a coil leak.
- Rev A is still about 30% of units every month, so the controls problem is ongoing.
- Cost model, log(cost) OLS with bootstrap CIs (R² 0.90): LegacyX costs 1.52x [1.45, 1.60] for the same fault. That's a $318K a year premium. Region, facility, supplier, and revision have no effect on cost once fault is known. Night shift carries a small premium, 1.06x [1.02, 1.09].
- Money: leaks are 26% of failures but $2.89M of the $3.83M field cost (76%).
- Null results worth showing: region, product family, disposition code, and EOLT station don't predict fault type.

## 3. Predictive models (Rigor, Evaluation, Predictive power)

Unit = one serial. Features = what's known at shipment (family, facility, supplier, revision, shift, region). Rework and EOLT labels are left out because they share the target's defect label (leakage). Train = Jan to Sep 2025 (n = 2,252), test = Oct 2025 to Jan 2026 (n = 748). Hyperparameters were tuned with 5-fold stratified CV on the training window only.

**Q1. Which fault will a failed unit have?** (4-class)

| Model | Test log loss | Skill vs priors | Balanced acc. | Accuracy | Leak AUC |
|---|---|---|---|---|---|
| Class priors (baseline) | 1.094 | 0% | 0.25 | 0.563 | 0.50 |
| Multinomial logistic | 0.694 | 37% | 0.43 | 0.639 | 0.86 |
| Logistic + pairwise interactions | 0.663 | 39% | 0.46 | 0.650 | 0.86 |
| Random forest | 0.661 | 40% | 0.46 | 0.662 | 0.86 |
| Gradient boosting | 0.637 | 42% | 0.48 | 0.664 | 0.86 |
| Decision tree (depth 4) | 0.633 | 42% | 0.41 | 0.659 | 0.86 |

- Log loss is the main metric because the decisions rank units by probability. Accuracy rewards always guessing power wiring (56%).
- **Caveat on model selection.** `03_models.py` names the "best" model (the depth-4 tree) by test log loss, which is selection on the test set. The tree and gradient boosting differ by 0.004, so no downstream result depends on the pick. The leak scores, AUCs and permutation importance below all come from the tree.
- One-vs-rest AUC: Controls 0.93, Fan 0.89, Leak 0.86, Power Wiring 0.81.
- **Three flags are the whole signal.** An 8-cell model built only from "Facility 2 or Supplier B?", "Rev A?" and "night shift?", fit on the training months, scores 0.632 (skill 42.3%, 95% CI 39.6 to 45.1%), accuracy 0.666, leak AUC 0.85. It ties the tree. All three rules hold in the training months alone with zero exceptions, so the flags are not learned from the test window. Because a pre-set model ties the best fitted one, the test-set model pick changes nothing. See `06_rule_check.py` and `out/06_rule_check_report.txt`.
- Correction to an earlier claim: the 72-cell lookup fit on training data (0.665) is not a ceiling. The tree beats it because it pools cells that behave alike. An optimistic in-sample bound (each test cell's own mix, fit on the test set) is 0.550.
- Bootstrap 95% CIs on the test window: classifier skill 42.2% [39.3, 45.2]; two-stage cost MAE reduction vs the training mean 33.3% [29.2, 37.7].
- Leak probabilities are reasonably calibrated: units scored about 0.56 leak at 55%; units scored 0 never leak.
- Capture-curve shares are among failed units only. What transfers to the plant is the rule (Facility 2 and Supplier B), not the 30% figure.
- Permutation importance: supplier > facility > revision > shift. Region and family add nothing.
- Be honest about accuracy. Controls and fan are never the single most likely class in any cell (max 32 to 38%), so argmax never predicts them. Use the probabilities, not hard labels.

**Q2. Expected repair cost per failed unit**

| Model | MAE | R² |
|---|---|---|
| Global mean (baseline) | $1,310 | 0.00 |
| Mean by family (baseline) | $1,307 | 0.02 |
| Ridge | $914 | 0.33 |
| Gradient boosting | $885 | 0.37 |
| Two-stage P(fault) x cost(fault, family) | $875 | 0.36 |

**Q3. Clustering.** 72 build cells (facility x supplier x revision x shift), clustered on their fault mix. Silhouette is best at k = 4 (0.69), and hierarchical clustering gives the identical split (ARI 1.0).

| Segment | Units | Profile | Mean cost |
|---|---|---|---|
| Leak-exposed (Fac 2 or Sup B) | 1,430 | 55% leak | $2,208 |
| Rev A controls | 480 | 44% controls | $757 |
| Night fan | 372 | 38% fan | $320 |
| Clean (wiring only) | 718 | 98% power wiring | $258 |

## 4. Decision impact

Cost pools: field cost ($3.83M) plus rework (labor at an assumed $45/hr, plus material, plus EOLT rework hours, about $1.51M), for $5.34M a year in total. Each catalog item is mapped to the fault classes it targets. Benefits stack multiplicatively on each unit, and nothing saves money before its implementation time. We use a 3-year horizon with no discounting.

- **Best portfolio at catalog values: Smart Torque Tools + Wiring Poka-Yoke + Predictive Quality Dashboard + Automated Leak Detection = $725K.** Year-1 savings $1.04M, run rate $2.14M a year, 3-year net $4.60M (ROI 6.3x).
- **Failures avoided (estimate):** about 1,100 of 3,000 field failures a year at steady state for this portfolio, and about 1,250 for the Torque + Vision + Leak Detection fallback. The fallback avoids more failures but saves less money, because it does less against leaks. This applies each item's field-side reduction to failure counts instead of cost; some catalog benefits are stated as cost reductions, so treat it as approximate. See `out/p_failures_avoided.csv`.
- Monte Carlo (2,000 draws, benefit scale by risk level: Low ±25%, Medium ±40%, High ±60%, plus 0 to 3 month delays): this portfolio is best in 80% of draws, its 5th percentile is $3.77M, and it never loses money. The draws compare the 40 best portfolios at catalog values.
- The pick doesn't change for labor at $30 to $70/hr or for every benefit cut to 50%. A uniform cut barely changes the ranking, so this test is weak. The one-item tests below matter more.
- **The pick depends on the Dashboard.** We apply its "15% reduction in quality escapes" to all field cost, because every unit in the data escaped. That is $574K a year, as much as the EOLT Station for a third of the price. At half that benefit the Dashboard drops out entirely, and the best portfolio becomes **Smart Torque Tools + Vision Inspection + Automated Leak Detection ($725K, 3-year net $4.36M)**. The Monte Carlo can't show this because the Medium-risk range never goes below 60% of catalog. Present the recommendation as conditional on the Dashboard, with Torque + Vision + Leak Detection as the fallback.
- Braze Training has the best standalone ROI, but adding it would push us to $775K.
- LegacyX Refresh (34% ROI) and Design Validation (72%) rank last. The 12-month lag kills Refresh's year 1.
- **Key judgment call.** Every field leak unit had already failed EOLT as a leak and passed after repair. That suggests detection isn't the gap. If Leak Detection and the EOLT Station only deliver half their catalog benefit, the best portfolio flips to prevention: Braze + Torque + Dashboard + Vision + Supplier ($750K, 3-year net $4.21M). The same argument applies to the Dashboard, which is also a detection tool. With the Dashboard also at half benefit, the best is Braze + Torque + Poka-Yoke + Vision + Supplier ($725K, 3-year net $3.66M).
- **Safe core: Smart Torque Tools.** It is the only investment in the best portfolio under every scenario we ran (catalog values, labor rate, uniform cuts, detection at half, Dashboard at half or zero). Vision Inspection is in every best portfolio once the Dashboard is cut.
- Controls is a Rev A problem, but EOLT and rework both log it as Sensor Miswire, so we credit Wiring Poka-Yoke against it. On the slide, frame it as a miswire that Rev A's design allows.
- **Model-targeted inspection.** Ranking held-out units by model leak risk captures 100% of leak cost in the top 49% of units, and 65% in the top 30% (random would be 30%). That means a leak check only has to cover the Facility 2 line and incoming Supplier B coils.
- **No-cost actions outside the catalog:** stop building Rev A (Rev B and C have zero controls failures), audit night-shift fan wiring, and open a Supplier B corrective action.

## 5. Recommendation

Because the catalog-value pick depends on two uncertain items (the Dashboard and Leak Detection), we recommend staging the spend:

- **Approve now, $100K: Smart Torque Tools.** It is in the best portfolio under every scenario we ran and is live in 3 months.
- **Pilot, then fund, $625K.** Buy Automated Leak Detection ($350K) only if a trial on Facility 2 and Supplier B units finds leaks that EOLT misses; otherwise put that money into prevention (Braze Training + Supplier Quality Program). Choose Dashboard + Poka-Yoke ($275K) or Vision Inspection ($275K) after a 3-month Dashboard pilot, on escapes per 1,000 units.
- **Free:** stop Rev A, audit night-shift fan wiring, open a Supplier B corrective action.
- **Measure monthly against today's baselines:** leak share at Facility 2 or Supplier B (55%), controls share on Rev A (31.7%), fan share on night shift (22.4%), field cost per failure ($1,275).

## Assumptions and limitations

| Assumption or limit | How we handled it |
|---|---|
| Every unit failed | No negatives, so we model which fault and what cost, not whether a unit fails |
| No warranty file (Dataset 2) | Field repair cost is the warranty proxy; every field event is warranty-eligible |
| Rework labor rate | Assumed $45/hr; tested $30 to $70/hr, same portfolio |
| Catalog benefits | Used as given; Monte Carlo ±25 to 60%; Dashboard and detection each cut to 50% |
| No build date | Time split on service date; no monthly drift (chi-square p = 0.32) |
| 98 contested labels | Refit with rework's labels: log loss 0.707 vs 1.137, same conclusions |
| Model picked on test set | Tree and boosting within 0.004 log loss; no decision depends on the pick |
| Observational data | Associations, not proven causes; confirmation steps in section 2 |
| Money over time | 3-year horizon, undiscounted |

## Presentation

The deck is a Claude artifact, "Airedale Field Failure Analysis". It has 14 main slides with no animations, a talk track in the speaker notes of about 6.5 minutes (922 words), and 3 backup slides.

## Files

- `01_clean.py` join, checks, label vote, outputs `out/clean.csv`
- `02_inference.py` hypothesis tests, cost model
- `03_models.py` classifiers, cost regression, clustering
- `04_portfolio.py` cost pools, 178 feasible portfolios, detection and Dashboard scenarios, Monte Carlo, failures avoided, capture curve
- `05_figs.py` slide charts in `figs/`
- `06_rule_check.py` three-flag rule model, bootstrap CIs, in-sample bound, leak calibration
