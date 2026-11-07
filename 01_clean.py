"""Step 1: join the four datasets on Serial Number and reconcile failure labels.

Each source names the same defect differently. We map every source label to one of
four root fault classes, then take a 2-of-3 majority vote across field service (D1),
manufacturing rework (D4) and EOLT (D5).
"""
import pandas as pd
from pathlib import Path

D = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

field = pd.read_excel(D / "D1_field.xlsx")
config = pd.read_excel(D / "D3_config.xlsx")
rework = pd.read_excel(D / "D4_rework.xlsx")
eolt = pd.read_excel(D / "D5_eolt.xlsx")

# ---- integrity checks -------------------------------------------------------
sets = [set(x["Serial Number"]) for x in (field, config, rework, eolt)]
assert all(s == sets[0] for s in sets), "serials differ across files"
assert all(x["Serial Number"].is_unique for x in (field, config, rework, eolt))

df = (field.merge(config, on="Serial Number", suffixes=("", "_cfg"))
           .merge(rework, on="Serial Number", suffixes=("", "_rw"))
           .merge(eolt, on="Serial Number", suffixes=("", "_eolt")))
checks = {
    "Product Family field==config": (df["Product Family"] == df["Product Family_cfg"]).mean(),
    "Facility config==rework": (df["Facility"] == df["Facility_rw"]).mean(),
    "Facility config==eolt": (df["Facility"] == df["Facility_eolt"]).mean(),
    "Shift rework==eolt": (df["Shift"] == df["Shift_eolt"]).mean(),
}
df = df.drop(columns=["Product Family_cfg", "Facility_rw", "Facility_eolt", "Shift_eolt"])

# ---- label reconciliation ---------------------------------------------------
FIELD_MAP = {
    "Refrigerant Leak": "Leak", "Suction Leak": "Leak", "Service Valve Leak": "Leak",
    "Sensor Fault": "Controls", "Communication Fault": "Controls",
    "Loose Power Wiring": "Power Wiring", "Condenser Fan Backwards": "Fan Rotation",
}
REWORK_MAP = {"Leak Repair": "Leak", "Sensor Rewire": "Controls",
              "Rewire": "Power Wiring", "Fan Rotation Correction": "Fan Rotation"}
EOLT_MAP = {"Refrigerant Leak": "Leak", "Sensor Miswire": "Controls",
            "Loose Terminal": "Power Wiring", "Fan Rotation Error": "Fan Rotation"}

df["fault_field"] = df["Failure Description"].map(FIELD_MAP)
df["fault_rework"] = df["Repair Type"].map(REWORK_MAP)
df["fault_eolt"] = df["Failure Mode"].map(EOLT_MAP)
assert df[["fault_field", "fault_rework", "fault_eolt"]].notna().all().all()


def vote(row):
    v = pd.Series([row.fault_field, row.fault_rework, row.fault_eolt]).value_counts()
    return v.index[0] if v.iloc[0] >= 2 else None


df["fault"] = df.apply(vote, axis=1)
df["agree_all3"] = (df.fault_field == df.fault_rework) & (df.fault_rework == df.fault_eolt)
assert df["fault"].notna().all(), "some units have no 2-of-3 majority"

# ---- tidy features ----------------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"])
df["month"] = df["Date"].dt.to_period("M").astype(str)
df["is_leak"] = (df["fault"] == "Leak").astype(int)
df = df.rename(columns={"Product Family": "family", "Facility": "facility",
                        "Controls Revision": "rev", "Coil Supplier": "supplier",
                        "Shift": "shift", "Customer Region": "region",
                        "Repair Cost": "field_cost", "Downtime Hours": "downtime_hrs",
                        "Labor Hours": "rework_labor_hrs", "Material Cost": "rework_material",
                        "Rework Hours": "eolt_rework_hrs", "Test Station": "eolt_station",
                        "Disposition Code": "disposition"})
df.to_csv(OUT / "clean.csv", index=False)

# ---- reports ----------------------------------------------------------------
with open(OUT / "01_clean_report.txt", "w") as fh:
    fh.write(f"rows {len(df)}, unique serials {df['Serial Number'].nunique()}\n")
    for k, v in checks.items():
        fh.write(f"{k}: {v:.3f}\n")
    fh.write(f"\nunits where all 3 sources agree: {df.agree_all3.mean():.3f}\n")
    for src in ["fault_field", "fault_rework", "fault_eolt"]:
        fh.write(f"{src} agrees with reconciled fault: {(df[src] == df.fault).mean():.3f}\n")
    fh.write("\nreconciled fault counts\n" + df.fault.value_counts().to_string() + "\n")
    fh.write("\nfield label vs reconciled\n" + pd.crosstab(df["Failure Description"], df.fault).to_string() + "\n")
    fh.write("\nEOLT label vs reconciled\n" + pd.crosstab(df["Failure Mode"], df.fault).to_string() + "\n")
    fh.write("\nrework label vs reconciled\n" + pd.crosstab(df["Repair Type"], df.fault).to_string() + "\n")
    fh.write("\nEOLT station vs reconciled fault (station looks uninformative)\n"
             + pd.crosstab(df.eolt_station, df.fault).to_string() + "\n")
    fh.write(f"\nservice dates {df.Date.min().date()} to {df.Date.max().date()}\n")
print(open(OUT / "01_clean_report.txt").read())
