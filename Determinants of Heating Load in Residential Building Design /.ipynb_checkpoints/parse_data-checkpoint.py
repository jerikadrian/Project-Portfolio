import pandas as pd

RAW_CSV = "FP_DATA.csv"
df0 = pd.read_csv(RAW_CSV)

df0 = df0.dropna(how="all")

keep_cols = ["X1","X2","X3","X4","X5","X6","X7","X8","Y1"]
missing = [c for c in keep_cols if c not in df0.columns]
if missing:
    raise ValueError(
        f"Missing expected columns in {RAW_CSV}: {missing}. "
        f"Found columns: {list(df0.columns)}"
    )

df = df0[keep_cols].copy()

df = df.rename(columns={
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
    "Y1": "heating_load"
})

for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

X = df.drop(columns=["heating_load"])
y = df["heating_load"]
