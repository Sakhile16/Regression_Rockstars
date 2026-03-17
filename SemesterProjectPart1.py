from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --------------------------
# 1) Resolve paths robustly
# --------------------------
def project_root() -> Path:
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    return here if (here / "DataBaseline_v2_Melusi.csv").exists() else here.parent

ROOT = project_root()
csv_path = ROOT / "DataBaseline_v2_Melusi.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"Could not find CSV at: {csv_path}")

print(f"[INFO] Reading: {csv_path}")

# ---------------------------------
# 2) Load CSV with encoding fallback
# ---------------------------------
encodings_to_try = ["utf-8", "cp1252", "latin-1"]
last_err = None
df = None
for enc in encodings_to_try:
    try:
        df = pd.read_csv(csv_path, encoding=enc)
        print(f"[INFO] Loaded with encoding: {enc}")
        break
    except UnicodeDecodeError as e:
        last_err = e
        continue

if df is None:
    print("[ERROR] Failed to read CSV with utf-8/cp1252/latin-1.")
    raise last_err

# ---------------------------------
# 3) Basic structure & health checks
# ---------------------------------
print("\n[HEAD]")
print(df.head(10))

print("\n[INFO]")
df.info()

print("\n[DESCRIBE - numeric]")
print(df.describe(include="number"))

print("\n[MISSING VALUES]")
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])

# ----------------------------------------------------------------
# 4) Prepare the hhr_member_total column
# ----------------------------------------------------------------
HMT = "hhr_member_total"
if HMT not in df.columns:
    raise KeyError(
        f"Column '{HMT}' not found. "
        f"Available columns (sample): {list(df.columns[:15])}"
    )

df[HMT] = pd.to_numeric(df[HMT], errors="coerce")

n_before = len(df)
df_plot = df.dropna(subset=[HMT]).copy()
n_after = len(df_plot)
n_dropped = n_before - n_after
if n_dropped > 0:
    print(f"[WARN] Dropped {n_dropped} rows with non-numeric or missing '{HMT}'.")

# ---------------------------------
# Helper: Show or export plots safely
# ---------------------------------
def safe_show(title: str, fname: str):
    try:
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[WARN] Could not display '{title}' interactively: {e}")
    finally:
        out_path = ROOT / fname
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved figure → {out_path}")

# ---------------------------------
# 5) Histogram (with counts on bars + stats box)
# ---------------------------------
plt.figure(figsize=(7, 4))

# Use integer bins from min..max inclusive
min_val = int(np.floor(df_plot[HMT].min()))
max_val = int(np.ceil(df_plot[HMT].max()))
bins = np.arange(min_val, max_val + 2, 1)  # +2 so the last integer has its own bin edge

# Plot and capture returned values for annotation
counts, bin_edges, patches = plt.hist(
    df_plot[HMT],
    bins=bins,
    edgecolor="black",
    color="#4c78a8"
)

plt.title("Household Size – Histogram")
plt.xlabel("Household Member Total")
plt.ylabel("Count")

# --- Annotate each bar with its count ---
for rect, count in zip(patches, counts):
    if count == 0:
        continue  # skip empty bars to reduce clutter
    # Get bar dimensions
    x = rect.get_x()
    w = rect.get_width()
    h = rect.get_height()
    # Place the label at the center top of the bar
    plt.text(
        x + w / 2,
        h,
        f"{int(count)}",
        ha="center",
        va="bottom",
        fontsize=9
    )

# --- Compute summary stats (drop NaNs just in case) ---
vals = df_plot[HMT].dropna()

mean_val = vals.mean()
median_val = vals.median()

modes = vals.mode()
if len(modes) == 0:
    mode_str = "—"
elif len(modes) == 1:
    mode_str = f"{int(modes.iloc[0]) if float(modes.iloc[0]).is_integer() else round(modes.iloc[0], 2)}"
else:
    # List all modes as comma-separated values (useful for discrete counts)
    mode_str = ", ".join(str(int(m)) if float(m).is_integer() else str(round(float(m), 2)) for m in modes.tolist())

min_v = vals.min()
max_v = vals.max()

# Format integers nicely (most household sizes are integers)
def fmt_num(x):
    return f"{int(x)}" if float(x).is_integer() else f"{x:.2f}"

stats_text = (
    f"Mean: {mean_val:.2f}\n"
    f"Median: {fmt_num(median_val)}\n"
    f"Mode: {mode_str}\n"
    f"Min: {fmt_num(min_v)}\n"
    f"Max: {fmt_num(max_v)}"
)

# --- Add a stats textbox at top-right of the axes ---
ax = plt.gca()
ax.text(
    0.98, 0.98,               # x, y in axes fraction coordinates (top-right)
    stats_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#333333")
)


# ---------------------------------
# 6) Boxplot
# ---------------------------------
plt.figure(figsize=(7, 2.8))
plt.boxplot(df_plot[HMT].dropna(), vert=False, patch_artist=True,
            boxprops=dict(facecolor="#4c78a8"))
plt.title("Household Size – Boxplot")
plt.xlabel("Household Member Total")

# ======================================================================
# 7) NEW — Pie Chart: Total household members by sex (with actual numbers + total)
# ======================================================================
print("\n[INFO] Creating Pie Chart for hhr_member_total by sex...")

sex_column = "sex"  # adjust if needed

if sex_column not in df.columns:
    raise KeyError(f"Column '{sex_column}' not found. Available columns: {df.columns.tolist()}")

# Clean & standardize sex column
df[sex_column] = df[sex_column].astype(str).str.strip().str.title()

# Filter valid rows
df_sex = df.dropna(subset=[sex_column, HMT]).copy()

# Group totals by sex
grouped = df_sex.groupby(sex_column)[HMT].sum()

print("\nTotal Household Members by Sex:")
print(grouped)

# Compute grand total
grand_total = grouped.sum()

# ---- Custom autopct function for percent + actual count ----
def autopct_with_counts(values):
    def inner_autopct(pct):
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}% ({count})"
    return inner_autopct

# ---- Plot ----
plt.figure(figsize=(8, 8))
colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]

wedges, texts, autotexts = plt.pie(
    grouped,
    labels=grouped.index,
    autopct=autopct_with_counts(grouped.values),
    startangle=90,
    colors=colors[:len(grouped)],
    wedgeprops={"edgecolor": "black"},
    textprops={"fontsize": 11}
)

# -----------------------------------------------------------
# Add TOTAL in the chart title
# -----------------------------------------------------------
plt.title(f"Total Individuals by Sex\n(Total = {grand_total})", fontsize=14)

# -----------------------------------------------------------
# Add TOTAL inside the center of the pie (donut style)
# -----------------------------------------------------------
# Make it a donut chart by adding a white circle
centre_circle = plt.Circle((0, 0), 0.65, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Add total text inside the donut
plt.text(
    0, 0,
    f"Total\n{grand_total}",
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold'
)

# ======================================================================
# 8) Scatter Plot of Coordinates
# ======================================================================
print("\n[INFO] Creating scatter plot for coordinates...")

lon_col, lat_col = "Longitude", "Latitude"
missing_cols = [c for c in (lon_col, lat_col) if c not in df.columns]
if missing_cols:
    raise KeyError(
        f"Missing columns: {missing_cols}. "
        f"Available columns (first 20): {list(df.columns[:20])}"
    )

# Convert Lon/Lat to numeric
for c in (lon_col, lat_col):
    df[c] = (
        df[c].astype(str).str.replace(",", ".", regex=False).str.strip()
    )
    df[c] = pd.to_numeric(df[c], errors="coerce")

before = len(df)
df_geo = df.dropna(subset=[lon_col, lat_col]).copy()

df_geo = df_geo[
    (df_geo[lon_col].between(-180, 180)) &
    (df_geo[lat_col].between(-90, 90))
]
after = len(df_geo)

if after == 0:
    raise ValueError("No valid coordinate pairs to plot after cleaning.")

if after < before:
    print(f"[INFO] Filtered out {before - after} invalid/missing coordinate rows.")

ax = df_geo.plot(
    kind="scatter",
    x=lon_col,
    y=lat_col,
    s=8,
    alpha=0.7,
    color="#4c78a8",
    figsize=(6, 6),
    title="Survey Points (Longitude / Latitude)"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect('equal')

plt.tight_layout()
plt.show()

print("\n[INFO] EDA complete.")