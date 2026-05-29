import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

############ Columns ############
# Dataset
# ISO8601_Datetime
# Latitude_[degree_north]
# Longitude_[degree_east]
# CTD_Temperature_WOCE_Flag
# CTD_Pressure_[dbar]
# Depth_From_Pressure_[meters_below_surface]
# Sample_Depth_[meters_below_surface]
# Combined_Depth_[meters_below_surface]
# Delta_O18_[permillle]
# Delta_O18_WOCE_Flag_Translated
# Delta_O18_WOCE_Flag_Original
# O18_Standard
# Barium
# Barium_WOCE_Flag
# Barium_Units
# CTD_Salinity_[dimensionless]
# CTD_Salinity_WOCE_Flag
# Bottle_Salinity_[dimensionless]
# Bottle_Salinity_WOCE_Flag
# Cruise_ID
# Station_ID_Name
# Site_Number_Name
# Sample_ID
# Cast_Number
# Bottle_Number
# Event
# Rosette_Position
# Platform
# BIOPOLE
# Freshwater
# Freshwater_Source
# Land_Based
# Temperature_[cel]

# -----------------------------
# User inputs
# -----------------------------
csv_file = "../data/bas_data/AISIT_Database_1.0.0.parquet"

# CSV column names
lat_name  = "Latitude_[degree_north]"
lon_name  = "Longitude_[degree_east]"
depth_name = "Combined_Depth_[meters_below_surface]"
time_name  = "ISO8601_Datetime"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_parquet(csv_file)

# Parse datetime — use infer so mixed formats are handled gracefully
df["_dt"] = pd.to_datetime(df[time_name], infer_datetime_format=True)

# Numeric representation of date for colourmapping (matplotlib date number)
df["_date_num"] = mdates.date2num(df["_dt"])

# -----------------------------
# Helper: build one Arctic map
# -----------------------------
def make_arctic_ax(fig, position=111):
    ax = fig.add_subplot(position, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())

    # Draw ocean first, then land on top — this is the fix for the
    # all-blue map: zorder must place scatter ABOVE both features.
    ax.add_feature(cfeature.OCEAN, facecolor="#cde4f0", zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor="#d9d4c7", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.gridlines(color="white", linewidth=0.4, linestyle="--", zorder=2)
    return ax


# ============================================================
# Figure 1 — Barium
# ============================================================
fig1 = plt.figure(figsize=(8, 8))
ax1  = make_arctic_ax(fig1)

barium = df["Barium"]

sc1 = ax1.scatter(
    df[lon_name].values,
    df[lat_name].values,
    c=barium,
    s=5,
    cmap="viridis",
    transform=ccrs.PlateCarree(),
    zorder=3,           # ← must be higher than land/ocean zorders
)

cb1 = plt.colorbar(sc1, orientation="vertical", shrink=0.7, pad=0.05, ax=ax1)
cb1.set_label("Ba (nmol kg⁻¹)")
ax1.set_title("Barium Observations", fontsize=14, pad=12)

plt.tight_layout()
fig1.savefig("Ba_scatter_map.png", dpi=150, bbox_inches="tight")
print("Saved Ba_scatter_map.png")


# ============================================================
# Figure 2 — Observation date (barium rows only)
# ============================================================
df_ba = df[df["Barium"].notna()]   # keep only rows with a Barium measurement

fig2 = plt.figure(figsize=(8, 8))
ax2  = make_arctic_ax(fig2)

sc2 = ax2.scatter(
    df_ba[lon_name].values,
    df_ba[lat_name].values,
    c=df_ba["_date_num"].values,
    s=5,
    cmap="plasma",
    transform=ccrs.PlateCarree(),
    zorder=3,
)

# Build a colourbar with human-readable date labels
cb2 = plt.colorbar(sc2, orientation="vertical", shrink=0.7, pad=0.05, ax=ax2)
cb2.set_label("Observation date")

# Choose ~6 evenly-spaced tick positions across the data range
date_min = df_ba["_date_num"].min()
date_max = df_ba["_date_num"].max()
tick_nums = np.linspace(date_min, date_max, 6)
cb2.set_ticks(tick_nums)
cb2.set_ticklabels(
    [mdates.num2date(t).strftime("%Y") for t in tick_nums]
)

ax2.set_title("Observation Dates", fontsize=14, pad=12)

plt.tight_layout()
fig2.savefig("Ba_date_scatter_map.png", dpi=150, bbox_inches="tight")
print("Saved Ba date_scatter_map.png")

plt.show()