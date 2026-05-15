import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from scipy.spatial import cKDTree
import hdbscan
# -------------------------------------------------------
# Example dataframe
# -------------------------------------------------------

# df columns:
# lon, lat, time, y
df = pd.read_parquet("../data/bas_data/AISIT_Database_1.0.0_uv.parquet")

temp_col  = "Temperature_[cel]"
sal_col   =  "CTD_Salinity_[dimensionless]"
depth_col = "Combined_Depth_[meters_below_surface]"
lon_col   = "Longitude_[degree_east]"
lat_col   = "Latitude_[degree_north]"
dt_col    = "ISO8601_Datetime"
u_col     = 'u'
v_col     = 'v'

# Ensure datetime
df["time"] = pd.to_datetime(df[dt_col])

# -------------------------------------------------------
# Convert time to numeric
# -------------------------------------------------------

t0 = df[dt_col].min()

df["time_days"] = (
    df[dt_col] - t0
).dt.total_seconds() / 86400.0

# -------------------------------------------------------
# Scale space + time
# -------------------------------------------------------

# IMPORTANT:
# Time scaling determines what "nearby" means.
#
# Example:
# 90 days ~ good starting point for Arctic and d18O
#

X = df[[lon_col, lat_col, "time_days"]].copy()

X["time_days"] = X["time_days"] / 90.0

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------
# Create spatiotemporal clusters
# -------------------------------------------------------



clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100
)

df["cluster"] = clusterer.fit_predict(X_scaled)

# -------------------------------------------------------
# Hold out entire clusters
# -------------------------------------------------------

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(
        df,
        groups=df["cluster"]
    )
)

train_df = df.iloc[train_idx].copy()
test_df  = df.iloc[test_idx].copy()

# -------------------------------------------------------
# Optional: spatial buffer removal
# -------------------------------------------------------
#
# Remove train points too close to test points
# to reduce leakage.
#

BUFFER_DEGREES = 0.25

train_xy = train_df[[lon_col, lat_col]].values
test_xy  = test_df[[lon_col, lat_col]].values

tree = cKDTree(test_xy)

dist, _ = tree.query(train_xy)

buffer_mask = dist > BUFFER_DEGREES

train_df = train_df.iloc[buffer_mask]

# -------------------------------------------------------
# Final sizes
# -------------------------------------------------------

print(f"Train size: {len(train_df)}")
print(f"Test size : {len(test_df)}")

print()
print("Train clusters:", train_df["cluster"].nunique())
print("Test clusters :", test_df["cluster"].nunique())

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------------
# Arctic stereographic plot
# -----------------------------
fig = plt.figure(figsize=(10, 10))

ax = plt.axes(projection=ccrs.NorthPolarStereo())

# Limit to Arctic region (adjust if needed)
ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())

# Add base features
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN)
ax.coastlines(linewidth=0.5)

# Scatter plot
sc = ax.scatter(
    train_df[lon_col],
    train_df[lat_col],
    s=1,
    label="train",
    transform=ccrs.PlateCarree()
)

# Scatter plot
sc = ax.scatter(
    test_df[lon_col],
    test_df[lat_col],
    s=1,
    label="test",
    transform=ccrs.PlateCarree()
)

plt.legend()
plt.savefig("test_spatial_temporal_holdout.png")
plt.show()