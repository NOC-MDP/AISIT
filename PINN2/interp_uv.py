import pandas as pd
import xarray as xr

# -----------------------------
# User inputs
# -----------------------------
csv_file = "../data/bas_data/pre_alpha_AISIT_0.1.4_new_headers.csv"
output_file = "../data/bas_data/pre_alpha_AISIT_0.1.4_new_headers_uv.csv"
dataset_file = "/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_vyo-vxo_monthly_climatology.nc"  # or .zarr

# CSV column names
lat_name = "lat"
lon_name = "lon"
depth_name = "depth"
time_name = "year"   # datetime string column

# Dataset coordinate names
ds_lat = "latitude"
ds_lon = "longitude"
ds_depth = "depth"
ds_time = "month"

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(csv_file)

# Convert time column to datetime
dt = pd.to_datetime(df[time_name], format="%d/%m/%Y")
dt = dt.dt.month

# -----------------------------
# Load dataset
# -----------------------------
if dataset_file.endswith(".zarr"):
    ds = xr.open_zarr(dataset_file)
else:
    ds = xr.open_dataset(dataset_file)

# -----------------------------
# Build vectorised query dataset
# -----------------------------
points = xr.Dataset(
    {
        ds_lat: (("points",), df[lat_name].values),
        ds_lon: (("points",), df[lon_name].values),
        ds_depth: (("points",), df[depth_name].values),
        ds_time: (("points",), dt.values),
    }
)

# -----------------------------
# Interpolate (4D)
# -----------------------------
sampled = ds.interp(
    {
        ds_lat: points[ds_lat],
        ds_lon: points[ds_lon],
        ds_depth: points[ds_depth],
        ds_time: points[ds_time],
    },
    method="linear"
)

# -----------------------------
# Extract u and v
# -----------------------------
df["u"] = sampled["vxo"].values
df["v"] = sampled["vyo"].values

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------------
# Compute speed
# -----------------------------
speed = np.sqrt(df["u"].values**2 + df["v"].values**2)

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
    df[lon_name].values,
    df[lat_name].values,
    c=speed,
    s=5,
    cmap="viridis",
    transform=ccrs.PlateCarree()
)

# Colourbar
cb = plt.colorbar(sc, orientation="vertical", shrink=0.7, pad=0.05)
cb.set_label("Speed (m/s)")

plt.title("Arctic Current Speed (|u,v|)")
plt.savefig("current_scatter_map.png")
plt.show()

# -----------------------------
# Save output
# -----------------------------
df.to_csv(output_file, index=False)

print(f"Saved output to {output_file}")

