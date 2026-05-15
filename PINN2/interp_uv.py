import pandas as pd
import xarray as xr


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
output_file = "../data/bas_data/AISIT_Database_1.0.0_uv.parquet"
dataset_file = "/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_vyo-vxo_monthly_climatology.nc"  # or .zarr

# CSV column names
lat_name = "Latitude_[degree_north]"
lon_name = "Longitude_[degree_east]"
depth_name = "Combined_Depth_[meters_below_surface]"
time_name = "ISO8601_Datetime"   # datetime string column

# Dataset coordinate names
ds_lat = "latitude"
ds_lon = "longitude"
ds_depth = "depth"
ds_time = "month"

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_parquet(csv_file)

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
df.to_parquet(output_file, index=False)

print(f"Saved output to {output_file}")

