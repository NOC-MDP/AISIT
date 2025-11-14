
import numpy as np
import pandas as pd
import plotly.express as px
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

latitude_field = "Latitude [degrees North]"
longitude_field = "Longitude [degrees East]"
salinity_field = "SALNTY [PSS-78]"
oxygen_iso_field = "O18/O16 [/MILLE]"
reference_field = "Sample ID:INTEGER"
dt_format = "yyyy-mm-ddThh:mm:ss.sss"
temperature_field = "TEMPERATURE [DEG C]"

# --- Load CSV ---
df = pd.read_csv("input_data/window_data_from_GLODAPv2.2023.txt", header=17)

# --- Clean data ---
# Replace missing or placeholder values (like '**') with NaN
df = df.replace("**", np.nan)
df = df.dropna(subset=[longitude_field, latitude_field, salinity_field, oxygen_iso_field,temperature_field])

# Convert columns to numeric if necessary
df[longitude_field] = pd.to_numeric(df[longitude_field])
df[latitude_field] = pd.to_numeric(df[latitude_field])
df[salinity_field] = pd.to_numeric(df[salinity_field])
df[oxygen_iso_field] = pd.to_numeric(df[oxygen_iso_field])

df["datetime"] = pd.to_datetime(df[dt_format])

# --- Polar stereographic projection ---
fig_map = px.scatter_geo(
    df,
    lon=longitude_field,
    lat=latitude_field,
    color=df["datetime"].dt.year,
    hover_name=reference_field,
    title="Arctic Sampling Locations from GLODAP database",
)

fig_map.update_geos(
    projection_type="stereographic",
    projection_rotation=dict(lat=90, lon=-10, roll=0),  # center on North Pole
    lataxis_range=[50, 90],
    showland=True,
    landcolor="lightgray",
    showocean=True,
    oceancolor="lightblue",
    showcoastlines=True,
    coastlinecolor="black",
)
fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig_map.show()

# --- Set up the projection ---
proj = ccrs.NorthPolarStereo()

# --- Create figure ---
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=proj)

# Add base features
ax.add_feature(cfeature.LAND, color="lightgray")
ax.add_feature(cfeature.OCEAN, color="lightblue")
ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
ax.gridlines(draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")

# Set extent (limit to Arctic)
ax.set_extent([-180, 180, 70, 90], ccrs.PlateCarree())

# --- Plot the sample locations ---
sc = ax.scatter(
    df[longitude_field],
    df[latitude_field],
    c=df["datetime"].dt.year,
    s=40,
    cmap="viridis",
    transform=ccrs.PlateCarree(),
    edgecolor="black",
    linewidth=0.4,
)

# --- Colorbar ---
cb = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.6, pad=0.05)
cb.set_label("Year")

# --- Title ---
plt.title("Arctic Sampling Locations from GLODAP database", fontsize=12)

plt.show()


# --- 2️⃣ Scatter plot: d18O vs. Salinity ---
fig_scatter = px.scatter(
    df,
    x=salinity_field,
    y=oxygen_iso_field,
    color=df["datetime"].dt.year,
    hover_data=[ latitude_field,longitude_field, reference_field],
    title="δ¹⁸O vs. Salinity from GLODAP database",
    # trendline="ols"  # optional: adds regression line
)
fig_scatter.update_layout(xaxis_title="Salinity", yaxis_title="δ¹⁸O (‰)")
fig_scatter.show()

# --- 2️⃣ Scatter plot: Temperature vs. Salinity ---
fig_scatter2 = px.scatter(
    df,
    x=salinity_field,
    y=temperature_field,
    color=df["datetime"].dt.year,
    hover_data=[ latitude_field,longitude_field, reference_field],
    title="Temperature vs. Salinity from GLODAP database",
    # trendline="ols"  # optional: adds regression line
)
fig_scatter2.update_layout(xaxis_title="Salinity", yaxis_title="Temperature")
fig_scatter2.show()