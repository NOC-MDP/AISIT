
import numpy as np
import pandas as pd
import plotly.express as px
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# --- Load CSV ---
df = pd.read_csv("../input_data/NASA_Global_Seawater_Oxygen-18_Database_clean.csv")

# --- Clean data ---
# Replace missing or placeholder values (like '**') with NaN
df = df.replace("**", np.nan)
df = df.dropna(subset=["Longitude", "Latitude", "Salinity", "d18O"])

# Convert columns to numeric if necessary
df["Longitude"] = pd.to_numeric(df["Longitude"])
df["Latitude"] = pd.to_numeric(df["Latitude"])
df["Salinity"] = pd.to_numeric(df["Salinity"])
df["d18O"] = pd.to_numeric(df["d18O"])
df["pTemperature"] = pd.to_numeric(df["pTemperature"])
df["datetime"] = pd.to_datetime(df["Year"])

# --- Polar stereographic projection ---
fig_map = px.scatter_geo(
    df,
    lon="Longitude",
    lat="Latitude",
    color="Year",
    hover_name="Reference",
    title="Arctic Sampling Locations from NASA GISS",
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
    df["Longitude"],
    df["Latitude"],
    c=df["Year"].values.astype(float),
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
plt.title("Arctic Sampling Locations from NASA GISS", fontsize=12)

plt.show()


# --- 2️⃣ Scatter plot: d18O vs. Salinity ---
fig_scatter = px.scatter(
    df,
    x="Salinity",
    y="d18O",
    color=df["datetime"].dt.year,
    hover_data=["Depth", "Latitude", "Longitude", "Reference"],
    title="δ¹⁸O vs. Salinity from NASA GISS",
    # trendline="ols"  # optional: adds regression line
)

fig_scatter.update_layout(xaxis_title="Salinity", yaxis_title="δ¹⁸O (‰)")
fig_scatter.show()

# --- 2️⃣ Scatter plot: Temperature vs. Salinity ---
fig_scatter2 = px.scatter(
    df,
    x="Salinity",
    y="pTemperature",
    color=df["datetime"].dt.year,
    hover_data=["Depth", "Latitude", "Longitude", "Reference"],
    title="Temperature vs. Salinity from NASA GISS",
    # trendline="ols"  # optional: adds regression line
)
fig_scatter2.update_layout(xaxis_title="Salinity", yaxis_title="Temperature")
fig_scatter2.show()
