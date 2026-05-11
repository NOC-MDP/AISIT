from streamline_tracers import streamlines_to_mass_fraction
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from matplotlib.colors import Normalize


def add_streamline_plot_to_ax(dax, day, axs, x="longitude", y="latitude", **figkwargs):
    lats, lons = dax[y].values, dax[x].values
    lon2d, lat2d = np.meshgrid(lons, lats)
    X, Y = dax.values, day.values
    X_src_crs = X / np.cos(lat2d / 180 * np.pi)
    Y_src_crs = Y
    magnitude = np.sqrt(X**2 + Y**2)
    magn_src_crs = np.sqrt(X_src_crs**2 + Y_src_crs**2)
    bu = X_src_crs * magnitude / magn_src_crs
    bv = Y_src_crs * magnitude / magn_src_crs
    lw = 25 * np.sqrt(bu**2 + bv**2)
    lw = np.nan_to_num(lw, nan=0.0)
    figkwargs = {
        "density": 4,
        "linewidth": lw,
        "arrowsize": 1,
        "color": magnitude,
        "cmap": "viridis",
        "maxlength": 50.0,        # (NPS metres × 1e-6 internally)
        "integration_direction": "forward", 
        **figkwargs,
    }
    p = axs.streamplot(lon2d, lat2d, bu, bv, transform=ccrs.PlateCarree(), **figkwargs)
    cbar = plt.colorbar(p.lines, ax=axs, pad=0.05)
    cbar.set_label("Speed (m/s)")
    axs.set_title(f"Velocity streamlines for year {year}")
    return p


ds = xr.open_dataset(
    "/work/scratch-pw5/thopri/cmems/cmems_mod_arc_phy_my_topaz4_P1M_vyo-vxo_"
    "180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-01-01.nc"
)
year = "1994"

proj_map  = ccrs.NorthPolarStereo()
proj_data = ccrs.PlateCarree()

fig = plt.figure(figsize=(9, 9))
ax  = plt.axes(projection=proj_map)
ax.set_extent([-180, 180, 70, 90], crs=proj_data)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.coastlines(resolution="50m")
ax.gridlines(draw_labels=False, linewidth=0.5, linestyle="--")

u = ds["vxo"].sel(time=year).isel(depth=0).mean("time").compute()
v = ds["vyo"].sel(time=year).isel(depth=0).mean("time").compute()


sp = add_streamline_plot_to_ax(u, v, ax)


lons = ds["longitude"].values
lats = ds["latitude"].values

LON2D, LAT2D = np.meshgrid(lons, lats)

# ── FIX: reproject mouths from lon/lat → NorthPolarStereo (metres) ──────────
mouths_lonlat = np.array([
    [ 72.04,  74.883],
    [ 85.826, 76.382],
    [120.49,  74.82 ],
    [164.2,   70.2  ],
    [-139.4,  70.0  ],
    [-72.667,77.531],
    [-14.373,82.603],
    [-20.985,70.299],
    [41.939,68.254],
    [-154.435,71.520],
    [-124.645,76.653],
    [-94.978,81.619],
    [130.897,74.589],
])
mouths_xy = proj_map.transform_points(
    proj_data,
    mouths_lonlat[:, 0],   # longitudes
    mouths_lonlat[:, 1],   # latitudes
)[:, :2]                   # drop the z column → shape (N, 2)

# ── FIX: reproject the output grid into the same NPS space ──────────────────
flat     = proj_map.transform_points(proj_data, LON2D.ravel(), LAT2D.ravel())
GRID_X   = flat[:, 0].reshape(LON2D.shape)   # easting  (metres)
GRID_Y   = flat[:, 1].reshape(LON2D.shape)   # northing (metres)

# NaN out points that fell outside the projection (e.g. over the other pole)
valid    = np.isfinite(GRID_X) & np.isfinite(GRID_Y)

# ── Now everything is in the same coordinate space as the path vertices ─────
concentration = streamlines_to_mass_fraction(
    sp, mouths_xy, GRID_X, GRID_Y,
    decay_length=10000_000,       # metres  (≈ 500 km e-folding scale)
    mouth_snap_radius=200_000,  # metres  (≈ 200 km catchment)
    max_interp_dist=150_000,   # ← 150 km; tune this
    initial_concentration=1.0,
)

masked = np.ma.masked_where((concentration < 0.01) | ~valid, concentration)

pcm = ax.pcolormesh(
    LON2D, LAT2D, masked,
    cmap="YlOrRd",
    norm=Normalize(vmin=0, vmax=1),
    shading="gouraud",
    zorder=4,
    alpha=0.88,
    transform=ccrs.PlateCarree(),
)
plt.colorbar(pcm, ax=ax, label="Mass fraction  C / C₀", pad=0.02, shrink=0.7)

ax.scatter(
    mouths_lonlat[:, 0], mouths_lonlat[:, 1],
    s=70, c="royalblue", zorder=5,
    edgecolors="white", linewidths=0.6,
    label="River mouth",
    transform=ccrs.PlateCarree(),
)
ax.legend(loc="lower right", fontsize=9)

plt.savefig("streamlines_mass_fraction.png", dpi=150, bbox_inches="tight")
plt.show()


