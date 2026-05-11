import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================================================
# USER SETTINGS
# ============================================================

ncfile = "/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_vyo-vxo_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-01-01.nc"

time_index = 0
depth_index = 0

dt = 0.05
nt = 250

decay_rate = 0.002
diffusivity = 5.0  # m^2/s (physical diffusion)

use_semi_lagrangian = True  # switch between Euler/upwind vs SL

R = 6371000.0  # Earth radius (m)

# ============================================================
# LOAD DATA
# ============================================================

ds = xr.open_dataset(ncfile)

u = ds["vxo"].isel(time=time_index)
v = ds["vyo"].isel(time=time_index)

if "depth" in u.dims:
    u = u.isel(depth=depth_index)
    v = v.isel(depth=depth_index)

u = u.values
v = v.values

lon = ds["longitude"].values
lat = ds["latitude"].values

if lon.ndim == 1 and lat.ndim == 1:
    lon2d, lat2d = np.meshgrid(lon, lat)
else:
    lon2d, lat2d = lon, lat

# ============================================================
# METRIC GRID (meters)
# ============================================================

lat_rad = np.deg2rad(lat2d)

dx = (np.gradient(lon2d, axis=1) * np.cos(lat_rad) * np.pi / 180.0) * R
dy = (np.gradient(lat2d, axis=0) * np.pi / 180.0) * R

dx = np.where(dx == 0, np.nanmean(dx), dx)
dy = np.where(dy == 0, np.nanmean(dy), dy)

# ============================================================
# CLEAN VELOCITY
# ============================================================

mask = np.isnan(u) | np.isnan(v)

u = np.where(mask, 0, u)
v = np.where(mask, 0, v)

speed = np.sqrt(u**2 + v**2)
u[speed > 10] = 0
v[speed > 10] = 0

# ============================================================
# TRACER INITIALISATION
# ============================================================

ny, nx = u.shape
tracer = np.zeros((ny, nx))

# ============================================================
# RIVERS (proper geographic distance)
# ============================================================

rivers = {
    "Ob": (70.0, 66.5),
    "Yenisei": (86.0, 72.0),
    "Lena": (126.0, 72.5),
    "Mackenzie": (-135.0, 69.0),
}

for name, (rlon, rlat) in rivers.items():

    dist = ( (lon2d - rlon) * np.cos(np.deg2rad(rlat)) )**2 + (lat2d - rlat)**2

    iy, ix = np.unravel_index(np.argmin(dist), dist.shape)

    tracer[iy-2:iy+3, ix-2:ix+3] = 1.0

# ============================================================
# NUMERICAL SCHEMES
# ============================================================

def laplacian(T, dx, dy):
    d2dx2 = (np.roll(T, -1, axis=1) - 2*T + np.roll(T, 1, axis=1)) / (dx**2)
    d2dy2 = (np.roll(T, -1, axis=0) - 2*T + np.roll(T, 1, axis=0)) / (dy**2)
    return d2dx2 + d2dy2

# ============================================================
# ADVECTION LOOP
# ============================================================

for n in range(nt):

    T = tracer.copy()

    if use_semi_lagrangian:
        # ----------------------------------------------------
        # Semi-Lagrangian (backtrace)
        # ----------------------------------------------------

        x_back = np.clip(
            (np.arange(nx)[None, :] - u * dt / dx),
            0, nx-1
        )

        y_back = np.clip(
            (np.arange(ny)[:, None] - v * dt / dy),
            0, ny-1
        )

        x0 = np.floor(x_back).astype(int)
        y0 = np.floor(y_back).astype(int)

        tracer = T[y0, x0]

    else:
        # ----------------------------------------------------
        # Euler + upwind
        # ----------------------------------------------------

        dTdx = np.zeros_like(T)
        dTdy = np.zeros_like(T)

        dTdx[:, 1:] = (T[:, 1:] - T[:, :-1]) / dx[:, 1:]
        dTdy[1:, :] = (T[1:, :] - T[:-1, :]) / dy[1:, :]

        tracer = T - dt * (u * dTdx + v * dTdy)

    # --------------------------------------------------------
    # Diffusion (physical Laplacian)
    # --------------------------------------------------------

    tracer += diffusivity * dt * laplacian(tracer, dx, dy)

    # --------------------------------------------------------
    # Decay
    # --------------------------------------------------------

    tracer *= np.exp(-decay_rate)

    # --------------------------------------------------------
    # Mask + cleanup
    # --------------------------------------------------------

    tracer[mask] = np.nan
    tracer[tracer < 0] = 0

    # --------------------------------------------------------
    # CFL check (optional warning)
    # --------------------------------------------------------

    cfl = np.nanmax(np.sqrt(u**2 + v**2) * dt / np.nanmin([dx, dy]))
    if cfl > 1.5:
        print(f"Warning CFL high: {cfl:.2f}")

# Normalize
tracer /= np.nanmax(tracer)

# ============================================================
# PLOTTING (CARTOPY)
# ============================================================

proj = ccrs.NorthPolarStereo()
data_crs = ccrs.PlateCarree()

fig = plt.figure(figsize=(14, 14))
ax = plt.axes(projection=proj)

ax.set_extent([-180, 180, 50, 90], crs=data_crs)

ax.add_feature(cfeature.LAND, facecolor="0.85")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)

# ------------------------------------------------------------
# FIELD
# ------------------------------------------------------------

pcm = ax.pcolormesh(
    lon2d,
    lat2d,
    tracer,
    transform=data_crs,
    cmap="turbo",
    shading="auto",
)

# ------------------------------------------------------------
# VELOCITY (quiver)
# ------------------------------------------------------------

skip = 6
ax.quiver(
    lon2d[::skip, ::skip],
    lat2d[::skip, ::skip],
    u[::skip, ::skip],
    v[::skip, ::skip],
    transform=data_crs,
    scale=50,
    color="white",
    width=0.002,
)

# ------------------------------------------------------------
# RIVERS
# ------------------------------------------------------------

for name, (rlon, rlat) in rivers.items():
    ax.plot(rlon, rlat, "ro", transform=data_crs)
    ax.text(rlon + 2, rlat, name, color="white", transform=data_crs)

# ------------------------------------------------------------
# COLORBAR
# ------------------------------------------------------------

plt.colorbar(pcm, ax=ax, shrink=0.7, label="Tracer")

ax.set_title("Improved Arctic Tracer Advection (Semi-Lagrangian + Physical Diffusion)")
plt.show()