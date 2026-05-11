import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import RegularGridInterpolator
from matplotlib.collections import LineCollection

# -- Functions -----------------------
def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. All args can be numpy arrays."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def jitter_seeds(river_mouths, n_jitter=5, spread=0.3, seed=42):
    rng = np.random.default_rng(seed)
    seeds = []
    for lon, lat in river_mouths:
        angles = rng.uniform(0, 2 * np.pi, n_jitter)
        radii  = rng.uniform(0, spread, n_jitter)
        candidates = np.column_stack([lon + radii * np.cos(angles),
                                      lat + radii * np.sin(angles)])
        seeds.append(candidates)
    return np.vstack(seeds) if seeds else np.empty((0, 2))
# ── Data ─────────────────────────────────────────────────────────────────────
ds = xr.open_dataset(
    "/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_vyo-vxo_monthly_climatology.nc"
)

# ── Figure ───────────────────────────────────────────────────────────────────
proj_map  = ccrs.NorthPolarStereo()
proj_data = ccrs.PlateCarree()

fig = plt.figure(figsize=(9, 9))
ax  = plt.axes(projection=proj_map)
ax.set_extent([-180, 180, 70, 90], crs=proj_data)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.coastlines(resolution="50m")
ax.gridlines(draw_labels=False, linewidth=0.5, linestyle="--")

u = ds["vxo"].isel(depth=0).mean("month").compute()
v = ds["vyo"].isel(depth=0).mean("month").compute()

river_mouths = np.array([
    [ -139.464,  71.656],
    [-130.218,72.035],
    [123.354,75.996],
    [110.933,78.967],
    [74.860,75.581],
    [-92.543,83.268],
    [40.871,70.896],
    [166.136,71.515],
    [-73.983,75.577],
    # [15.748,71.302],
    [-53.013,84.431],
    [155.525,73.056],
    # [-168.638,67.858],
    [-13.680,74.762],
    [8.199,77.549],
    
    
])  # [lon, lat]

threshold_km = 500 # distance threshold used to assign streamlines to river mouths
source_value = 1 # d18O of river water at the river mouth
lats, lons = u['latitude'].values, u['longitude'].values
lon2d, lat2d = np.meshgrid(lons, lats)
X, Y = u.values, v.values
X_src_crs = X / np.cos(lat2d / 180 * np.pi)
Y_src_crs = Y
magnitude = np.sqrt(X**2 + Y**2)
magn_src_crs = np.sqrt(X_src_crs**2 + Y_src_crs**2)
bu = X_src_crs * magnitude / magn_src_crs
bv = Y_src_crs * magnitude / magn_src_crs

# Build interpolator on the ocean mask (NaN = land)
u_interp = RegularGridInterpolator(
    (lats, lons),          # note: (lat, lon) order for RegularGridInterpolator
    u.values,
    method='nearest',
    bounds_error=False,
    fill_value=np.nan,
)

jittered = jitter_seeds(river_mouths, n_jitter=64, spread=2.5)

# Query interpolator — jittered is (lon, lat) so flip to (lat, lon) for lookup
ocean_vals = u_interp(jittered[:, ::-1])
ocean_mask = ~np.isnan(ocean_vals)
jittered = jittered[ocean_mask]

# Transform surviving points to projected coords
transformed = proj_map.transform_points(proj_data, jittered[:, 0], jittered[:, 1])
start_points_proj = transformed[:, :2]

# Filter to projected grid bounds
grid_transformed = proj_map.transform_points(proj_data, lon2d.ravel(), lat2d.ravel())
x_proj, y_proj = grid_transformed[:, 0], grid_transformed[:, 1]
x_min, x_max = np.nanmin(x_proj), np.nanmax(x_proj)
y_min, y_max = np.nanmin(y_proj), np.nanmax(y_proj)

proj_mask = (
    (start_points_proj[:, 0] >= x_min) & (start_points_proj[:, 0] <= x_max) &
    (start_points_proj[:, 1] >= y_min) & (start_points_proj[:, 1] <= y_max)
)
start_points_proj = start_points_proj[proj_mask]

p = ax.streamplot(
    lon2d, lat2d, bu, bv,
    transform=ccrs.PlateCarree(),
    start_points=start_points_proj,
    integration_direction='forward',
    maxlength=2.0,
    density=64,
    color='steelblue',
    linewidth=1,
)
paths       = p.lines.get_paths()
verts_list  = [path.vertices.copy() for path in paths]
n           = len(verts_list)

# ── 1. Convert all vertices to lon/lat (for distance check) ──────────────────
lonlat_list = []
for verts in verts_list:
    ll = proj_data.transform_points(proj_map, verts[:, 0], verts[:, 1])
    lonlat_list.append(ll[:, :2])          # shape (K, 2): lon, lat

# ── 2. Build successor/predecessor maps using endpoint proximity ──────────────
#    Tolerance in projected metres; connected segments share endpoints exactly
#    (or within float noise), so 500 m is plenty for Arctic-scale data.
tol = 500.0

start_lookup = {}                          # rounded-start-point -> segment idx
for i, verts in enumerate(verts_list):
    key = tuple(np.round(verts[0] / tol).astype(int))
    start_lookup[key] = i

successor   = [-1] * n
predecessor = [-1] * n
for i, verts in enumerate(verts_list):
    end_key = tuple(np.round(verts[-1] / tol).astype(int))
    j = start_lookup.get(end_key, -1)
    if j != -1 and j != i:
        successor[i]   = j
        predecessor[j] = i

# Mark the original mouth locations (not the jittered seeds)
ax.scatter(
    river_mouths[:, 0], river_mouths[:, 1],
    color='red', zorder=5, transform=ccrs.PlateCarree()
)
ax.legend()
plt.savefig("seeded_streamlines.png", dpi=150, bbox_inches="tight")
plt.show()
        
# ── 3. Walk chains into complete streamlines ─────────────────────────────────
visited     = [False] * n
streamlines = []                           # list of ordered segment-index lists

for i in range(n):
    if visited[i]:
        continue
    # Walk back to the true start of this streamline
    start = i
    while predecessor[start] != -1 and not visited[predecessor[start]]:
        start = predecessor[start]
    # Walk forward collecting every segment
    chain, cur = [], start
    while cur != -1 and not visited[cur]:
        chain.append(cur)
        visited[cur] = True
        cur = successor[cur]
    streamlines.append(chain)

# ── Assign a colour to each POI ───────────────────────────────────────────────
poi_colors = plt.cm.tab20(np.linspace(0, 1, len(river_mouths))) 

field_sum   = np.zeros_like(lat2d, dtype=float)
field_count = np.zeros_like(lat2d, dtype=float)

# ── 4. Highlight streamlines, coloured by the closest POI ────────────────────
for chain in streamlines:
    all_ll  = np.vstack([lonlat_list[i] for i in chain])
    lons_v, lats_v = all_ll[:, 0], all_ll[:, 1]

    # Find the closest POI and its distance
    best_dist, best_poi_idx = np.inf, -1
    for k, (plon, plat) in enumerate(river_mouths):
        d = _haversine_km(lats_v, lons_v, plat, plon).min()
        if d < best_dist:
            best_dist, best_poi_idx = d, k

    if best_dist <= threshold_km:
        all_proj = np.vstack([verts_list[i] for i in chain])

        # ── Split into consecutive point-pairs (one per segment) ─────────
        points   = all_proj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Segment midpoints in projected coordinates
        mid_proj = 0.5 * (segments[:, 0, :] + segments[:, 1, :])
        
        # Convert midpoint coordinates back to lon/lat
        mid_ll = proj_data.transform_points(
            proj_map,
            mid_proj[:, 0],
            mid_proj[:, 1]
        )
        
        mid_lon = mid_ll[:, 0]
        mid_lat = mid_ll[:, 1]



        # ── Alpha fades 1 → 0 along the streamline ───────────────────────
        n_segs = len(segments)

        # ── Alpha: full opacity at the end nearest the POI ───────────────
        plon, plat = river_mouths[best_poi_idx]
        dist_start = _haversine_km(lats_v[0],  lons_v[0],  plat, plon)
        dist_end   = _haversine_km(lats_v[-1], lons_v[-1], plat, plon)

        if dist_end < dist_start:
            continue                        # POI is downstream — discard

        # Fade opaque → transparent, matching the arrow/flow direction
        alphas = np.linspace(1.0, 0.0, n_segs)


                # cumulative streamline distance
        seg_dx = _haversine_km(
            lats_v[:-1],
            lons_v[:-1],
            lats_v[1:],
            lons_v[1:]
        )
        
        cumdist = np.concatenate([[0], np.cumsum(seg_dx)])
        
        # distance per segment midpoint
        seg_dist = 0.5 * (cumdist[:-1] + cumdist[1:])
        
        L = 2000.0  # km decay scale
        
        seg_values = source_value * np.exp(-seg_dist / L)

        for mlon, mlat, val in zip(mid_lon, mid_lat, seg_values):

            # Find nearest grid indices
            iy = np.argmin(np.abs(lats - mlat))
            ix = np.argmin(np.abs(lons - mlon))
        
            if np.isfinite(val):
                field_sum[iy, ix] += val
                field_count[iy, ix] += 1

        # ── Build per-segment RGBA colours ───────────────────────────────
        rgb    = poi_colors[best_poi_idx][:3]
        colors = np.column_stack([
            np.tile(rgb, (n_segs, 1)),
            alphas
        ])

        lc = LineCollection(segments, colors=colors, linewidth=3, zorder=10)
        ax.add_collection(lc)

field = np.full_like(field_sum, 0)

mask = field_count > 0
field[mask] = field_sum[mask] / field_count[mask]

# ── Scatter POIs with matching colours ───────────────────────────────────────
ax.scatter(
    river_mouths[:, 0], river_mouths[:, 1],
    s=70, c=poi_colors, zorder=5,
    edgecolors="black", linewidths=1.0,
    label="River mouth",
    transform=ccrs.PlateCarree(),
)
ax.legend(loc="lower right", fontsize=9)
plt.savefig("streamlines_distance.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(9,9))
ax2 = plt.axes(projection=proj_map)

ax2.set_extent([-180, 180, 70, 90], crs=proj_data)

pcm = ax2.pcolormesh(
    lon2d,
    lat2d,
    field,
    transform=proj_data,
    cmap="viridis"
)

ax2.coastlines()
ax2.add_feature(cfeature.LAND, facecolor="lightgray")

plt.colorbar(pcm, label="δ18O")
plt.savefig("mass_fraction_field.png", dpi=150, bbox_inches="tight")
plt.show()


ds = xr.Dataset(
    data_vars={
        "field": (("latitude", "longitude"), field)
    },
    coords={
        "longitude": (("longitude"), lons),
        "latitude": (("latitude"), lats),
    },
)

ds["longitude"].attrs = {
    "standard_name": "longitude",
    "units": "degrees_east",
}

ds["latitude"].attrs = {
    "standard_name": "latitude",
    "units": "degrees_north",
}

ds["field"].attrs = {
    "coordinates": "lon lat"
}
ds.to_netcdf("massfraction_field.nc")
print(ds)