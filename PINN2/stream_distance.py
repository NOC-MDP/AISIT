import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from matplotlib.collections import LineCollection

def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. All args can be numpy arrays."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Data ─────────────────────────────────────────────────────────────────────
ds = xr.open_dataset(
    "/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_vyo-vxo_monthly_climatology.nc"
)
# year = "1994"

# ── Figure ───────────────────────────────────────────────────────────────────
proj_map  = ccrs.NorthPolarStereo()
proj_data = ccrs.PlateCarree()

fig = plt.figure(figsize=(9, 9))
ax  = plt.axes(projection=proj_map)
ax.set_extent([-180, 180, 70, 90], crs=proj_data)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.coastlines(resolution="50m")
ax.gridlines(draw_labels=False, linewidth=0.5, linestyle="--")

# u = ds["vxo"].sel(time=year).isel(depth=0).mean("time").compute()
# v = ds["vyo"].sel(time=year).isel(depth=0).mean("time").compute()

u = ds["vxo"].isel(depth=0).mean("month").compute()
v = ds["vyo"].isel(depth=0).mean("month").compute()

# Points of interest as (lat, lon) tuples
poi = np.array([
    # [74.883,  72.04 ],
    [76.382,85.826],
   [74.82,120.49 ],
   [70.2,164.2  ],
   [70.0,-139.4  ],
   [77.531,-72.667],
   [82.603,-14.373],
   [70.299,-20.985],
   [68.254, 41.939],
   [71.520,-154.435],
   [76.653,-124.645],
   [81.619,-94.978],
   [74.589,130.897],
    [70.870,62.539],
    [70.482,-120.556],
    [73.477,148.805],
    [80.297,103.017],
    [80.665,16.114],
    [81.119,52.134],
    
])  # shape (N, 2): columns are [lat, lon]


threshold_km=350

lats, lons = u['latitude'].values, u['longitude'].values
lon2d, lat2d = np.meshgrid(lons, lats)
X, Y = u.values, v.values
X_src_crs = X / np.cos(lat2d / 180 * np.pi)
Y_src_crs = Y
magnitude = np.sqrt(X**2 + Y**2)
magn_src_crs = np.sqrt(X_src_crs**2 + Y_src_crs**2)
bu = X_src_crs * magnitude / magn_src_crs
bv = Y_src_crs * magnitude / magn_src_crs
lw = 25 * np.sqrt(bu**2 + bv**2)
lw = np.nan_to_num(lw, nan=0.0)

base_kwargs = {
    "density": 16,
    "linewidth": lw,
    "arrowsize": 1,
    "maxlength": 10.0,
    "integration_direction": "forward"
}

p = ax.streamplot(
    lon2d, lat2d, bu, bv,
    transform=ccrs.PlateCarree(),
    color=(0.55, 0.55, 0.55, 0.25),
    **base_kwargs,
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
poi_colors = plt.cm.tab20(np.linspace(0, 1, len(poi))) 

field_sum   = np.zeros_like(lat2d, dtype=float)
field_count = np.zeros_like(lat2d, dtype=float)

# ── 4. Highlight streamlines, coloured by the closest POI ────────────────────
for chain in streamlines:
    all_ll  = np.vstack([lonlat_list[i] for i in chain])
    lons_v, lats_v = all_ll[:, 0], all_ll[:, 1]

    # Find the closest POI and its distance
    best_dist, best_poi_idx = np.inf, -1
    for k, (plat, plon) in enumerate(poi):
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
        plat, plon = poi[best_poi_idx]
        dist_start = _haversine_km(lats_v[0],  lons_v[0],  plat, plon)
        dist_end   = _haversine_km(lats_v[-1], lons_v[-1], plat, plon)

        if dist_end < dist_start:
            continue                        # POI is downstream — discard

        # Fade opaque → transparent, matching the arrow/flow direction
        alphas = np.linspace(1.0, 0.0, n_segs)

        source_value = -21.0  # δ18O value at the river mouth
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

field = np.full_like(field_sum, np.nan)

mask = field_count > 0
field[mask] = field_sum[mask] / field_count[mask]

# ── Scatter POIs with matching colours ───────────────────────────────────────
ax.scatter(
    poi[:, 1], poi[:, 0],
    s=70, c=poi_colors, zorder=5,
    edgecolors="black", linewidths=1.0,
    label="River mouth",
    transform=ccrs.PlateCarree(),
)
ax.legend(loc="lower right", fontsize=9)
plt.savefig("streamlines_distance.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8,8))
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
plt.savefig("mass_fraction_field.png")
plt.show()

