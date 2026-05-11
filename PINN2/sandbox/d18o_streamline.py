"""
d18o_streamline.py
==================
Spatially varying δ¹⁸O freshwater fraction field via streamline particle tracking.

Pipeline
--------
1. Seed particles at river mouth locations (with small spread)
2. Integrate forward along u/v velocity field using RK4
3. Apply exponential mixing decay along each trajectory
4. Reconstruct a continuous 2D field from all particle positions
   using Gaussian kernel weighting (via KD-tree)

The result naturally produces high δ¹⁸O values along current pathways
and low values in cross-stream or stagnant regions — far more physically
realistic than an isotropic distance decay.

Notes on the "decay"
--------------------
δ¹⁸O is a conservative tracer — it doesn't actually decay.
The exponential term here parameterises *dilution by mixing* with the
background ocean, which has a known (lower) δ¹⁸O value.  The decay
timescale should therefore be interpreted as a mixing timescale.
A 30-day e-folding is a reasonable first guess for Arctic shelves;
tune it against observed plume extents if you have them.

Grid support
------------
Regular (1-D lon/lat) grids: works directly.
Curvilinear grids (NEMO ORCA, MOM6, etc.): see `build_velocity_interpolators`
for notes on switching to index-space integration.

Usage
-----
    import xarray as xr
    from d18o_streamline import compute_d18o_field

    ds = xr.open_dataset("model_velocities.nc")

    rivers = [
        {"name": "Ob",        "lon": 73.6,  "lat": 68.5,  "d18o": -18.5},
        {"name": "Yenisei",   "lon": 81.7,  "lat": 71.8,  "d18o": -19.2},
        {"name": "Lena",      "lon": 126.4, "lat": 72.4,  "d18o": -20.1},
        {"name": "Mackenzie", "lon": -135.0,"lat": 69.4,  "d18o": -21.3},
    ]

    d18o = compute_d18o_field(
        ds, rivers,
        u_var="uo", v_var="vo",
        lon_var="longitude", lat_var="latitude",
        n_per_source=200,
        n_steps=720,
        dt_seconds=6 * 3600,       # 6-hour steps
        decay_timescale=30 * 86400, # 30-day mixing timescale
        bandwidth_deg=0.75,
    )

    d18o.to_netcdf("d18o_field.nc")
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Particle seeding
# ─────────────────────────────────────────────────────────────────────────────

def seed_particles(
    river_sources: list[dict],
    n_per_source: int = 100,
    spread_deg: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Seed particles at river mouth locations.

    Parameters
    ----------
    river_sources : list of dict
        Each entry must have keys ``lon``, ``lat``, ``d18o``.
        Optional key ``name`` is used for logging only.
    n_per_source : int
        Number of particles per river mouth.
    spread_deg : float
        Half-width of uniform random spatial spread around each mouth (degrees).
        Adds slight spatial jitter to avoid singularities in the field
        reconstruction step. Set to 0 for a point source.
    rng : np.random.Generator, optional
        Pass a seeded generator for reproducibility, e.g. ``np.random.default_rng(42)``.

    Returns
    -------
    particles : ndarray of shape (N, 4)
        Columns: [lon, lat, d18o, age_seconds]
    """
    if rng is None:
        rng = np.random.default_rng()

    rows = []
    for src in river_sources:
        lons  = src["lon"] + rng.uniform(-spread_deg, spread_deg, n_per_source)
        lats  = src["lat"] + rng.uniform(-spread_deg, spread_deg, n_per_source)
        d18o  = np.full(n_per_source, float(src["d18o"]))
        age   = np.zeros(n_per_source)
        rows.append(np.column_stack([lons, lats, d18o, age]))

    return np.vstack(rows)                  # shape: (n_rivers * n_per_source, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Velocity interpolation
# ─────────────────────────────────────────────────────────────────────────────

def build_velocity_interpolators(
    ds: xr.Dataset,
    u_var: str = "u",
    v_var: str = "v",
    lon_var: str = "lon",
    lat_var: str = "lat",
) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:

    lons = ds[lon_var].values.astype(float)
    lats = ds[lat_var].values.astype(float)

    # ── FIX 1: Use xarray to extract u/v with guaranteed (lat, lon) axis order ──
    # .transpose() is a no-op if dims are already in this order.
    u_da = ds[u_var]
    v_da = ds[v_var]

    # Handle optional depth/time dimensions by selecting surface
    extra_dims = [d for d in u_da.dims if d not in (lat_var, lon_var)]
    if extra_dims:
        sel = {d: u_da[d].values[0] for d in extra_dims}
        u_da = u_da.sel(sel)
        v_da = v_da.sel(sel)
        print(f"  [velocity] Dropping extra dims {extra_dims}, selecting index 0 each.")

    # Now safely transpose to (lat, lon)
    u_da = u_da.transpose(lat_var, lon_var)
    v_da = v_da.transpose(lat_var, lon_var)

    u = u_da.values.astype(float)
    v = v_da.values.astype(float)

    # ── FIX 2: Ensure axes are monotonically increasing ──────────────────────
    # RegularGridInterpolator requires this; descending lat is common in models.
    if lats[0] > lats[-1]:
        print("  [velocity] Flipping latitude axis to ascending order.")
        lats = lats[::-1]
        u   = u[::-1, :]
        v   = v[::-1, :]

    if lons[0] > lons[-1]:
        print("  [velocity] Flipping longitude axis to ascending order.")
        lons = lons[::-1]
        u   = u[:, ::-1]
        v   = v[:, ::-1]

    # ── FIX 3: Sanity-check that we actually have nonzero velocities ──────────
    u_rms = np.sqrt(np.nanmean(u**2))
    v_rms = np.sqrt(np.nanmean(v**2))
    print(f"  [velocity] RMS u={u_rms:.4f} m/s, RMS v={v_rms:.4f} m/s")
    if u_rms < 1e-6 and v_rms < 1e-6:
        raise ValueError(
            "Velocity field is effectively zero everywhere. "
            "Check u_var/v_var names, units, masking, and coordinate names."
        )

    u = np.where(np.isfinite(u), u, 0.0)
    v = np.where(np.isfinite(v), v, 0.0)

    kw = dict(bounds_error=False, fill_value=0.0)
    interp_u = RegularGridInterpolator((lats, lons), u, **kw)
    interp_v = RegularGridInterpolator((lats, lons), v, **kw)

    return interp_u, interp_v

def _velocity_at(
    particles: np.ndarray,
    interp_u: RegularGridInterpolator,
    interp_v: RegularGridInterpolator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, v) at each particle position. Input: (N,4) array."""
    pts = particles[:, [1, 0]]          # scipy expects (lat, lon)
    return interp_u(pts), interp_v(pts)
    
# ─────────────────────────────────────────────────────────────────────────────
# 3.  RK4 integration
# ─────────────────────────────────────────────────────────────────────────────

def _rk4_step(
    particles: np.ndarray,
    interp_u: RegularGridInterpolator,
    interp_v: RegularGridInterpolator,
    dt: float,
) -> np.ndarray:
    """
    Advance particle positions one RK4 step.

    Converts velocity (m/s) to (degrees/s) using:
        dlon/dt = u / (111_320 * cos(lat))
        dlat/dt = v / 111_320

    The d18o column and age column are **not** updated here;
    decay and aging are applied in ``_apply_decay`` after each step.

    Parameters
    ----------
    dt : float
        Timestep in seconds.
    """
    def derivs(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u, v = _velocity_at(p, interp_u, interp_v)
        cos_lat = np.cos(np.deg2rad(p[:, 1]))
        cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)   # avoid /0 at poles
        dlon = u / (111_320.0 * cos_lat)
        dlat = v / 111_320.0
        return dlon, dlat

    p = particles.copy()

    k1l, k1p = derivs(p)

    p2 = p.copy()
    p2[:, 0] += 0.5 * dt * k1l
    p2[:, 1] += 0.5 * dt * k1p
    k2l, k2p = derivs(p2)

    p3 = p.copy()
    p3[:, 0] += 0.5 * dt * k2l
    p3[:, 1] += 0.5 * dt * k2p
    k3l, k3p = derivs(p3)

    p4 = p.copy()
    p4[:, 0] += dt * k3l
    p4[:, 1] += dt * k3p
    k4l, k4p = derivs(p4)

    out = particles.copy()
    out[:, 0] += (dt / 6.0) * (k1l + 2*k2l + 2*k3l + k4l)
    out[:, 1] += (dt / 6.0) * (k1p + 2*k2p + 2*k3p + k4p)
    return out


def _apply_decay(
    particles: np.ndarray,
    dt: float,
    decay_timescale: float,
    background_d18o: float,
) -> np.ndarray:
    """
    Apply exponential mixing decay toward background ocean δ¹⁸O.

    The freshwater anomaly (d18o - background) decays exponentially:
        anomaly(t+dt) = anomaly(t) * exp(-dt / tau)

    This correctly approaches the background value at large travel times
    rather than decaying toward zero.
    """
    anomaly = particles[:, 2] - background_d18o
    anomaly *= np.exp(-dt / decay_timescale)
    out = particles.copy()
    out[:, 2] = background_d18o + anomaly
    out[:, 3] += dt                          # age
    return out


def integrate_streamlines(particles, interp_u, interp_v, n_steps=720,
                          dt_seconds=6*3600, decay_timescale=30*86400,
                          background_d18o=-1.0, threshold_anomaly=0.05,
                          verbose=True) -> np.ndarray:

    all_tracks = []
    active = particles.copy()

    for step in range(n_steps):
        all_tracks.append(active[:, :4].copy())

        active = _rk4_step(active, interp_u, interp_v, dt_seconds)
        active = _apply_decay(active, dt_seconds, decay_timescale, background_d18o)

        # ── FIX 4: Diagnose displacement at step 1 ────────────────────────────
        if step == 0 and verbose:
            disp = np.sqrt(
                (active[:, 0] - particles[:, 0])**2 +
                (active[:, 1] - particles[:, 1])**2
            )
            print(f"  [step 1] mean displacement = {disp.mean():.4f}°  "
                  f"max = {disp.max():.4f}°")
            if disp.max() < 1e-6:
                raise RuntimeError(
                    "Particles did not move at step 1. "
                    "Velocity interpolation is returning zeros. "
                    "Check coordinate names (lon_var, lat_var) match the dataset exactly."
                )

        alive = np.abs(active[:, 2] - background_d18o) > threshold_anomaly
        if not np.any(alive):
            if verbose:
                print(f"  All particles below threshold at step {step}. Stopping.")
            break
        active = active[alive]

        if verbose and (step + 1) % 100 == 0:
            pct = 100 * (1 - len(active) / len(particles))
            print(f"  Step {step+1}/{n_steps}  |  {len(active)} particles active  "
                  f"({pct:.0f}% culled)")

    return np.vstack(all_tracks)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Field reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_d18o_field(
    tracks: np.ndarray,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    bandwidth_deg: float = 0.5,
    k_neighbours: int = 50,
    background_d18o: float = -1.0,
) -> np.ndarray:
    """
    Reconstruct a continuous 2-D δ¹⁸O field from particle track positions
    using Gaussian kernel-weighted averaging (via KD-tree).

    For each grid cell the weighted mean of the k nearest track-points is
    computed.  Grid cells far from any track inherit the background value.

    Parameters
    ----------
    tracks : ndarray (M, 4)
        Output of ``integrate_streamlines``.
    grid_lon, grid_lat : 2-D ndarrays
        Target grid coordinates (meshgrid convention).
    bandwidth_deg : float
        Gaussian kernel standard deviation in degrees.
        Tune to ~ 1–2 × grid resolution.
    k_neighbours : int
        Number of nearest track-points considered per grid cell.
    background_d18o : float
        Value to fill in grid cells with negligible particle influence.

    Returns
    -------
    field : 2-D ndarray, same shape as ``grid_lon``
    """
    # Build KD-tree on track lon/lat
    tree = cKDTree(tracks[:, :2])

    # Query grid
    grid_pts = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    k = min(k_neighbours, len(tracks))
    dists, idxs = tree.query(grid_pts, k=k, workers=-1)   # parallel query

    # Gaussian weights; set weight to 0 beyond 3σ
    weights = np.exp(-0.5 * (dists / bandwidth_deg) ** 2)

    d18o_nn = tracks[idxs, 2]                              # (n_grid, k)
    total_w  = weights.sum(axis=1)

    # Weighted mean; fall back to background where weights are negligible
    min_w = np.exp(-0.5 * 9.0)                             # weight at 3σ
    d18o_flat = np.where(
        total_w > min_w,
        (weights * d18o_nn).sum(axis=1) / total_w,
        background_d18o,
    )

    return d18o_flat.reshape(grid_lon.shape)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_d18o_field(
    ds: xr.Dataset,
    river_sources: list[dict],
    u_var: str = "u",
    v_var: str = "v",
    lon_var: str = "lon",
    lat_var: str = "lat",
    # Particle seeding
    n_per_source: int = 200,
    spread_deg: float = 0.1,
    seed: int | None = 42,
    # Integration
    n_steps: int = 720,
    dt_seconds: float = 6 * 3600,
    # Physics
    decay_timescale: float = 30 * 86400,
    background_d18o: float = -1.0,
    threshold_anomaly: float = 0.05,
    # Field reconstruction
    bandwidth_deg: float = 0.5,
    k_neighbours: int = 50,
    # Misc
    verbose: bool = True,
) -> xr.DataArray:
    """
    Full pipeline: seed → integrate → reconstruct → return DataArray.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with velocity fields and 1-D lon/lat coordinates.
    river_sources : list of dict
        Each dict: ``{"lon": float, "lat": float, "d18o": float, "name": str}``.
        ``name`` is optional; ``d18o`` is the mouth value in per mil.
    n_per_source : int
        Particles per river (more → smoother field, slower).
    n_steps : int
        Max integration steps. Total simulated time = n_steps × dt_seconds.
        720 steps × 6 h = 180 days.
    dt_seconds : float
        Integration timestep in seconds. 6 h is usually stable.
    decay_timescale : float
        Mixing e-folding timescale (seconds). Default: 30 days.
    background_d18o : float
        Ocean background δ¹⁸O (per mil). Typically −1 to −2 ‰ in the Arctic.
    bandwidth_deg : float
        Gaussian kernel half-width for field reconstruction (degrees).
        Start at ~ grid resolution; increase if the field looks noisy.
    k_neighbours : int
        Number of track-points used per grid cell during reconstruction.

    Returns
    -------
    xr.DataArray
        δ¹⁸O field on the model grid, with attributes recording key parameters.
    """
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"[d18O] Seeding {len(river_sources)} river(s) × {n_per_source} particles")
    particles = seed_particles(river_sources, n_per_source, spread_deg, rng)

    if verbose:
        print("[d18O] Building velocity interpolators")
    interp_u, interp_v = build_velocity_interpolators(ds, u_var, v_var, lon_var, lat_var)

    total_days = n_steps * dt_seconds / 86400
    if verbose:
        print(f"[d18O] Integrating up to {n_steps} steps "
              f"({dt_seconds/3600:.0f} h each = {total_days:.0f} day max)")
    tracks = integrate_streamlines(
        particles, interp_u, interp_v,
        n_steps=n_steps,
        dt_seconds=dt_seconds,
        decay_timescale=decay_timescale,
        background_d18o=background_d18o,
        threshold_anomaly=threshold_anomaly,
        verbose=verbose,
    )

    if verbose:
        print(f"[d18O] Reconstructing field from {len(tracks):,} track points")
    lons = ds[lon_var].values.astype(float)
    lats = ds[lat_var].values.astype(float)
    grid_lon, grid_lat = np.meshgrid(lons, lats)

    field = reconstruct_d18o_field(
        tracks, grid_lon, grid_lat,
        bandwidth_deg=bandwidth_deg,
        k_neighbours=k_neighbours,
        background_d18o=background_d18o,
    )

    return xr.DataArray(
        field,
        coords={lat_var: ds[lat_var], lon_var: ds[lon_var]},
        dims=[lat_var, lon_var],
        name="d18o",
        attrs={
            "long_name"             : "δ¹⁸O freshwater tracer (streamline tracking)",
            "units"                 : "per mil",
            "background_d18o"      : background_d18o,
            "decay_timescale_days" : decay_timescale / 86400,
            "n_particles_total"    : len(river_sources) * n_per_source,
            "n_track_points"       : len(tracks),
            "integration_days_max" : total_days,
            "bandwidth_deg"        : bandwidth_deg,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Optional diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def plot_tracks_and_field(
    tracks: np.ndarray,
    d18o: xr.DataArray,
    river_sources: list[dict],
    lon_var: str = "lon",
    lat_var: str = "lat",
) -> None:
    """
    Quick-look plot of particle tracks overlaid on the reconstructed field.
    Requires matplotlib and cartopy.
    """
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
    except ImportError:
        print("matplotlib / cartopy not available — skipping plot.")
        return

    proj = ccrs.NorthPolarStereo()
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": proj})
    ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=0.6)

    # Reconstructed field
    lons = d18o[lon_var].values
    lats = d18o[lat_var].values
    mesh = ax.pcolormesh(
        lons, lats, d18o.values,
        transform=ccrs.PlateCarree(),
        cmap="Blues_r", shading="auto",
    )
    plt.colorbar(mesh, ax=ax, label="δ¹⁸O (‰)", shrink=0.7)

    # Decimate tracks for plotting (every 10th point)
    tr = tracks[::10]
    ax.scatter(
        tr[:, 0], tr[:, 1], c=tr[:, 2],
        transform=ccrs.PlateCarree(),
        s=0.5, alpha=0.3, cmap="Blues_r",
    )

    # River mouths
    for src in river_sources:
        ax.plot(src["lon"], src["lat"], "r^", ms=8,
                transform=ccrs.PlateCarree(), zorder=5)
        ax.text(src["lon"] + 0.5, src["lat"], src.get("name", ""),
                transform=ccrs.PlateCarree(), fontsize=7, color="red")

    ax.set_title("δ¹⁸O field — streamline particle tracking")
    plt.tight_layout()
    plt.savefig("d18o_fw_f_field.png")
    plt.show()