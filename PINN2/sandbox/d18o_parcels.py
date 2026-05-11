"""
d18o_parcels.py
===============
Spatially varying δ¹⁸O freshwater fraction field via OceanParcels
particle tracking.

Replaces the hand-rolled RK4 loop in d18o_streamline.py with OceanParcels,
which provides:
  - Built-in AdvectionRK4 on a spherical mesh (handles cos-lat correctly)
  - C-compiled JIT kernels (10–100× faster than pure Python)
  - Proper out-of-bounds / beaching recovery
  - Zarr/netCDF trajectory output for free

Pipeline
--------
1.  Build a parcels FieldSet from the xarray velocity dataset
2.  Seed a ParticleSet at river mouth locations
3.  Execute: AdvectionRK4 + MixingDecay + CullParticle kernels
4.  Load trajectory output, reconstruct 2-D δ¹⁸O field via KD-tree

Requirements
------------
    pip install parcels scipy xarray numpy

Parcels ≥ 2.4 is assumed (zarr output default in v3; netCDF in v2).
Set OUTPUT_FMT = "netcdf" if on parcels v2.

Usage
-----
    import xarray as xr
    from d18o_parcels import compute_d18o_field

    ds = xr.open_dataset("model_velocities.nc")

    rivers = [
        {"name": "Ob",        "lon":  73.6,  "lat": 68.5,  "d18o": -18.5},
        {"name": "Yenisei",   "lon":  81.7,  "lat": 71.8,  "d18o": -19.2},
        {"name": "Lena",      "lon": 126.4,  "lat": 72.4,  "d18o": -20.1},
        {"name": "Mackenzie", "lon": -135.0, "lat": 69.4,  "d18o": -21.3},
    ]

    d18o = compute_d18o_field(
        ds, rivers,
        u_var="uo", v_var="vo",
        lon_var="longitude", lat_var="latitude",
        depth_var="depth",            # set None if no depth dim
        n_per_source=200,
        n_steps=720,
        dt_seconds=6 * 3600,
        decay_timescale=30 * 86400,
        bandwidth_deg=0.75,
    )

    d18o.to_netcdf("d18o_field.nc")
"""

from __future__ import annotations

import math
import os
import tempfile
from datetime import timedelta

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

# ── parcels imports ───────────────────────────────────────────────────────────
try:
    from parcels import (
        FieldSet,
        ParticleSet,
        JITParticle,
        Variable,
        AdvectionRK4,
    )
except ImportError as e:
    raise ImportError(
        "OceanParcels is required: pip install parcels"
    ) from e


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Custom particle class
# ─────────────────────────────────────────────────────────────────────────────

class D18OParticle(JITParticle):
    """
    JIT-compiled particle carrying δ¹⁸O and age.

    Both variables are stored as 32-bit floats by parcels.
    ``d18o`` is initialised per-particle at seeding time.
    ``age`` accumulates advection time in seconds.
    """
    d18o = Variable("d18o", initial=0.0)
    age  = Variable("age",  initial=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Kernels
#     Rules for JIT-compiled kernels:
#       - Plain Python scalar maths only (no numpy inside kernel body)
#       - Use `math.*` for transcendentals
#       - Access fieldset constants via fieldset.<name>
#       - particle.dt holds the current timestep in seconds
# ─────────────────────────────────────────────────────────────────────────────

def MixingDecay(particle, fieldset, time):
    """
    Exponential mixing decay of the δ¹⁸O anomaly toward the ocean background.

    anomaly(t + dt) = anomaly(t) × exp(−dt / τ)

    Parameterises dilution by lateral mixing with background seawater.
    τ (decay_timescale) is stored as a fieldset constant.
    """
    anomaly = particle.d18o - fieldset.background_d18o
    anomaly *= math.exp(-particle.dt / fieldset.decay_timescale)
    particle.d18o = fieldset.background_d18o + anomaly
    particle.age += particle.dt


def CullParticle(particle, fieldset, time):
    """
    Delete particles whose δ¹⁸O anomaly has fallen below the threshold.
    This mirrors the early-stopping culling in the original code and
    avoids accumulating negligible-weight track-points in the output.
    """
    if math.fabs(particle.d18o - fieldset.background_d18o) < fieldset.threshold_anomaly:
        particle.delete()


def DeleteParticle(particle, fieldset, time):
    """
    Recovery kernel: silently delete particles that go out of bounds or
    beach on land (NaN velocity).  Without this, parcels raises an error
    and halts the run.
    """
    particle.delete()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FieldSet construction
# ─────────────────────────────────────────────────────────────────────────────

def build_fieldset(
    ds: xr.Dataset,
    u_var: str,
    v_var: str,
    lon_var: str,
    lat_var: str,
    depth_var: str | None,
    decay_timescale: float,
    background_d18o: float,
    threshold_anomaly: float,
) -> FieldSet:
    """
    Build a parcels FieldSet from an xarray Dataset.

    Handles:
      - Optional depth dimension (surface layer selected automatically)
      - Optional time dimension (steady-state assumed; extrapolation enabled)
      - Monotonicity check (parcels also requires ascending coords)

    Parameters
    ----------
    ds : xr.Dataset
        Must contain u_var, v_var and coordinate arrays lon_var, lat_var.
    depth_var : str or None
        Name of the depth coordinate.  Pass ``None`` if already 2-D.
    decay_timescale, background_d18o, threshold_anomaly
        Physics constants stored on the FieldSet for kernel access.

    Returns
    -------
    fieldset : parcels.FieldSet
        Ready for ParticleSet execution.
    """
    u_da = ds[u_var]
    v_da = ds[v_var]

    # ── drop non-spatial dimensions (depth, time) ────────────────────────────
    extra_dims = [d for d in u_da.dims if d not in (lat_var, lon_var)]
    if extra_dims:
        isel_kw = {d: 0 for d in extra_dims}
        u_da = u_da.isel(isel_kw)
        v_da = v_da.isel(isel_kw)
        print(f"  [fieldset] Dropped dims {extra_dims} (selected index 0).")

    # ── guarantee (lat, lon) axis ordering ───────────────────────────────────
    u_da = u_da.transpose(lat_var, lon_var)
    v_da = v_da.transpose(lat_var, lon_var)

    lons = ds[lon_var].values.astype(np.float32)
    lats = ds[lat_var].values.astype(np.float32)

    # parcels also requires ascending coordinates
    if lats[0] > lats[-1]:
        print("  [fieldset] Reversing descending latitude axis.")
        lats = lats[::-1]
        u_da = u_da.isel({lat_var: slice(None, None, -1)})
        v_da = v_da.isel({lat_var: slice(None, None, -1)})

    if lons[0] > lons[-1]:
        print("  [fieldset] Reversing descending longitude axis.")
        lons = lons[::-1]
        u_da = u_da.isel({lon_var: slice(None, None, -1)})
        v_da = v_da.isel({lon_var: slice(None, None, -1)})

    u = np.where(np.isfinite(u_da.values), u_da.values, 0.0).astype(np.float32)
    v = np.where(np.isfinite(v_da.values), v_da.values, 0.0).astype(np.float32)

    # ── sanity check ─────────────────────────────────────────────────────────
    u_rms = float(np.sqrt(np.mean(u ** 2)))
    v_rms = float(np.sqrt(np.mean(v ** 2)))
    print(f"  [fieldset] RMS u={u_rms:.4f} m/s  RMS v={v_rms:.4f} m/s")
    if u_rms < 1e-6 and v_rms < 1e-6:
        raise ValueError(
            "Velocity field is effectively zero everywhere. "
            "Check variable names, units, and masking."
        )

    # ── build FieldSet from raw arrays ───────────────────────────────────────
    # Using from_data (not from_xarray_dataset) gives explicit control over
    # dimension order and avoids version-dependent xarray parsing quirks.
    fieldset = FieldSet.from_data(
        data={"U": u, "V": v},
        dimensions={"lon": lons, "lat": lats},
        mesh="spherical",           # handles cos-lat in AdvectionRK4 for us
        allow_time_extrapolation=True,
    )

    # ── attach physics constants (accessible inside kernels) ─────────────────
    fieldset.add_constant("decay_timescale",   float(decay_timescale))
    fieldset.add_constant("background_d18o",   float(background_d18o))
    fieldset.add_constant("threshold_anomaly", float(threshold_anomaly))

    return fieldset


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Particle seeding
# ─────────────────────────────────────────────────────────────────────────────

def make_seed_arrays(
    river_sources: list[dict],
    n_per_source: int,
    spread_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (lons, lats, d18o_values) seed arrays for ParticleSet initialisation.

    Each river gets ``n_per_source`` particles jittered uniformly within
    ``±spread_deg`` of its mouth location.
    """
    lons_out, lats_out, d18o_out = [], [], []
    for src in river_sources:
        lons_out.append(src["lon"] + rng.uniform(-spread_deg, spread_deg, n_per_source))
        lats_out.append(src["lat"] + rng.uniform(-spread_deg, spread_deg, n_per_source))
        d18o_out.append(np.full(n_per_source, float(src["d18o"])))
    return (
        np.concatenate(lons_out),
        np.concatenate(lats_out),
        np.concatenate(d18o_out),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Field reconstruction  (unchanged from original)
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
    using Gaussian kernel-weighted averaging (KD-tree).

    Parameters
    ----------
    tracks : ndarray (M, 3)
        Columns: [lon, lat, d18o] — all recorded particle positions.
    grid_lon, grid_lat : 2-D ndarrays
        Target grid meshgrid arrays.
    bandwidth_deg : float
        Gaussian kernel σ in degrees (~1–2 × grid resolution).
    k_neighbours : int
        Track-points considered per grid cell.
    background_d18o : float
        Fill value for cells far from any track.

    Returns
    -------
    field : 2-D ndarray
    """
    tree = cKDTree(tracks[:, :2])

    grid_pts = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    k = min(k_neighbours, len(tracks))
    dists, idxs = tree.query(grid_pts, k=k, workers=-1)

    weights   = np.exp(-0.5 * (dists / bandwidth_deg) ** 2)
    d18o_nn   = tracks[idxs, 2]
    total_w   = weights.sum(axis=1)

    min_w = np.exp(-0.5 * 9.0)          # weight at 3σ — below this → background
    d18o_flat = np.where(
        total_w > min_w,
        (weights * d18o_nn).sum(axis=1) / total_w,
        background_d18o,
    )
    return d18o_flat.reshape(grid_lon.shape)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Load tracks from parcels output
# ─────────────────────────────────────────────────────────────────────────────

def load_tracks_from_output(output_path: str) -> np.ndarray:
    """
    Read parcels trajectory output (zarr or netcdf).
    """

    if output_path.endswith(".zarr") or os.path.isdir(output_path):
        ds_out = xr.open_zarr(output_path)

    elif output_path.endswith(".nc"):
        ds_out = xr.open_dataset(output_path, engine="netcdf4")

    else:
        raise ValueError(
            f"Unknown trajectory format: {output_path}\n"
            "Use a .zarr or .nc extension explicitly."
        )

    lons = ds_out["lon"].values.ravel().astype(float)
    lats = ds_out["lat"].values.ravel().astype(float)
    d18o = ds_out["d18o"].values.ravel().astype(float)

    mask = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(d18o)

    print(
        f"  [tracks] {mask.sum():,} valid observations loaded "
        f"from {output_path}"
    )

    return np.column_stack([lons[mask], lats[mask], d18o[mask]])

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_d18o_field(
    ds: xr.Dataset,
    river_sources: list[dict],
    # Variable / coordinate names
    u_var: str = "uo",
    v_var: str = "vo",
    lon_var: str = "longitude",
    lat_var: str = "latitude",
    depth_var: str | None = "depth",
    # Seeding
    n_per_source: int = 200,
    spread_deg: float = 0.1,
    seed: int | None = 42,
    # Integration
    n_steps: int = 720,
    dt_seconds: float = 6 * 3600,
    output_dt_seconds: float | None = None,   # defaults to dt_seconds (every step)
    output_path: str | None = None,            # where to save parcels output
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
    Full pipeline: FieldSet → ParticleSet → execute → reconstruct → DataArray.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing surface velocity fields.
    river_sources : list of dict
        Each entry: ``{"lon": float, "lat": float, "d18o": float, "name": str}``.
    n_per_source : int
        Particles per river mouth.
    n_steps : int
        Number of integration steps. Total time = n_steps × dt_seconds.
    dt_seconds : float
        Integration timestep (s). 6 h is typically stable for ocean models.
    output_dt_seconds : float, optional
        How often to record particle positions.  Defaults to every step.
        Increase (e.g. 6 × dt_seconds) to reduce output file size.
    output_path : str, optional
        Path for the parcels trajectory output file.
        A temp file is used and deleted if not specified.
    decay_timescale : float
        Mixing e-folding timescale (s). Default 30 days.
    background_d18o : float
        Background ocean δ¹⁸O (‰). Particles decay toward this value.
    bandwidth_deg : float
        Gaussian kernel σ for field reconstruction (degrees).
    k_neighbours : int
        Nearest track-points per grid cell in reconstruction.

    Returns
    -------
    xr.DataArray
        δ¹⁸O field on the model grid.
    """
    rng = np.random.default_rng(seed)

    if output_dt_seconds is None:
        output_dt_seconds = dt_seconds

    # ── use a temp file if no output path given ───────────────────────────────
    _tmp_dir   = None
    _owns_path = output_path is None
    if _owns_path:
        _tmp_dir    = tempfile.mkdtemp(prefix="d18o_parcels_")
        output_path = os.path.join(_tmp_dir, "tracks.zarr")

    total_days = n_steps * dt_seconds / 86400

    # ── 1. FieldSet ───────────────────────────────────────────────────────────
    if verbose:
        print("[d18O] Building parcels FieldSet")
    fieldset = build_fieldset(
        ds, u_var, v_var, lon_var, lat_var, depth_var,
        decay_timescale, background_d18o, threshold_anomaly,
    )

    # ── 2. Seeding ────────────────────────────────────────────────────────────
    if verbose:
        print(f"[d18O] Seeding {len(river_sources)} river(s) "
              f"× {n_per_source} particles = "
              f"{len(river_sources) * n_per_source} total")
    lons_seed, lats_seed, d18o_seed = make_seed_arrays(
        river_sources, n_per_source, spread_deg, rng
    )

    # Validate mouths are inside domain
    lon_min, lon_max = fieldset.U.grid.lon.min(), fieldset.U.grid.lon.max()
    lat_min, lat_max = fieldset.U.grid.lat.min(), fieldset.U.grid.lat.max()
    for src in river_sources:
        in_lon = lon_min <= src["lon"] <= lon_max
        in_lat = lat_min <= src["lat"] <= lat_max
        status = "✓" if (in_lon and in_lat) else "✗ OUTSIDE DOMAIN"
        if verbose:
            print(f"  {src.get('name','?'):12s}  "
                  f"lon={src['lon']:8.2f}  lat={src['lat']:7.2f}  {status}")

    pset = ParticleSet(
        fieldset=fieldset,
        pclass=D18OParticle,
        lon=lons_seed,
        lat=lats_seed,
        d18o=d18o_seed,
    )

    # ── 3. Output file ────────────────────────────────────────────────────────
    output_file = pset.ParticleFile(
        name=output_path,
        outputdt=timedelta(seconds=output_dt_seconds),
    )

    # ── 4. Execute ────────────────────────────────────────────────────────────
    if verbose:
        print(f"[d18O] Running parcels: {n_steps} steps × "
              f"{dt_seconds/3600:.0f} h = {total_days:.0f} days")

    pset.execute(
        [AdvectionRK4, MixingDecay, CullParticle],
        runtime=timedelta(seconds=n_steps * dt_seconds),
        dt=timedelta(seconds=dt_seconds),
        output_file=output_file,
        verbose_progress=verbose,
    )

    # ── 5. Load tracks ────────────────────────────────────────────────────────
    if verbose:
        print(f"[d18O] Loading trajectory output from {output_path}")
    tracks = load_tracks_from_output(output_path)

    # ── 6. Reconstruct field ──────────────────────────────────────────────────
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

    # ── 7. Clean up temp output ───────────────────────────────────────────────
    if _owns_path and _tmp_dir is not None:
        import shutil
        shutil.rmtree(_tmp_dir, ignore_errors=True)

    return xr.DataArray(
        field,
        coords={lat_var: ds[lat_var], lon_var: ds[lon_var]},
        dims=[lat_var, lon_var],
        name="d18o",
        attrs={
            "long_name"             : "δ¹⁸O freshwater tracer (parcels RK4)",
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
# 8.  Diagnostics  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tracks_and_field(
    tracks: np.ndarray,
    d18o: xr.DataArray,
    river_sources: list[dict],
    lon_var: str = "longitude",
    lat_var: str = "latitude",
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

    lons = d18o[lon_var].values
    lats = d18o[lat_var].values

    tr = tracks[::10]
    vmin = np.nanmin(d18o.values)
    vmax = np.nanmax(d18o.values)
    
    mesh = ax.pcolormesh(
        lons,
        lats,
        d18o.values,
        transform=ccrs.PlateCarree(),
        cmap="Blues_r",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    
    ax.scatter(
        tr[:, 0],
        tr[:, 1],
        c=tr[:, 2],
        transform=ccrs.PlateCarree(),
        s=0.5,
        alpha=0.3,
        cmap="Blues_r",
        vmin=vmin,
        vmax=vmax,
    )

    for src in river_sources:
        ax.plot(src["lon"], src["lat"], "r^", ms=8,
                transform=ccrs.PlateCarree(), zorder=5)
        ax.text(src["lon"] + 0.5, src["lat"], src.get("name", ""),
                transform=ccrs.PlateCarree(), fontsize=7, color="red")

    ax.set_title("δ¹⁸O field — OceanParcels RK4 tracking")
    plt.tight_layout()
    plt.savefig("d18o_parcels_field.png", dpi=150)
    plt.show()