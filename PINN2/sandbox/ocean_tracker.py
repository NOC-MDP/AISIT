"""
Arctic River Mouth Particle Tracking
=====================================
Continuously seeds particles at major Arctic river mouths using
OceanTracker with a CMEMS Topaz4 Arctic Ocean reanalysis dataset
on a regular (structured) grid.

Simulation duration : 5 years
Release             : continuous, one particle per river per time step
Output              : NetCDF tracks + trajectory plot

Dependencies:
    pip install oceantracker matplotlib cartopy numpy xarray
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
from datetime import datetime

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

# ── OceanTracker ──────────────────────────────────────────────────────────────
from oceantracker.main import run


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────────────────
#   Point INPUT_DIR at the folder containing your downloaded CMEMS Topaz4
#   NetCDF files.  Files may be daily, monthly, or multi-year – OceanTracker
#   will glob them automatically.
INPUT_DIR  = "/work/scratch-pw5/thopri/cmems/"
OUTPUT_DIR = "/work/scratch-pw5/thopri/arctic_tracking_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Simulation window ─────────────────────────────────────────────────────────
#   Adjust start / end dates to match your downloaded dataset.
START_DATE = "2015-01-01"   # ISO 8601
END_DATE   = "2019-12-31"   # 5 years later

# ── Release interval ─────────────────────────────────────────────────────────
#   How often (seconds) a new particle is released at each river mouth.
#   3600 = one particle per hour per river.
RELEASE_INTERVAL_S = 3600

# ── Particle lifetime ─────────────────────────────────────────────────────────
#   Maximum age (seconds) before a particle is deactivated.  Set to None to
#   keep particles alive for the full simulation.
MAX_PARTICLE_AGE_S = None   # e.g. 365 * 24 * 3600 for 1-year lifetime

# ── CMEMS / Topaz4 variable names ─────────────────────────────────────────────
#   Standard CMEMS naming – adjust if your files differ.
FIELD_VARIABLE_MAP = {
    "u": "vxo",          # eastward sea water velocity
    "v": "vyo",          # northward sea water velocity
    "water_depth": "depth",   # optional static bathymetry
}

# ── Depth of release (m, positive down) ──────────────────────────────────────
RELEASE_DEPTH_M = 0.5   # near-surface (Topaz top layer ~0.5 m)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ARCTIC RIVER MOUTH LOCATIONS
#     (lon, lat, name)  –  extend / trim as needed
# ══════════════════════════════════════════════════════════════════════════════

RIVER_MOUTHS = [
    # ── Eurasian rivers ────────────────────────────────────────────────────
    ( 54.2,  68.9, "Pechora"),
    ( 40.5,  64.6, "Northern Dvina"),
    ( 69.0,  66.6, "Ob"),
    ( 82.7,  71.8, "Yenisei"),
    (126.7,  72.4, "Lena"),
    (141.0,  71.5, "Indigirka"),
    (161.3,  69.5, "Kolyma"),
    # ── North American rivers ──────────────────────────────────────────────
    (-134.1, 69.4, "Mackenzie"),
    (-164.8, 63.0, "Yukon"),
    ( -63.6, 58.5, "Churchill"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD OCEANTRACKER PARAMETER DICTIONARY
# ══════════════════════════════════════════════════════════════════════════════

def build_release_groups():
    """One continuously-releasing point source per river mouth."""
    groups = []
    for lon, lat, name in RIVER_MOUTHS:
        rg = {
            "class_name": "oceantracker.release_groups.point_release.PointRelease",
            "points":     [[lon, lat, RELEASE_DEPTH_M]],
            "release_interval": RELEASE_INTERVAL_S,
            "name": name.replace(" ", "_"),
            # Pulse start/end span the full simulation (default behaviour)
        }
        if MAX_PARTICLE_AGE_S is not None:
            rg["max_age"] = MAX_PARTICLE_AGE_S
        groups.append(rg)
    return groups


params = {
    # ── Run metadata ─────────────────────────────────────────────────────────
    "output_file_base": "arctic_rivers",
    "root_output_dir":  OUTPUT_DIR,

    # ── Time control ─────────────────────────────────────────────────────────
    "time_step": 3600,          # advection time step (s) — 1 hour

    # ── Reader — CMEMS Topaz4 on a regular (structured) grid ─────────────────
    #   OceanTracker's generic structured-grid NetCDF reader handles regular
    #   lat/lon grids such as those produced by TOPAZ4 / CMEMS Arctic MFC.
    "reader": {
        "class_name": (
            "oceantracker.reader"
            ".dev_generic_reader"
            ".GenericUnstructuredReader"  
        ),
        "input_dir":  INPUT_DIR,
        "file_mask":  "*.nc",           # glob pattern for your CMEMS files
        
        # Simulation window – lives inside the reader block
        "start_datetime": START_DATE,
        "end_datetime":   END_DATE,

        # Coordinate / variable names inside the NetCDF files
        "dimension_map": {
            "time": "time",
            "x":    "longitude",
            "y":    "latitude",
            "z":    "depth",
        },
        "field_variable_map": FIELD_VARIABLE_MAP,

        # CMEMS files use "hours since …" or "seconds since …" – OceanTracker
        # reads the CF-compliant units attribute automatically.
        "isodate_of_hindcast_time_zero": "1950-01-01",  # Topaz4 epoch

        # Retain only near-surface layer(s) to save memory (index 0 = surface)
        "z_level_range": [0, 3],   # top 4 depth levels
    },

    # ── Trajectory output ─────────────────────────────────────────────────────
    "tracks_writer": {
        "class_name": (
            "oceantracker.tracks_writer"
            ".track_writer_compact.TrackWriterCompact"
        ),
        # Write particle positions every 6 hours to limit output size
        "output_step_count": 6,
    },

    # ── Release groups ────────────────────────────────────────────────────────
    "release_groups": build_release_groups(),

    # ── Optional: random walk diffusion ──────────────────────────────────────
    # Uncomment to add horizontal diffusivity (m²/s)
    # "velocity_modifiers": [
    #     {
    #         "class_name": (
    #             "oceantracker.velocity_modifiers"
    #             ".random_walk.RandomWalk"
    #         ),
    #         "diffusion_coefficient_H": 10.0,
    #     }
    # ],
}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  RUN THE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation():
    print("=" * 60)
    print("  Arctic River Mouth Particle Tracking")
    print(f"  Period : {START_DATE}  →  {END_DATE}")
    print(f"  Rivers : {len(RIVER_MOUTHS)}")
    print("=" * 60)
    run(params)
    print("\nSimulation complete.  Output written to:", OUTPUT_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PLOT TRAJECTORIES
# ══════════════════════════════════════════════════════════════════════════════

# Colour-blind-friendly palette – one colour per river
RIVER_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999",
    "#117733", "#88CCEE",
]


def load_tracks(track_file: str) -> tuple:
    """
    Read an OceanTracker compact track NetCDF file directly with xarray.

    OceanTracker's TrackWriterCompact stores data as 2-D arrays
    (particle × time-step) and uses NaN / fill-values for positions
    where a particle does not yet exist or has been deactivated.

    Returns
    -------
    x        : (n_particles, n_steps) float32 – longitude
    y        : (n_particles, n_steps) float32 – latitude
    group_id : (n_particles,)         int32   – release-group index
    """
    ds = xr.open_dataset(track_file, mask_and_scale=True)

    # OceanTracker compact writer uses these variable names.
    # If your version differs, inspect ds.data_vars to find the right names.
    x        = ds["x"].values.astype("float32")          # longitude
    y        = ds["y"].values.astype("float32")          # latitude
    group_id = ds["IDrelease_group"].values.astype(int)  # per-particle group

    ds.close()
    return x, y, group_id


def plot_trajectories(track_file: str | None = None):
    """
    Read OceanTracker compact track output and plot on a polar stereographic
    Arctic map.

    Parameters
    ----------
    track_file : path to a *_tracks.nc file produced by the simulation.
                 If None the function searches OUTPUT_DIR automatically.
    """
    import glob

    # ── Locate output file ────────────────────────────────────────────────────
    if track_file is None:
        files = glob.glob(
            os.path.join(OUTPUT_DIR, "**", "*tracks*.nc"), recursive=True
        )
        if not files:
            raise FileNotFoundError(
                f"No track files found under {OUTPUT_DIR}. "
                "Run the simulation first."
            )
        track_file = sorted(files)[0]
        print(f"Reading: {track_file}")

    # ── Load data – directly from NetCDF via xarray ───────────────────────────
    x, y, group_id = load_tracks(track_file)
    # x, y   shape: (n_particles, n_timesteps)
    # group_id shape: (n_particles,)

    # ── Figure setup ─────────────────────────────────────────────────────────
    proj = ccrs.NorthPolarStereo(central_longitude=90)
    fig, ax = plt.subplots(
        figsize=(12, 12),
        subplot_kw={"projection": proj},
    )
    ax.set_extent([-180, 180, 55, 90], crs=ccrs.PlateCarree())

    # ── Basemap features ──────────────────────────────────────────────────────
    ax.add_feature(cfeature.OCEAN,     facecolor="#cce5f0", zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor="#e8e0d0", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6,       zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=":", zorder=2)
    ax.gridlines(
        draw_labels=False, linewidth=0.4, color="grey",
        alpha=0.5, linestyle="--",
    )

    # ── Plot one trajectory bundle per river ──────────────────────────────────
    transform = ccrs.PlateCarree()

    for idx, (lon_r, lat_r, name) in enumerate(RIVER_MOUTHS):
        color  = RIVER_COLORS[idx % len(RIVER_COLORS)]
        mask   = group_id == idx          # particles from this river
        xi, yi = x[mask], y[mask]        # (n_p, n_t)

        # Sub-sample for plotting performance (at most 200 tracks per river)
        step = max(1, xi.shape[0] // 200)

        for i in range(0, xi.shape[0], step):
            lons = xi[i]
            lats = yi[i]
            # Skip NaN positions (particle not yet released or deactivated)
            valid = np.isfinite(lons) & np.isfinite(lats)
            if valid.sum() < 2:
                continue
            ax.plot(
                lons[valid], lats[valid],
                transform=transform,
                color=color, alpha=0.25, linewidth=0.5,
                zorder=3,
            )

        # Mark the release point
        ax.plot(
            lon_r, lat_r,
            transform=transform,
            marker="o", markersize=7,
            color=color, markeredgecolor="black", markeredgewidth=0.8,
            zorder=5, label=name,
        )

    # ── Legend & title ────────────────────────────────────────────────────────
    ax.legend(
        title="River mouth",
        loc="lower left",
        fontsize=8,
        title_fontsize=9,
        framealpha=0.85,
        edgecolor="grey",
    )
    ax.set_title(
        f"Arctic river plume trajectories\n"
        f"{START_DATE}  →  {END_DATE}   "
        f"(CMEMS Topaz4 · OceanTracker)",
        fontsize=13, pad=14,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_png = os.path.join(OUTPUT_DIR, "arctic_trajectories.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved: {out_png}")
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Step 1 – run the particle tracking simulation
    run_simulation()

    # Step 2 – plot the resulting trajectories
    plot_trajectories()