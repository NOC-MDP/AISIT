"""
run_d18o.py
===========
Main script for computing a spatially varying δ¹⁸O freshwater fraction
field using streamline particle tracking.

Usage
-----
    python run_d18o.py                        # use defaults below
    python run_d18o.py --config my_config.py  # override with a config file
    python run_d18o.py --plot                 # show diagnostic plot after run
    python run_d18o.py --dry-run              # validate inputs without integrating

Edit the CONFIG section below to match your dataset and river observations.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from d18o_parcels import compute_d18o_field, plot_tracks_and_field


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG  —  edit this section to match your data
# ═════════════════════════════════════════════════════════════════════════════

CONFIG = {

    # ── Input data ────────────────────────────────────────────────────────────
    "velocity_file": "/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_vyo-vxo_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-01-01.nc",       # path to your NetCDF velocity file
    "u_var":         "vxo",                  # zonal velocity variable name
    "v_var":         "vyo",                  # meridional velocity variable name
    "lon_var":       "longitude",           # 1-D longitude coordinate name
    "lat_var":       "latitude",            # 1-D latitude coordinate name

    # If your file has a time dimension and you want a specific index:
    "time_index":    0,                     # None → assume no time dim

    # ── River sources  ────────────────────────────────────────────────────────
    # d18o values in per mil (‰).  Use mouth locations, not headwaters.
    "rivers": [
        {"name": "Ob",        "lon":  72.04,  "lat": 74.883,  "d18o": -18.5},
        {"name": "Yenisei",   "lon":  85.826,  "lat": 76.382,  "d18o": -19.2},
        {"name": "Lena",      "lon": 120.49,  "lat": 74.82,  "d18o": -20.1},
        {"name": "Kolyma",    "lon": 164.2,  "lat": 70.2,  "d18o": -19.8},
        {"name": "Mackenzie", "lon": -139.4, "lat": 70,  "d18o": -21.3},
    ],

    # ── Ocean background ──────────────────────────────────────────────────────
    # Arctic surface seawater δ¹⁸O.  Particles asymptote toward this value.
    # Typical range: −1.0 to −2.0 ‰.
    "background_d18o": -1.5,

    # ── Particle seeding ──────────────────────────────────────────────────────
    "n_per_source":  200,     # particles per river (more → smoother, slower)
    "spread_deg":    0.15,    # jitter half-width around mouth (degrees)
    "seed":          42,      # RNG seed for reproducibility

    # ── Integration ───────────────────────────────────────────────────────────
    "n_steps":       15000,     # max steps  (720 × 6 h = 180 days)
    "dt_seconds":    6*3600,  # timestep in seconds (6 h recommended)

    # ── Physics ───────────────────────────────────────────────────────────────
    "decay_timescale":   365 * 86400,  # mixing e-folding timescale (30 days)
    "threshold_anomaly": 0.05,        # cull particles below this anomaly (‰)

    # ── Field reconstruction ──────────────────────────────────────────────────
    "bandwidth_deg":  0.75,   # Gaussian kernel width — tune to ~1-2× grid res
    "k_neighbours":   50,     # track-points per grid cell in reconstruction

    # ── Output ────────────────────────────────────────────────────────────────
    "output_file":   "d18o_field.zarr",
    "output_var":    "d18o",

}

# ═════════════════════════════════════════════════════════════════════════════


def load_velocities(cfg: dict) -> xr.Dataset:
    """Load and optionally slice the velocity dataset."""
    path = Path(cfg["velocity_file"])
    if not path.exists():
        sys.exit(f"[ERROR] Velocity file not found: {path.resolve()}")

    print(f"[load] Opening {path}")
    ds = xr.open_dataset(path)

    # Validate required variables
    for var in (cfg["u_var"], cfg["v_var"], cfg["lon_var"], cfg["lat_var"]):
        if var not in ds and var not in ds.coords:
            sys.exit(
                f"[ERROR] Variable '{var}' not found in dataset.\n"
                f"        Available: {list(ds.data_vars) + list(ds.coords)}"
            )

    # Squeeze out a time dimension if present
    t_idx = cfg.get("time_index")
    if t_idx is not None:
        time_dims = [d for d in ds[cfg["u_var"]].dims if "time" in d.lower()]
        if time_dims:
            dim = time_dims[0]
            ds = ds.isel({dim: t_idx})
            print(f"[load] Selected time index {t_idx} along dim '{dim}'")

    # Squeeze any remaining size-1 dims (e.g. depth already at surface)
    ds = ds.squeeze(drop=True)

    lon = ds[cfg["lon_var"]]
    lat = ds[cfg["lat_var"]]
    print(f"[load] Grid: {len(lat)} lat × {len(lon)} lon  "
          f"({float(lat.min()):.1f}–{float(lat.max()):.1f} °N, "
          f"{float(lon.min()):.1f}–{float(lon.max()):.1f} °E)")

    return ds


def validate_rivers(rivers: list[dict], ds: xr.Dataset,
                    lon_var: str, lat_var: str) -> None:
    """Warn if any river mouth falls outside the model domain."""
    lon_min = float(ds[lon_var].min())
    lon_max = float(ds[lon_var].max())
    lat_min = float(ds[lat_var].min())
    lat_max = float(ds[lat_var].max())

    for r in rivers:
        ok = (lon_min <= r["lon"] <= lon_max) and (lat_min <= r["lat"] <= lat_max)
        status = "OK" if ok else "OUTSIDE DOMAIN ⚠"
        print(f"  {r['name']:12s}  lon={r['lon']:7.1f}  lat={r['lat']:5.1f}  "
              f"d18o={r['d18o']:+.1f} ‰   [{status}]")


def save_output(d18o: xr.DataArray, cfg: dict) -> Path:
    """Write the δ¹⁸O field to NetCDF."""
    out_path = Path(cfg["output_file"])
    ds_out = d18o.to_dataset(name=cfg["output_var"])

    # Attach river metadata as a global attribute
    river_summary = "; ".join(
        f"{r['name']} ({r['d18o']:+.1f}‰)" for r in cfg["rivers"]
    )
    ds_out.attrs.update({
        "title":   "δ¹⁸O freshwater fraction — streamline particle tracking",
        "rivers":  river_summary,
        "history": f"Generated by run_d18o.py",
    })

    ds_out.to_netcdf(out_path)
    print(f"[save] Written to {out_path.resolve()}")
    return out_path


def print_summary(d18o: xr.DataArray, cfg: dict, elapsed: float) -> None:
    """Print a brief summary of the output field."""
    vals = d18o.values
    finite = vals[np.isfinite(vals)]
    print()
    print("─" * 50)
    print("  δ¹⁸O field summary")
    print("─" * 50)
    print(f"  Shape            : {d18o.shape}")
    print(f"  Min              : {finite.min():.3f} ‰")
    print(f"  Max              : {finite.max():.3f} ‰")
    print(f"  Mean             : {finite.mean():.3f} ‰")
    print(f"  Background       : {cfg['background_d18o']:.3f} ‰")
    frac_signal = np.mean(finite > cfg["background_d18o"] + cfg["threshold_anomaly"])
    print(f"  Cells with signal: {100*frac_signal:.1f}%")
    print(f"  Elapsed          : {elapsed:.1f} s")
    print("─" * 50)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute spatially varying δ¹⁸O field via streamline tracking."
    )
    p.add_argument("--plot",    action="store_true",
                   help="Show diagnostic plot after run (requires cartopy).")
    p.add_argument("--dry-run", action="store_true",
                   help="Load data and validate inputs only; skip integration.")
    p.add_argument("--config",  metavar="FILE",
                   help="Path to a Python file that overrides CONFIG dict values.")
    return p.parse_args()


def load_config_override(path: str, cfg: dict) -> dict:
    """Exec a user config file and merge any CONFIG dict it defines."""
    override_path = Path(path)
    if not override_path.exists():
        sys.exit(f"[ERROR] Config file not found: {override_path}")
    ns: dict = {}
    exec(override_path.read_text(), ns)
    if "CONFIG" not in ns:
        sys.exit("[ERROR] Config file must define a CONFIG dict.")
    cfg.update(ns["CONFIG"])
    print(f"[config] Loaded overrides from {override_path}")
    return cfg


def main() -> None:
    args = parse_args()
    cfg  = CONFIG.copy()

    if args.config:
        cfg = load_config_override(args.config, cfg)

    print("=" * 60)
    print("  δ¹⁸O streamline tracking")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    ds = load_velocities(cfg)
    # ── Validate rivers ───────────────────────────────────────────────────────
    print(f"\n[rivers] {len(cfg['rivers'])} source(s):")
    validate_rivers(cfg["rivers"], ds, cfg["lon_var"], cfg["lat_var"])

    if args.dry_run:
        print("\n[dry-run] Validation complete — exiting without integration.")
        return

    # ── Run pipeline ──────────────────────────────────────────────────────────
    print()
    t0 = time.perf_counter()

    d18o = compute_d18o_field(
        ds,
        cfg["rivers"],
        u_var            = cfg["u_var"],
        v_var            = cfg["v_var"],
        lon_var          = cfg["lon_var"],
        lat_var          = cfg["lat_var"],
        n_per_source     = cfg["n_per_source"],
        spread_deg       = cfg["spread_deg"],
        seed             = cfg["seed"],
        n_steps          = cfg["n_steps"],
        dt_seconds       = cfg["dt_seconds"],
        decay_timescale  = cfg["decay_timescale"],
        background_d18o  = cfg["background_d18o"],
        threshold_anomaly= cfg["threshold_anomaly"],
        bandwidth_deg    = cfg["bandwidth_deg"],
        k_neighbours     = cfg["k_neighbours"],
        output_path = cfg["output_file"],
        verbose          = True,
    )

    elapsed = time.perf_counter() - t0

    # ── Save ──────────────────────────────────────────────────────────────────
    print()
    # save_output(d18o, cfg)
    print_summary(d18o, cfg, elapsed)

    # ── Optional plot ─────────────────────────────────────────────────────────
    if args.plot:
        print("\n[plot] Opening diagnostic plot…")
        # Re-run integration just for track data (reuses cached result in practice)
        plot_tracks_and_field(
            np.empty((0, 4)),   # pass empty tracks; field is what matters for quick-look
            d18o,
            cfg["rivers"],
            lon_var=cfg["lon_var"],
            lat_var=cfg["lat_var"],
        )


if __name__ == "__main__":
    main()