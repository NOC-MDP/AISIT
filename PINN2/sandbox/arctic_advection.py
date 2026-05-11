#!/usr/bin/env python3
"""
Arctic River Particle Advection and Mass Fraction Mapping
=========================================================

This script:

1. Loads annual-mean CMEMS Arctic velocity fields
2. Continuously seeds particles at Arctic river mouths
3. Advects particles using OceanParcels
4. Accumulates particle occupancy through time
5. Converts occupancy to a normalized mass-fraction field
6. Projects onto Arctic polar stereographic coordinates
7. Plots using Cartopy

Author: OpenAI ChatGPT
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from datetime import timedelta

from parcels import (
    FieldSet,
    ParticleSet,
    JITParticle,
    AdvectionRK4,
    Variable,
)

import cartopy.crs as ccrs

from pyproj import Transformer


# =============================================================================
# User Settings
# =============================================================================

CMEMS_FILE = "/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_vyo-vxo_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-01-01.nc"

OUTPUT_ZARR = "arctic_particles.zarr"

# Runtime
YEARS = 5
DT_HOURS = 6

# Particle release
RELEASE_INTERVAL_DAYS = 5
PARTICLES_PER_RELEASE = 100

# Diffusion coefficient (m2/s)
KH = 50.0

# Polar stereographic grid resolution (meters)
GRID_RESOLUTION = 25000.0  # 25 km

# Arctic domain in EPSG:3413
XMIN = -3.5e6
XMAX =  3.5e6
YMIN = -3.5e6
YMAX =  3.5e6


# =============================================================================
# River Mouth Definitions
# =============================================================================

river_sources = {
    "Mackenzie":  (70.529, -138.046, 1.0),
}

# Format:
# name: (lat, lon, relative discharge weight)


# =============================================================================
# Load Dataset
# =============================================================================

print("Loading CMEMS dataset...")

ds = xr.open_dataset(CMEMS_FILE)
# =============================================================================
# Create OceanParcels FieldSet
# =============================================================================

print("Building fieldset...")

print("Preparing surface velocities...")

ds_surface = ds.isel(depth=0).transpose("time", "latitude", "longitude")

fieldset = FieldSet.from_xarray_dataset(
    ds_surface,
    variables={"U": "vxo", "V": "vyo"},
    dimensions={"lon": "longitude", "lat": "latitude", "time": "time"},
    allow_time_extrapolation=True,
)


# =============================================================================
# Particle Class
# =============================================================================

class RiverParticle(JITParticle):
    age = Variable("age", dtype=np.float32, initial=0.0)


# =============================================================================
# Particle Kernels
# =============================================================================

def AgeParticle(particle, fieldset, time):
    particle.age += particle.dt

def DeleteParticle(particle, fieldset, time):
    particle.delete()



# =============================================================================
# Create Initial ParticleSet
# =============================================================================

print("Creating initial particles...")

release_lons = []
release_lats = []

for river, (lat, lon, weight) in river_sources.items():

    n_particles = int(PARTICLES_PER_RELEASE * weight)

    release_lons.extend([lon] * n_particles)
    release_lats.extend([lat] * n_particles)

pset = ParticleSet(
    fieldset=fieldset,
    pclass=RiverParticle,
    lon=release_lons,
    lat=release_lats,
)

# =============================================================================
# Output File
# =============================================================================

output_file = pset.ParticleFile(
    name=OUTPUT_ZARR,
    outputdt=timedelta(days=RELEASE_INTERVAL_DAYS),
)

# =============================================================================
# Continuous Release Advection Loop
# =============================================================================

print("Running advection simulation...")

n_steps = int((YEARS * 365) / RELEASE_INTERVAL_DAYS)

kernel = AdvectionRK4 + pset.Kernel(AgeParticle)

for step in range(n_steps):

    print(f"Step {step+1}/{n_steps}")

    # -------------------------------------------------------------------------
    # Add new particles
    # -------------------------------------------------------------------------

    new_pset = ParticleSet(
        fieldset=fieldset,
        pclass=RiverParticle,
        lon=release_lons,
        lat=release_lats,
    )

    pset.add(new_pset)

    # -------------------------------------------------------------------------
    # Advect
    # -------------------------------------------------------------------------

    pset.execute(
        kernel,
        runtime=timedelta(days=RELEASE_INTERVAL_DAYS),
        dt=timedelta(hours=DT_HOURS),
        output_file=output_file,
    )

print("Simulation complete.")


# =============================================================================
# Load Trajectories
# =============================================================================

print("Loading trajectories...")

traj = xr.open_zarr(OUTPUT_ZARR)

lon = traj["lon"].values.flatten()
lat = traj["lat"].values.flatten()

mask = np.isfinite(lon) & np.isfinite(lat)

lon = lon[mask]
lat = lat[mask]

print(f"Total particle positions: {len(lon):,}")


# =============================================================================
# Convert to Polar Stereographic Coordinates
# =============================================================================

print("Projecting coordinates...")

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:3413",
    always_xy=True,
)

x, y = transformer.transform(lon, lat)
print(x.min(), x.max())
print(y.min(), y.max())

# =============================================================================
# Create Polar Grid
# =============================================================================

print("Building occupancy grid...")

x_bins = np.arange(XMIN, XMAX + GRID_RESOLUTION, GRID_RESOLUTION)
y_bins = np.arange(YMIN, YMAX + GRID_RESOLUTION, GRID_RESOLUTION)

occupancy, _, _ = np.histogram2d(
    y,
    x,
    bins=[y_bins, x_bins],
)
occupancy = gaussian_filter(occupancy, sigma=1.5)

# =============================================================================
# Convert Occupancy to Mass Fraction
# =============================================================================

print("Computing mass fraction field...")

mass_fraction = occupancy / occupancy.max()
mass_fraction = np.ma.masked_where(
    occupancy == 0,
    mass_fraction
)

print(
    f"Mass fraction range: "
    f"{mass_fraction.min():.4f} - "
    f"{mass_fraction.max():.4f}"
)


# =============================================================================
# Plot
# =============================================================================

print("Plotting...")

fig = plt.figure(figsize=(12, 12))

ax = plt.axes(projection=ccrs.NorthPolarStereo())

ax.set_extent([-180, 180, 55, 90], ccrs.PlateCarree())

# Use full bin edges
xx, yy = np.meshgrid(x_bins, y_bins)

# Plot occupancy field
mesh = ax.pcolormesh(
    xx,
    yy,
    mass_fraction,
    transform=ccrs.epsg(3413),
    shading="flat",
    cmap="viridis"
)

# Coastlines
ax.coastlines(resolution="50m", linewidth=0.8)

# Gridlines
ax.gridlines(
    draw_labels=False,
    linewidth=0.5,
    linestyle="--",
)

# River locations
for river, (lat0, lon0, weight) in river_sources.items():

    ax.plot(
        lon0,
        lat0,
        marker="o",
        markersize=6,
        transform=ccrs.PlateCarree(),
    )

    ax.text(
        lon0,
        lat0,
        river,
        fontsize=8,
        transform=ccrs.PlateCarree(),
    )

# Colorbar
cb = plt.colorbar(
    mesh,
    orientation="vertical",
    shrink=0.7,
    pad=0.05,
)

cb.set_label("Normalized Mass Fraction")

plt.title(
    "Arctic River Particle Mass Fraction Field\n"
    f"{YEARS}-Year Continuous Advection Simulation"
)

plt.tight_layout()

plt.savefig(
    "arctic_mass_fraction.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
print("Done.")