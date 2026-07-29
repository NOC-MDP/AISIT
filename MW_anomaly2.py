
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
import cmocean
import os
import glob

cfg = {
    "work_dir":"/gws/ssde/j25a/nemo/vol4/thopri/PINN",
    "create_inventory": False,
    "mw_output_path" : "arctic_meteoric_inventory_ECCO_1990_2025.nc",
    "sim_output_path" : "arctic_seaicemelt_inventory_ECCO_1990_2025.nc",
    "frac_output_path" : "arctic_fractions_ECCO_1990_2025.nc",
    "ML_model_dir": "infer_output_ECCO/",
    "baseline_start": "1992",
    "baseline_end": "2002",
    "anomaly_start": "2017",
    "anomaly_end": "2017",
    "inventory_start": 1992,
    "inventory_end": 2017,
    "inference_target": "/work/scratch-pw5/thopri/ECCO",
    "salinity": "SALT",
    "depth": "Z",
    "tracer_pred": "tracer_pred",
    "time": "time",
    "tracer_uncertainty": "tracer_uncertainty",
}
# Masks
Baltic = {"lat_min":54.0,"lat_max":66.0,"lon_min":10.0,"lon_max":30.0}
Pacific = {"lat_max":65.0,"lon_min":-120.0,"lon_max":140.0}

# Mass Balance
MassBal = {
    "S_mar" : 34.8,
    "delta_mar" : 0.3,
    "S_met" : 0.0,
    "delta_met" : -20.0,
    "S_sim" : 4.0,
    "delta_sim" : -2.0,
    "S_ref": 34.8,
}

# Volume Boxes # NOT implemented yet
Beaufort_Gyre = {"lats": [70, 70, 80, 80, 70], "lons":  [-160, -130, -130, -160, -160] }
Arctic = {}

# Flux Transects
transects = {#Nares_Strait : {},
"N_Baffin_Bay" : {'lat1':73.447,"lon1":-77.783,"lat2":76.448,"lon2":-68.575,"num_points": 50},
# "Davis_Strait" : {'lat1':73.447,"lon1":-77.783,"lat2":76.448,"lon2":-68.575,"num_points": 50},
"Fram_Strait" : {'lat1':81.432,"lon1":-12.256,"lat2":79.551,"lon2":11.428,"num_points": 50},
"Bering_Strait" : {'lat1':66.1,"lon1":-169.777,"lat2":66.1,"lon2":-166.666,"num_points": 50},
}

start_time = time.time()

def log_status(message):
    elapsed = time.time() - start_time
    print(f"[{elapsed:6.1f}s] {message}")

if cfg['create_inventory']:
    # --- 0. INITIALIZE DASK CLUSTER ---
    log_status("Initializing Dask cluster and client...")
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit="16GB")
    client = Client(cluster)
    print(f"Dask Dashboard link for monitoring: {client.dashboard_link}")

    # --- 1. SET UP PATHS ---
    log_status("Generating file paths...")
    start_year = cfg['inventory_start']
    end_year = cfg['inventory_end']
    model_years = [year for year in range(start_year, end_year + 1)]
    paths = []
    ecco_paths = []
    for model_year in model_years:
        for i in range(12):
            paths.append(
                f"{cfg['work_dir']}/{cfg['ML_model_dir']}/tracer_predicted_{model_year}_{i + 1:02d}.nc"
            )


    # --- 2. LOAD DATASETS WITH CHUNKS ---
    chunks = {"time": 12, "Z": 10}
    ecco_paths = ecco_paths + glob.glob(os.path.join(cfg['inference_target'], "*.nc"))
    log_status("Loading ECCO v4r4 dataset lazily...")
    # Use glob to grab all NetCDF files in the folder

    # if not file_list:
    #     sys.exit(f"[ERROR] No NetCDF files found in: {nc_path}")

    # Open and combine all datasets
    # data_vars='minimal' merges variables that don't share all dimensions without bloating memory
    # coords='minimal' does the same for coordinates
    ds_s_raw = xr.open_mfdataset(ecco_paths, combine='by_coords', data_vars='minimal', coords='minimal')
    # ds_s_raw = xr.open_dataset(
    #     f"/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_so_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-03-01.nc",
    #     chunks=chunks,
    # )

    log_status(
        f"Loading {len(paths)} PINN predicted tracer files via open_mfdataset..."
    )
    ds_o = xr.open_mfdataset(
        paths, combine="nested", concat_dim="time", chunks=chunks
    )

    log_status("Aligning timelines between physics and tracer arrays...")
    ds_s = ds_s_raw.sel(time=ds_o[cfg['time']], method="nearest")

    log_status("Masking out the Baltic Sea and Pacific Ocean regions...")

    # 1. Define the bounding box for the Baltic Sea
    baltic_lat_min, baltic_lat_max = 54.0, 66.0
    baltic_lon_min, baltic_lon_max = 10.0, 30.0

    # Create Baltic Mask
    is_baltic = (
        (ds_s["latitude"] >= Baltic["lat_min"])
        & (ds_s["latitude"] <= Baltic['lat_max'])
        & (ds_s["longitude"] >= Baltic['lon_min'])
        & (ds_s["longitude"] <= Baltic['lon_max'])
    )

    # 2. Define the boundary for the North Pacific / Bering Sea
    # Bering Strait is at roughly 66°N. We mask everything south of 66°N
    # in the Pacific longitude sector (roughly 100°E to 100°W / -100° to 180° & -180° to -100°)
    pacific_lat_max = Pacific['lat_max']

    # In a -180 to 180 longitude setup, the North Pacific covers:
    # From 140°E (which is 140) to -120°W (which is -120)
    is_pacific = (ds_s["latitude"] < pacific_lat_max) & (
        (ds_s["longitude"] >= Pacific['lon_max']) | (ds_s["longitude"] <= Pacific['lon_min'])
    )

    # 3. Combine masks: We want data where it is NOT Baltic and NOT Pacific
    valid_ocean_mask = ~(is_baltic | is_pacific)

    # Apply the combined mask to your input variables
    ds_s[cfg['salinity']] = ds_s[cfg['salinity']].where(valid_ocean_mask)
    ds_o[cfg["tracer_pred"]] = ds_o[cfg["tracer_pred"]].where(valid_ocean_mask)


    ds_o[cfg['tracer_uncertainty']] = ds_o[cfg['tracer_uncertainty']].where(valid_ocean_mask)

    # --- 3. MASS BALANCE & ERROR PROPAGATION (Lazy) ---
    log_status("Building mass balance and error propagation graphs...")

    denom = (MassBal['S_mar'] - MassBal['S_sim']) * (MassBal['delta_mar'] - MassBal['delta_met']) \
        - (MassBal['S_mar'] - MassBal['S_met']) * (MassBal['delta_mar'] - MassBal['delta_sim'])

    ds = xr.Dataset()

    # Base fractions
    ds["f_met"] = (
        (ds_s[cfg['salinity']] - MassBal['S_mar']) * (MassBal['delta_mar'] - MassBal['delta_sim'])
        - (ds_o["tracer_pred"] - MassBal['delta_mar']) * (MassBal['S_mar'] - MassBal['S_sim'])
    ) / denom
    ds["f_sim"] = (
        (ds_o["tracer_pred"] - MassBal['delta_mar']) * (MassBal['S_mar'] - MassBal['S_met'])
        - (ds_s[cfg['salinity']] - MassBal['S_mar']) * (MassBal['delta_mar'] - MassBal['delta_met'])
    ) / denom

    # Propagated Fraction Uncertainties (Analytical Derivatives * Tracer Uncertainty)
    ds["f_met_unc"] = np.abs(-(MassBal['S_mar'] - MassBal['S_sim']) / denom) * ds_o[cfg['tracer_uncertainty']]
    ds["f_sim_unc"] = np.abs((MassBal['S_mar'] - MassBal['S_met']) / denom) * ds_o[cfg['tracer_uncertainty']]

    # --- 4. PREPARE VERTICAL INTEGRATION ---
    log_status("Calculating depth layer thicknesses (dz)...")

    depths = ds_s[cfg['depth']].values
    bounds = np.zeros(len(depths) + 1)

    # 1. Calculate the midpoints between layers (this works regardless of orientation)
    bounds[1:-1] = (depths[:-1] + depths[1:]) / 2

    # 2. Determine orientation and set the boundary conditions
    if depths[0] > depths[-1]:
        # Case A: Sorted Surface-to-Bottom (e.g., 0, -10, -50, -100)
        bounds[0] = 0  # Surface is at the start
        # Extrapolate the bottom boundary downwards
        bounds[-1] = depths[-1] + (depths[-1] - bounds[-2])
    else:
        # Case B: Sorted Bottom-to-Surface (e.g., -100, -50, -10, 0)
        bounds[-1] = 0  # Surface is at the end
        # Extrapolate the bottom boundary downwards
        bounds[0] = depths[0] - (bounds[1] - depths[0])

    # 3. Calculate thickness and force it to be positive using np.abs
    dz_values = np.abs(np.diff(bounds))

    # 4. Wrap back into xarray DataArray
    dz = xr.DataArray(dz_values, coords=[ds_s[cfg['depth']]], dims=[cfg['depth']])


    valid_mask = (ds_s[cfg['salinity']] < MassBal['S_ref']) & (ds_s[cfg['depth']].notnull())
    fw_weight = (MassBal['S_ref'] - ds_s[cfg['salinity']]) / MassBal['S_ref']

    # Core Value Components
    meteoric_fw_fraction = fw_weight * ds["f_met"]
    seaicemelt_fw_fraction = fw_weight * ds["f_sim"]

    # Uncertainty Components (per layer)
    met_layer_unc = fw_weight * ds["f_met_unc"] * dz
    sim_layer_unc = fw_weight * ds["f_sim_unc"] * dz

    log_status("Assembling vertical integration and error graphs...")
    # Standard Integration for inventories
    meteoric_inventory = (meteoric_fw_fraction * dz).where(valid_mask).sum(dim=cfg['depth'])
    seaicemelt_inventory = (
        (seaicemelt_fw_fraction * dz).where(valid_mask).sum(dim=cfg['depth'])
    )

    # Root-Sum-of-Squares Integration for uncertainties
    meteoric_inventory_unc = (
        np.sqrt((met_layer_unc**2).where(valid_mask).sum(dim=cfg['depth']))
    )
    seaicemelt_inventory_unc = (
        np.sqrt((sim_layer_unc**2).where(valid_mask).sum(dim=cfg['depth']))
    )

    # --- 5. COMBINE FIELDS INTO FINAL DATASETS ---
    mw_ds = xr.Dataset(
        {
            "meteoric_freshwater_thickness": meteoric_inventory,
            "meteoric_freshwater_thickness_uncertainty": meteoric_inventory_unc,
        }
    )
    mw_ds["meteoric_freshwater_thickness"].attrs = {
        "units": "meters",
        "description": f"Meteoric liquid freshwater inventory integrated down to Salinity = {MassBal['S_ref']}",
    }
    mw_ds["meteoric_freshwater_thickness_uncertainty"].attrs = {
        "units": "meters",
        "description": f"Propagated 1-sigma uncertainty for meteoric inventory",
    }

    sim_ds = xr.Dataset(
        {
            "seaicemelt_freshwater_thickness": seaicemelt_inventory,
            "seaicemelt_freshwater_thickness_uncertainty": seaicemelt_inventory_unc,
        }
    )
    sim_ds["seaicemelt_freshwater_thickness"].attrs = {
        "units": "meters",
        "description": f"Sea Ice Melt liquid freshwater inventory integrated down to Salinity = {MassBal['S_ref']}",
    }
    sim_ds["seaicemelt_freshwater_thickness_uncertainty"].attrs = {
        "units": "meters",
        "description": f"Propagated 1-sigma uncertainty for sea ice melt inventory",
    }

    # --- 6. TRIGGER COMPUTATION VIA DASK ---
    log_status("Handing execution over to workers...")
    mw_ds = mw_ds.compute()
    sim_ds = sim_ds.compute()
    log_status("Dask computation completed successfully!")

    # --- 7. SAVE TO DISK ---
    ds.to_netcdf(cfg['frac_output_path'])
    log_status(f"Saved meteoric and seaicemet fractions to: {cfg['frac_output_path']}")
    mw_ds.to_netcdf(cfg['mw_output_path'])
    log_status(f"Saved meteoric dataset with uncertainties to: {cfg['mw_output_path']}")
    sim_ds.to_netcdf(cfg['sim_output_path'])
    log_status(
        f"Saved sea ice melt dataset with uncertainties to: {cfg['sim_output_path']}"
    )

    client.close()
    cluster.close()

# --- 7. PLOTTING ANOMALY MAP ---

log_status("Loading generated inventory file for plotting...")
ds_map = xr.open_dataset(f"{cfg['work_dir']}/{cfg['mw_output_path']}")

log_status("Computing baseline vs recent climatology values and uncertainties...")

# 1. Slice your inventory value and uncertainty data arrays
mw_baseline_data = ds_map["meteoric_freshwater_thickness"].sel(
    time=slice(cfg['baseline_start'], cfg['baseline_end'])
)
mw_baseline_unc = ds_map["meteoric_freshwater_thickness_uncertainty"].sel(
    time=slice(cfg['baseline_start'], cfg['baseline_end'])
)

mw_recent_data = ds_map["meteoric_freshwater_thickness"].sel(
    time=slice(cfg['anomaly_start'], cfg['anomaly_end'])
)
mw_recent_unc = ds_map["meteoric_freshwater_thickness_uncertainty"].sel(
    time=slice(cfg['anomaly_start'], cfg['anomaly_end'])
)

# 2. Compute the standard means for the values
baseline = mw_baseline_data.mean(dim=cfg["time"])
recent = mw_recent_data.mean(dim=cfg["time"])
mw_anomaly = recent - baseline

# 3. Propagate uncertainty through the TIME MEAN
# Formula: sqrt(sum(unc^2)) / N
N_baseline = mw_baseline_unc.sizes[cfg["time"]]
baseline_uncertainty = np.sqrt((mw_baseline_unc**2).sum(dim=cfg["time"])) / N_baseline

N_recent = mw_recent_unc.sizes[cfg["time"]]
recent_uncertainty = np.sqrt((mw_recent_unc**2).sum(dim=cfg["time"])) / N_recent

# 4. Propagate uncertainty through the SUBTRACTION (recent - baseline)
# Formula: sqrt(unc_recent^2 + unc_baseline^2)
mw_anomaly_uncertainty = np.sqrt(recent_uncertainty**2 + baseline_uncertainty**2)

log_status("Generating Polar Stereographic Plot...")
projection = ccrs.NorthPolarStereo(central_longitude=0)
data_crs = ccrs.PlateCarree()

fig = plt.figure(figsize=(10, 8), dpi=150)
ax = plt.axes(projection=projection)
ax.set_extent([-180, 180, 50, 90], crs=data_crs)

theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
ax.gridlines(
    draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
)
# --- Draw the Beaufort Gyre Bounding Box ---
# Define the corner coordinates of the box (closing the loop back to the start)
box_lons = [-160, -130, -130, -160, -160]
box_lats = [70, 70, 80, 80, 70]

# Plot the bounding box
ax.plot(
    box_lons,
    box_lats,
    color="black",          # High contrast outline (or use 'yellow' / '#e63946' to pop)
    linewidth=2.0,          # Clear line thickness
    linestyle="--",         # Dashed line style
    transform=ccrs.PlateCarree(),  # Tells Cartopy these coordinates are standard Lat/Lon
    zorder=3,               # Ensures the box sits on top of the data map
    label="Beaufort Gyre Domain"
)

# # Optional: Add a text label inside or right next to the box
# ax.text(
#     -145,
#     70.5,
#     "Beaufort Gyre",
#     color="black",
#     fontsize=10,
#     weight="bold",
#     transform=ccrs.PlateCarree(),
#     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
#     zorder=4,
#     ha="center"
# )
vmax = 1.0
im = mw_anomaly.plot.pcolormesh(
    ax=ax,
    x="longitude",
    y="latitude",
    transform=data_crs,
    cmap=cmocean.cm.balance,
    vmin=-vmax,
    vmax=vmax,
    add_colorbar=False,
    zorder=1,
)

cbar = plt.colorbar(
    im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, extend="both"
)
cbar.set_label("Meteoric Freshwater Inventory Anomaly (meters)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

plt.title(
    f"Arctic Meteoric Freshwater Inventory Anomaly\n(Mean {cfg['anomaly_start']}/{cfg['anomaly_end']} minus Baseline {cfg['baseline_start']}–{cfg['baseline_end']})",
    fontsize=12,
    fontweight="bold",
    pad=20,
)

plt.tight_layout()
plot_output = "arctic_meteoric_anomaly_map.png"
plt.savefig(plot_output, bbox_inches="tight")
plt.show()
log_status(f"Map plotted and saved to: {plot_output}")

##################################
# --- 7. PLOTTING ANOMALY MAP ---
##################################
log_status("Loading generated inventory file for plotting...")
ds_map = xr.open_dataset(f"{cfg['work_dir']}/{cfg['sim_output_path']}")

log_status("Computing baseline vs recent climatology values and uncertainties...")

# 1. Slice your inventory value and uncertainty data arrays
sim_baseline_data = ds_map["seaicemelt_freshwater_thickness"].sel(
    time=slice(cfg['baseline_start'], cfg['baseline_end'])
)
sim_baseline_unc = ds_map["seaicemelt_freshwater_thickness_uncertainty"].sel(
    time=slice(cfg['baseline_start'], cfg['baseline_end'])
)

sim_recent_data = ds_map["seaicemelt_freshwater_thickness"].sel(
    time=slice(cfg['anomaly_start'], cfg['anomaly_end'])
)
sim_recent_unc = ds_map["seaicemelt_freshwater_thickness_uncertainty"].sel(
    time=slice(cfg['anomaly_start'], cfg['anomaly_end'])
)

# 2. Compute the standard means for the values
baseline = sim_baseline_data.mean(dim="time")
recent = sim_recent_data.mean(dim="time")
sim_anomaly = recent - baseline

# 3. Propagate uncertainty through the TIME MEAN
# Formula: sqrt(sum(unc^2)) / N
N_baseline = sim_baseline_unc.sizes["time"]
baseline_uncertainty = np.sqrt((sim_baseline_unc**2).sum(dim="time")) / N_baseline

N_recent = sim_recent_unc.sizes["time"]
recent_uncertainty = np.sqrt((sim_recent_unc**2).sum(dim="time")) / N_recent

# 4. Propagate uncertainty through the SUBTRACTION (recent - baseline)
# Formula: sqrt(unc_recent^2 + unc_baseline^2)
sim_anomaly_uncertainty = np.sqrt(recent_uncertainty**2 + baseline_uncertainty**2)

log_status("Generating Polar Stereographic Plot...")
projection = ccrs.NorthPolarStereo(central_longitude=0)
data_crs = ccrs.PlateCarree()

fig = plt.figure(figsize=(10, 8), dpi=150)
ax = plt.axes(projection=projection)
ax.set_extent([-180, 180, 50, 90], crs=data_crs)

theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
ax.gridlines(
    draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
)
# --- Draw the Beaufort Gyre Bounding Box ---
# Define the corner coordinates of the box (closing the loop back to the start)
box_lons = [-160, -130, -130, -160, -160]
box_lats = [70, 70, 80, 80, 70]

# Plot the bounding box
ax.plot(
    box_lons,
    box_lats,
    color="black",          # High contrast outline (or use 'yellow' / '#e63946' to pop)
    linewidth=2.0,          # Clear line thickness
    linestyle="--",         # Dashed line style
    transform=ccrs.PlateCarree(),  # Tells Cartopy these coordinates are standard Lat/Lon
    zorder=3,               # Ensures the box sits on top of the data map
    label="Beaufort Gyre Domain"
)

# # Optional: Add a text label inside or right next to the box
# ax.text(
#     -145,
#     70.5,
#     "Beaufort Gyre",
#     color="black",
#     fontsize=10,
#     weight="bold",
#     transform=ccrs.PlateCarree(),
#     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
#     zorder=4,
#     ha="center"
# )

vmax = 2.0
im = sim_anomaly.plot.pcolormesh(
    ax=ax,
    x="longitude",
    y="latitude",
    transform=data_crs,
    cmap=cmocean.cm.balance,
    vmin=-vmax,
    vmax=vmax,
    add_colorbar=False,
    zorder=1,
)

cbar = plt.colorbar(
    im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, extend="both"
)
cbar.set_label("SeaIce Melt Freshwater Inventory Anomaly (meters)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

plt.title(
    f"Arctic Sea Ice Melt Freshwater Inventory Anomaly\n Mean {cfg['anomaly_start']}/{cfg['anomaly_end']} minus Baseline {cfg['baseline_start']}–{cfg['baseline_end']})",
    fontsize=12,
    fontweight="bold",
    pad=20,
)

plt.tight_layout()
plot_output = "arctic_seaicemelt_anomaly_map.png"
plt.savefig(plot_output, bbox_inches="tight")
plt.show()
log_status(f"Map plotted and saved to: {plot_output}")
log_status("All tasks successfully completed!")

# ==============================================================================
# 1. METEORIC WATER VOLUME & UNCERTAINTY PROPAGATION
# ==============================================================================
log_status("Calculating grid cell areas for Meteoric volume conversion...")

R = 6371000.0
lats = np.deg2rad(mw_anomaly["latitude"].values)
lons = np.deg2rad(mw_anomaly["longitude"].values)
dlat = np.abs(np.diff(lats)[0])
dlon = np.abs(np.diff(lons)[0])

lat_mesh, _ = np.meshgrid(
    mw_anomaly["latitude"].values, mw_anomaly["longitude"].values, indexing="ij"
)
cell_areas_m2 = (R**2) * np.cos(np.deg2rad(lat_mesh)) * dlat * dlon

da_area = xr.DataArray(
    cell_areas_m2,
    coords=[mw_anomaly["latitude"], mw_anomaly["longitude"]],
    dims=["latitude", "longitude"],
)

# Convert Anomaly meters & Uncertainty meters to Cubic Kilometers (km3)
volume_km3_grid = (mw_anomaly * da_area) / 1e9
volume_uncertainty_km3_grid = (mw_anomaly_uncertainty * da_area) / 1e9

# --- Regional Volumetric Influxes (Beaufort Gyre) ---
bg_volume = volume_km3_grid.sel(
    latitude=slice(70, 80), longitude=slice(-160, -130)
)
bg_volume_uncertainty = volume_uncertainty_km3_grid.sel(
    latitude=slice(70, 80), longitude=slice(-160, -130)
)

total_bg_mw_increase = bg_volume.sum().values
# Root-sum-of-squares (RSS) for regional sum propagation
total_bg_mw_increase_uncertainty = np.sqrt(
    (bg_volume_uncertainty**2).sum()
).values

print(f"--- Meteoric Volumetric Analysis ---")
print(
    f"Total Net Meteoric Freshwater Accumulated in Beaufort Gyre: {total_bg_mw_increase:.3f} ± {total_bg_mw_increase_uncertainty:.3f} km³"
)

# --- Broad Scale Accumulation (North of 60°N) ---
arctic_volume = volume_km3_grid.sel(latitude=slice(60, 90))
arctic_volume_uncertainty = volume_uncertainty_km3_grid.sel(
    latitude=slice(60, 90))

total_arctic_mw_increase = arctic_volume.sum().values
total_arctic_mw_increase_uncertainty = np.sqrt(
    (arctic_volume_uncertainty**2).sum()
).values

print(
    f"Total Arctic-wide (North of 60°N) Meteoric Freshwater Change: {total_arctic_mw_increase:.3f} ± {total_arctic_mw_increase_uncertainty:.3f} km³"
)


# ==============================================================================
# 2. SEA ICE MELT VOLUME & UNCERTAINTY PROPAGATION
# ==============================================================================
log_status("Calculating grid cell areas for Sea Ice Melt volume conversion...")

lats = np.deg2rad(sim_anomaly["latitude"].values)
lons = np.deg2rad(sim_anomaly["longitude"].values)
dlat = np.abs(np.diff(lats)[0])
dlon = np.abs(np.diff(lons)[0])

lat_mesh, _ = np.meshgrid(
    sim_anomaly["latitude"].values, sim_anomaly["longitude"].values, indexing="ij"
)
cell_areas_m2 = (R**2) * np.cos(np.deg2rad(lat_mesh)) * dlat * dlon

da_area = xr.DataArray(
    cell_areas_m2,
    coords=[sim_anomaly["latitude"], sim_anomaly["longitude"]],
    dims=["latitude", "longitude"],
)

# Convert Anomaly meters & Uncertainty meters to Cubic Kilometers (km3)
volume_km3_grid = (sim_anomaly * da_area) / 1e9
volume_uncertainty_km3_grid = (sim_anomaly_uncertainty * da_area) / 1e9

# --- Regional Volumetric Influxes (Beaufort Gyre) ---
bg_volume = volume_km3_grid.sel(
    latitude=slice(70, 80), longitude=slice(-160, -130)
)
bg_volume_uncertainty = volume_uncertainty_km3_grid.sel(
    latitude=slice(70, 80), longitude=slice(-160, -130)
)

total_bg_sim_increase = bg_volume.sum().values
total_bg_sim_increase_uncertainty = np.sqrt(
    (bg_volume_uncertainty**2).sum()
).values

print(f"\n--- Sea Ice Melt Volumetric Analysis ---")
print(
    f"Total Net Sea Ice Melt Freshwater Accumulated in Beaufort Gyre: {total_bg_sim_increase:.3f} ± {total_bg_sim_increase_uncertainty:.3f} km³"
)

# --- Broad Scale Accumulation (North of 60°N) ---
arctic_volume = volume_km3_grid.sel(latitude=slice(60, 90))
arctic_volume_uncertainty = volume_uncertainty_km3_grid.sel(
    latitude=slice(60, 90))

total_arctic_sim_increase = arctic_volume.sum().values
total_arctic_sim_increase_uncertainty = np.sqrt(
    (arctic_volume_uncertainty**2).sum()
).values

print(
    f"Total Arctic-wide (North of 60°N) Sea Ice Melt Freshwater Change: {total_arctic_sim_increase:.3f} ± {total_arctic_sim_increase_uncertainty:.3f} km³"
)

# --- Calculate Meteoric Flux through a Davis Strait Transect ---
log_status("Setting up Labrador Sea entry transect...")
ds_frac = xr.open_dataset(f"{cfg['work_dir']}/{cfg['frac_output_path']}")
ecco_paths2 = glob.glob(os.path.join(cfg['inference_target'], "*.nc"))
log_status("Loading ECCO v4r4 dataset lazily...")
# Use glob to grab all NetCDF files in the folder

# if not file_list:
#     sys.exit(f"[ERROR] No NetCDF files found in: {nc_path}")

# Open and combine all datasets
# data_vars='minimal' merges variables that don't share all dimensions without bloating memory
# coords='minimal' does the same for coordinates
ds_v = xr.open_mfdataset(ecco_paths2, combine='by_coords', data_vars='minimal', coords='minimal')


log_status("Setting up diagonal transect calculation...")
for t_name, transect in transects.items():
    # --- 2. GENERATE PATH COORDINATES & DISTANCES ---
    # R = 6371000.0  # Earth's radius in meters

    # Linear interpolation in coordinate space
    transect_lats = np.linspace(transect['lat1'], transect['lat2'], transect['num_points'])
    transect_lons = np.linspace(transect['lon1'], transect['lon2'], transect['num_points'])

    # Calculate the distance metrics (dx, dy) between neighboring points along the track
    lat_rad = np.deg2rad(transect_lats)
    lon_rad = np.deg2rad(transect_lons)

    dlat_trans = np.diff(lat_rad)
    dlon_trans = np.diff(lon_rad)
    mid_lats = (lat_rad[:-1] + lat_rad[1:]) / 2.0

    # Physical distance components for each segment (in meters)
    dx = R * np.cos(mid_lats) * dlon_trans
    dy = R * dlat_trans
    ds_segment = np.sqrt(dx**2 + dy**2)  # Total segment length

    # Midpoints of segments where we will sample our data variables
    sample_lats = np.rad2deg(mid_lats)
    sample_lons = np.rad2deg((lon_rad[:-1] + lon_rad[1:]) / 2.0)

    # Convert arrays into Xarray DataArrays for vectorized profile extraction
    da_lats = xr.DataArray(sample_lats, dims="segment")
    da_lons = xr.DataArray(sample_lons, dims="segment")
    da_ds = xr.DataArray(ds_segment, dims="segment")
    da_dx = xr.DataArray(dx, dims="segment")
    da_dy = xr.DataArray(dy, dims="segment")

    # --- 3. EXTRACT TRANSECT PROFILES VIA INTERPOLATION ---
    log_status("Extracting grid parameters along diagonal path...")

    # Extract variables across the diagonal profile using 2D bilinear interpolation
    # f_met_trans = ds_frac["f_met"].interp(latitude=da_lats, longitude=da_lons)
    f_sim_trans = ds_frac["f_sim"].interp(latitude=da_lats,longitude=da_lons)
    vxo_trans = ds_v["EVEL"].interp(latitude=da_lats, longitude=da_lons)
    vyo_trans = ds_v["NVEL"].interp(latitude=da_lats, longitude=da_lons)

    # Grab and match our layer thicknesses (dz)
    # # 4. Align depth layer thicknesses (dz) matching the depth dimensions
    # # Use the same dz array calculated in your main script
    # # Broadcast it so it matches the dimensions of our transect slice
    depths2 = ds_frac["Z"].values
    bounds2 = np.zeros(len(depths2) + 1)
    bounds2[1:-1] = (depths2[:-1] + depths2[1:]) / 2
    bounds2[0] = 0
    bounds2[-1] = depths2[-1] + (depths2[-1] - bounds2[-2])
    dz_values2 = np.diff(bounds2)
    dz2 = xr.DataArray(dz_values2, coords=[ds_v["Z"]], dims=["Z"])
    # dz_trans = dz2.sel(Z=f_met_trans["Z"])
    dz_trans = dz2.sel(Z=f_sim_trans["Z"])

    # --- 4. PROJECT VELOCITY VECTORS PERPENDICULAR TO PATH ---
    log_status("Projecting velocity vectors perpendicular to transect...")

    # Normal vector formula: v_normal = (-vxo * dy + vyo * dx) / ds
    # This calculates the exact volume component moving orthogonally across your track line
    v_normal = (-vxo_trans * da_dy + vyo_trans * da_dx) / da_ds

    # --- 5. COMPUTE MASS INTEGRATION & CONVERT TO mSv ---
    log_status("Integrating total column flux...")

    # Cell Flux = Velocity_normal * Meteoric_fraction * depth_layer * segment_width
    # met_cell_flux_m3s = v_normal * f_met_trans * dz_trans * da_ds
    sim_cell_flux_m3s = v_normal * f_sim_trans * dz_trans * da_ds

    # Sum over depth and the segments along the transect line
    # met_total_flux_m3s = met_cell_flux_m3s.sum(dim=["Z", "segment"])
    sim_total_flux_m3s = sim_cell_flux_m3s.sum(dim=["Z", "segment"])
    # Convert from cubic meters per second to milliSverdrups (1 mSv = 1000 m3/s)
    # met_flux_mSv = met_total_flux_m3s / 1000.0
    sim_flux_mSv = sim_total_flux_m3s / 1000.0
    # # Clean up attributes
    # met_flux_mSv.name = "meteoric_flux_labrador"
    # met_flux_mSv.attrs["units"] = "mSv"
    # met_flux_mSv.attrs["description"] = (
    #     "Meteoric liquid water volume transport flux in milliSverdrups"
    # )
    sim_flux_mSv.name = "seaicemelt_flux_labrador"
    sim_flux_mSv.attrs["units"] = "mSv"
    sim_flux_mSv.attrs["description"] = (
        "Seaice melt liquid water volume transport flux in milliSverdrups"
    )

    log_status("Flux pipeline assembled. Processing completed!")



    # --- Plot the Time Series ---
    log_status("Plotting flux time series...")


    # plt.figure(figsize=(10, 4), dpi=150)
    # met_flux_mSv.plot(linewidth=1.5, color=cmocean.cm.balance(0.2))

    # plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    # plt.title(
    #     f"Meteoric Water Volume Flux through {t_name}",
    #     fontsize=12,
    #     fontweight="bold",
    # )
    # plt.ylabel("Flux mSv/s", fontsize=11)
    # plt.xlabel("Year", fontsize=11)
    # plt.grid(alpha=0.3)
    # plt.savefig(f"{t_name}_meteoric_flux.png", bbox_inches="tight")
    # plt.show()

    plt.figure(figsize=(10, 4), dpi=150)
    sim_flux_mSv.plot(linewidth=1.5, color=cmocean.cm.balance(0.2))

    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title(
        f"Seaice Melt Water Volume Flux through {t_name}",
        fontsize=12,
        fontweight="bold",
    )
    plt.ylabel("Flux mSv/s", fontsize=11)
    plt.xlabel("Year", fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(f"{t_name}sim_flux.png", bbox_inches="tight")
    plt.show()


# --- 1. SET UP THE POLAR PROJECTED MAP ---
fig2 = plt.subplots(figsize=(8, 8), subplot_kw={"projection": ccrs.NorthPolarStereo()})
ax2 = fig2[1]

# Set spatial boundaries around the Labrador Sea / Baffin Bay region
# Format: [lon_min, lon_max, lat_min, lat_max]
ax2.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())

# --- 2. ADD LANDMASS AND GEOGRAPHY METRICS ---
ax2.add_feature(cfeature.LAND, facecolor="#f4f4f4", edgecolor="gray", zorder=1)
ax2.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#555555", zorder=2)
ax2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray", zorder=2)

# Add gridlines with labels
gl = ax2.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color="gray",
    alpha=0.5,
    linestyle="--",
    zorder=3,
)
gl.top_labels = False
gl.right_labels = False

for t_name, transect in transects.items():

    # Plot the path line
    # Note: transform=ccrs.Geodetic() forces Cartopy to draw a true great-circle line
    ax2.plot(
        [transect['lon1'], transect['lon2']],
        [transect['lat1'], transect['lat2']],
        color="#e63946",
        linewidth=3,
        linestyle="-",
        transform=ccrs.Geodetic(),
        zorder=4,
        label="Flux Transect Line",
    )

    # Plot the endpoint markers (Point A and Point B)
    ax2.scatter(
        [transect['lon1'], transect['lon2']],
        [transect['lat1'], transect['lat2']],
        color="#1d3557",
        s=60,
        edgecolor="white",
        linewidth=1,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # Add text annotations next to the points
    ax2.text(
        transect['lon1'] - 2,
        transect['lat1'] + 0.5,
        t_name,
        transform=ccrs.PlateCarree(),
        fontsize=12,
        weight="bold",
        zorder=6,
    )
    # ax2.text(
    #     transect['lon2'] + 1,
    #     transect['lat2'] - 0.5,
    #     "B",
    #     transform=ccrs.PlateCarree(),
    #     fontsize=12,
    #     weight="bold",
    #     zorder=6,
    # )

# --- 4. FINALIZE AND DISPLAY ---
ax2.set_title(
    f"Transect Locations", fontsize=14, pad=20, weight="bold"
)
ax2.legend(loc="lower left", framealpha=0.9)

plt.savefig("transect_location_map.png", bbox_inches="tight", dpi=200)
plt.show()
