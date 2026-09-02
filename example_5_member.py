import numpy as np
import xarray as xr
from scipy.optimize import nnls
from dask.distributed import Client, LocalCluster
import numba

# -------------------------------------------------------------------------
# 1. Base Setup & Parameters (5 Tracers, 5 End Members)
# Tracers: [Mass (1.0), Salinity, d18O, Barium, Potential Temp (degC)]
# -------------------------------------------------------------------------
# Columns: [Mass, SALT, d18O (‰), Ba (nmol/kg), PTemp (°C)]
base_end_members = {
    "ATL": np.array([1.0, 34.80,   0.30,  48.0,  2.50]),  # Atlantic Water (Fram Strait / Barents)
    "PAC": np.array([1.0, 32.50,  -1.10,  75.0,  0.50]),  # Pacific Summer/Halocline Water
    "NAM": np.array([1.0,  0.00, -20.50,  95.0,  0.00]),  # North American River Runoff
    "EUR": np.array([1.0,  0.00, -17.50,  45.0,  0.00]),  # Eurasian River Runoff
    "SIM": np.array([1.0,  4.00,   0.80,   5.0, -1.80]),  # Sea Ice Meltwater
}
# Columns: [Mass, SALT, d18O (‰), Ba (nmol/kg), PTemp (°C)]
end_member_std = {
    "ATL": np.array([0.0, 0.05, 0.05,  3.0, 0.50]),  # Higher PTemp variance due to warming/pulses
    "PAC": np.array([0.0, 0.30, 0.15,  5.0, 0.50]),  # Solid for Pacific Upper Halocline/Summer Water
    "NAM": np.array([0.0, 0.00, 1.50, 10.0, 1.00]),  # Reduced Ba uncertainty to reflect flow-weighted mean
    "EUR": np.array([0.0, 0.00, 1.20, 10.0, 1.00]),  # Slightly increased Ba/d18O to reflect multi-river blend
    "SIM": np.array([0.0, 1.00, 0.80,  3.0, 0.10]),  # Increased d18O spread (FYI vs MYI), reduced PTemp
}
water_masses = ["ATL", "PAC", "NAM", "EUR", "SIM"]

# Column Order: [Mass, SALT, d18O, Ba, PTemp]
# 1. Observational/Measurement Uncertainty
obs_uncertainty = np.array([0.0, 0.01, 0.05, 1.5, 0.02])
# 2. Recommended Parameter Weights for OMP / pyOMP
# Normalized such that Mass = 100.0 (or 1.0 depending on solver scaling)
parameter_weights = np.array([100.0, 25.0, 30.0, 10.0, 8.0])

A_base = np.column_stack([base_end_members[wm] for wm in water_masses])
A_std_matrix = np.column_stack([end_member_std[wm] for wm in water_masses])

# -------------------------------------------------------------------------
# 2. Vectorized Point/Block Solver (1D core dim = 'tracer')
# -------------------------------------------------------------------------
def solve_omp_1d(obs_vec, n_iter=100):
    """
    Solves OMP over a single 1D tracer vector of length 5:
    obs_vec = [Mass, Salinity, d18O, Ba, Temp]
    Returns 1D array of length 10 -> [5 means, 5 stds]
    """
    if np.isnan(obs_vec).any():
        return np.full((10,), np.nan, dtype=np.float64)

    n_tracers, n_members = 5, 5
    W = parameter_weights[:, None]

    # Pre-generate noise for iterations
    noise_A = np.random.normal(0, 1, size=(n_iter, n_tracers, n_members)) * A_std_matrix[None, :, :]
    A_perturbed = (A_base[None, :, :] + noise_A) * W[None, :, :]

    noise_obs = np.random.normal(0, 1, size=(n_iter, n_tracers)) * obs_uncertainty[None, :]
    obs_perturbed = (obs_vec[None, :] + noise_obs) * parameter_weights[None, :]

    results = np.zeros((n_iter, n_members), dtype=np.float64)

    # Solve across realizations using Fast Non-Negative Least Squares
    for k in range(n_iter):
        x, _ = nnls(A_perturbed[k], obs_perturbed[k])
        f_sum = np.sum(x)
        results[k, :] = x / f_sum if f_sum > 0 else x

    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)

    return np.concatenate([means, stds])


# -------------------------------------------------------------------------
# 3. Corrected xarray Wrapper (Dask-Parallelized)
# -------------------------------------------------------------------------
def run_omp_xarray_fast(ds_o, ds_Ba, ds_s, sal_var="SALT", temp_var="THETA", n_iter=100):
    """
    Applies point-wise vectorized OMP across spatial Dask chunks safely.
    """
    # 1. Create uniform 1.0 array for Mass conservation
    mass_arr = xr.ones_like(ds_s[sal_var])

    # 2. Stack input tracers along a new 'tracer' dimension
    obs_combined = xr.concat([
        mass_arr,
        ds_s[sal_var],
        ds_o["tracer_pred"],
        ds_Ba["tracer_pred"],
        ds_s[temp_var]
    ], dim="tracer")

    # Ensure tracer is unchunked (size 5)
    obs_combined = obs_combined.chunk({"tracer": -1})

    # 3. Apply ufunc (only 'tracer' is the core dimension; lat/lon remain broadcast dimensions)
    combined_stats = xr.apply_ufunc(
        solve_omp_1d,
        obs_combined,
        kwargs={"n_iter": n_iter},
        input_core_dims=[["tracer"]],
        output_core_dims=[["stat_and_mass"]],
        vectorize=True,  # Automatically loops over lat/lon chunks
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "output_sizes": {"stat_and_mass": 10}
        },
    )

    out_ds = xr.Dataset()
    for i, wm in enumerate(water_masses):
        out_ds[f"{wm}_fraction_mean"] = combined_stats.isel(stat_and_mass=i)
        out_ds[f"{wm}_fraction_std"] = combined_stats.isel(stat_and_mass=i + 5)

    return out_ds


# -------------------------------------------------------------------------
# 4. Example Usage with Synthetic Monthly Data
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Example Execution & Formatted Output Print
# -------------------------------------------------------------------------


if __name__ == "__main__":
    # 1. Connect to cluster (e.g. SLURM / Local cluster)
    # For a local multi-core machine:
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dask Dashboard link for monitoring: {client.dashboard_link}")
    # If on an HPC cluster using dask_jobqueue (e.g., SLURM):
    # from dask_jobqueue import SLURMCluster
    # cluster = SLURMCluster(...)
    # cluster.scale(jobs=10)
    # client = Client(cluster)

    MONTH = "09"
    YEAR = "2016"
    DEPTH = 10

    # 2. Open Datasets WITH CHUNKS (Crucial for Dask)
    # Choose chunk sizes based on spatial/time dimensions (e.g. chunking spatial dimensions)
    ds_o = xr.open_dataset(f"/gws/ssde/j25a/nemo/vol4/thopri/PINN/infer_output_ECCO/tracer_predicted_{YEAR}_{MONTH}.nc", chunks={"latitude": 100, "longitude": 100,'Z': 1})
    ds_Ba = xr.open_dataset(f"/gws/ssde/j25a/nemo/vol4/thopri/PINN_BARIUM/infer_output_ECCO/tracer_predicted_{YEAR}_{MONTH}.nc", chunks={"latitude": 100, "longitude": 100,'Z': 1})
    ds_s = xr.open_dataset(f"/work/scratch-pw5/thopri/ECCO/OCEAN_TEMPERATURE_SALINITY_mon_mean_{YEAR}-{MONTH}_ECCO_V4r4_latlon_0p50deg.nc", chunks={"time": 1, "latitude": 100, "longitude": 100,'Z': 1})
    ds_s = ds_s.sel(time=f"{YEAR}-{MONTH}-15",Z=DEPTH, method='nearest')
    ds_o = ds_o.sel(Z=DEPTH, method='nearest')
    ds_Ba = ds_Ba.sel(Z=DEPTH,method='nearest')

    # Print chunks to confirm it split into multiple tasks:
    print(ds_s["SALT"].chunks)

    print("Running OMP Monte Carlo across xarray cube via Dask...")
    results_ds = run_omp_xarray_fast(ds_o, ds_Ba, ds_s, n_iter=100)

    # 3. Compute and Save using Dask
    # to_netcdf with dask arrays triggers distributed computation automatically
    write_job = results_ds.to_netcdf(
        f"/gws/ssde/j25a/nemo/vol4/thopri/MASSBAL/watermass_fractions_depth_{DEPTH}_{YEAR}_{MONTH}.nc",
        compute=False  # Returns a dask delayed object
    )


    print("Computing on Dask cluster...")
    write_job.compute()
    # ---------------------------------------------------------------------
    # Table Print: Selected Single Point (e.g., time=0, lat=71, lon=-149)
    # ---------------------------------------------------------------------
    sel_time = f"{YEAR}-{MONTH}-15"
    sel_lat = 71
    sel_lon = -149

    pt_res = results_ds.sel(latitude=sel_lat, longitude=sel_lon,method='nearest')

    print(
        f"=== OMP Results for Single Grid Cell (time={sel_time}, lat={sel_lat}, lon={sel_lon}) ==="
    )
    print(
        f"{'Water Mass':<14} | {'Mean Fraction':<15} | {'Std Dev (±)':<12}"
    )
    print("-" * 47)

    for wm in water_masses:
        m = pt_res[f"{wm}_fraction_mean"].values * 100
        s = pt_res[f"{wm}_fraction_std"].values * 100
        print(f"{wm:<14} | {m:>6.2f}%         | ±{s:>5.2f}%")

    print("\n")

    # ---------------------------------------------------------------------
    # Table Print: Spatial Mean across the entire domain for Month 1
    # ---------------------------------------------------------------------
    domain_mean = results_ds.mean(dim=["latitude", "longitude"])

    print(f"=== Domain-Averaged OMP Results for {sel_time} ===")
    print(
        f"{'Water Mass':<14} | {'Mean Fraction':<15} | {'Std Dev (±)':<12}"
    )
    print("-" * 47)

    for wm in water_masses:
        m = domain_mean[f"{wm}_fraction_mean"].values * 100
        s = domain_mean[f"{wm}_fraction_std"].values * 100
        print(f"{wm:<14} | {m:>6.2f}%         | ±{s:>5.2f}%")
