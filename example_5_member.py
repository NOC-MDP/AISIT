import numpy as np
import xarray as xr
from scipy.optimize import lsq_linear
from dask.distributed import Client, LocalCluster

# -------------------------------------------------------------------------
# 1. Base Setup & Parameters
# -------------------------------------------------------------------------
base_end_members = {
    "ATL": np.array([1.0, 34.80, 0.30, 45.0]),
    "PAC": np.array([1.0, 32.50, -1.10, 78.0]),
    "NAM": np.array([1.0, 0.00, -19.50, 130.0]),
    "EUR": np.array([1.0, 0.00, -19.00, 45.0]),
    "SIM": np.array([1.0, 4.00, 2.00, 10.0]),
}

end_member_std = {
    "ATL": np.array([0.0, 0.05, 0.05, 3.0]),
    "PAC": np.array([0.0, 0.30, 0.15, 5.0]),
    "NAM": np.array([0.0, 0.00, 1.50, 20.0]),
    "EUR": np.array([0.0, 0.00, 1.00, 8.0]),
    "SIM": np.array([0.0, 1.00, 0.50, 3.0]),
}

obs_uncertainty = np.array([0.0, 0.01, 0.05, 1.5])
weights = np.array([100.0, 25.0, 10.0, 5.0])
water_masses = ["ATL", "PAC", "NAM", "EUR", "SIM"]


# -------------------------------------------------------------------------
# 2. Vectorized 1D Solvers (Core Engine)
# -------------------------------------------------------------------------
def solve_single_5comp_omp(A_raw, obs_raw, min_sim=-0.20):
    """Solves one OMP realization for a single grid cell."""
    mean_vals = np.mean(A_raw, axis=1)
    std_vals = np.std(A_raw, axis=1)
    std_vals[0] = 1.0
    mean_vals[0] = 0.0

    A_norm = (A_raw - mean_vals[:, None]) / std_vals[:, None]
    A_weighted = A_norm * weights[:, None]

    obs_norm = (obs_raw - mean_vals) / std_vals
    obs_weighted = obs_norm * weights

    lower_bounds = [0.0, 0.0, 0.0, 0.0, min_sim]
    upper_bounds = [1.0, 1.0, 1.0, 1.0, 1.00]

    res = lsq_linear(A_weighted, obs_weighted, bounds=(lower_bounds, upper_bounds))
    x = res.x
    f_sum = np.sum(x)

    return x / f_sum if f_sum != 0 else x


def point_monte_carlo(sal, d18O, ba, n_iter=1000, min_sim=-0.20):
    """
    1D target function for xr.apply_ufunc.
    Handles NaN values (e.g., land masks) and returns mean & std.
    """
    if np.isnan(sal) or np.isnan(d18O) or np.isnan(ba):
        # Return NaNs for 5 means and 5 stds
        return np.full((10,), np.nan)

    base_obs = np.array([1.0, sal, d18O, ba])
    results = np.zeros((n_iter, 5))

    for i in range(n_iter):
        # Perturb End-Members
        A_perturbed = np.zeros((4, 5))
        for j, wm in enumerate(water_masses):
            noise = np.random.normal(0, end_member_std[wm])
            A_perturbed[:, j] = base_end_members[wm] + noise

        # Perturb Observations
        obs_noise = np.random.normal(0, obs_uncertainty)
        obs_perturbed = base_obs + obs_noise

        results[i, :] = solve_single_5comp_omp(
            A_perturbed, obs_perturbed, min_sim=min_sim
        )

    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)

    return np.concatenate([means, stds])


# -------------------------------------------------------------------------
# 3. xarray Wrapper Function
# -------------------------------------------------------------------------
def run_omp_xarray(ds_o,ds_Ba,ds_s, sal_var="SALT", n_iter=500):
    """
    Runs Monte Carlo OMP across all dimensions of an xarray Dataset.
    """
    # xr.apply_ufunc applies the function across all grid cells
    combined_stats = xr.apply_ufunc(
        point_monte_carlo,
        ds_s[sal_var],
        ds_o["tracer_pred"],
        ds_Ba["tracer_pred"],
        kwargs={"n_iter": n_iter},
        input_core_dims=[[], [], []],
        output_core_dims=[["stat_and_mass"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"stat_and_mass": 10}},
    )

    # Re-organize output vector into distinct Dataset variables
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
    DEPTH = 100

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
    results_ds = run_omp_xarray(ds_o, ds_Ba, ds_s, n_iter=100)

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
