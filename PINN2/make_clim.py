import xarray as xr


def monthly_climatology(
    input_path: str, output_path: str, time_dim: str = "time"
) -> xr.Dataset:
    """
    Calculate the monthly climatological mean of a dataset and save as NetCDF.

    Parameters
    ----------
    input_path  : Path to the input NetCDF file.
    output_path : Path for the output NetCDF file.
    time_dim    : Name of the time dimension (default: "time").

    Returns
    -------
    clim : xarray.Dataset containing the 12-month climatological means.
    """
    # Load dataset
    ds = xr.open_dataset(input_path)

    # Group by month and compute the mean across all years
    clim = ds.groupby(f"{time_dim}.month").mean(dim=time_dim)

    # Add descriptive metadata
    clim.attrs.update(
        {
            "description": "Monthly climatological mean",
            "source_file": input_path,
        }
    )

    # Save to NetCDF
    clim.to_netcdf(output_path)
    print(f"Climatology saved to: {output_path}")

    return clim


# --- Example usage ---
clim = monthly_climatology(
    input_path="/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_vyo-vxo_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-01-01.nc",
    output_path="/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_vyo-vxo_monthly_climatology.nc",
)

print(clim)
