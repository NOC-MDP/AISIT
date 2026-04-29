import xarray as xr

work_dir = "/work/scratch-pw5/thopri"
ds1 = xr.open_dataset(
    f"{work_dir}/cmems_mod_arc_phy_my_topaz4_P1M_thetao-so_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-01-01.nc"
)
ds2 = xr.open_dataset(
    f"{work_dir}/cmems_mod_arc_phy_my_topaz4_P1M_vyo-vxo_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-01-01.nc"
)

ds = xr.merge([ds1, ds2])
ds.to_netcdf(f"{work_dir}/cmems_combined.nc")
