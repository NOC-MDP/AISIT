import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmocean
import pandas as pd
import gsw

# def solve_mixing(d18O_obs, S_obs):
#     """Solve the 3 end-member system for scalar or array inputs."""

#         # --- End member definitions ---
#     d18O_sw = 0.0       # Seawater
#     d18O_fw = -21 # -70.0     # Freshwater (meteoric)
#     d18O_si = 2.0       # Sea ice
    
#     S_sw    = 34.8      # Seawater salinity
#     S_fw    = 0.0       # Freshwater salinity
#     S_si    = 4.0       # Sea ice salinity
    
#     A = np.array([
#         [1,       1,       1      ],
#         [S_sw,    S_fw,    S_si   ],
#         [d18O_sw, d18O_fw, d18O_si]
#     ])
    
#     b = np.stack([
#         np.ones_like(d18O_obs),
#         S_obs,
#         d18O_obs
#     ], axis=-1)                          # shape (..., 3)

#     # np.linalg.solve broadcasts over leading dims with (..., 3, 3) / (..., 3)
#     A_inv = np.linalg.inv(A)
#     fractions = b @ A_inv.T              
#     # shape (..., 3)
#     return fractions

# def mass_fraction(oxy_iso, sal, dmw):
#     """
#     Function for computing the mass fraction of freshwater oxygen isotopes.
#     Parameters
#     ----------
#     oxy_iso, sal, dmw : xarray.DataArray
#         DataArrays for oxygen isotopes, salinity, and meteoric water d18O
#     Returns
#     -------
#     fsw, fmw, fsi : xarray.DataArray
#     """
#     ssw, smw, ssi = 34.8, 0.0, 3.0
#     dsw, dsi = 0.34, 2.1

#     # Convert everything to numpy — avoids ALL xarray alignment issues
#     d18o_np = np.asarray(oxy_iso)
#     sal_np  = np.asarray(sal)
#     dmw_np  = np.asarray(dmw)

#     # Broadcast dmw to match oxy_iso/sal shape if needed (e.g. no depth dim)
#     dmw_np = np.broadcast_to(dmw_np, d18o_np.shape)

#     def _solve(d18o_val, sal_val, dmw_val):
#         k = np.array([[1,   1,       1  ],
#                       [dsw, dmw_val, dsi],
#                       [ssw, smw,     ssi]])
#         rhs = np.array([1.0, d18o_val, sal_val])
#         return np.linalg.solve(k, rhs)

#     result = np.vectorize(_solve, signature="(),(),()->(n)")(d18o_np, sal_np, dmw_np)
#     # result shape: (*original_dims, 3)

#     # Wrap back into DataArrays using oxy_iso as the coordinate template
#     fsw = xr.DataArray(result[..., 0], coords=oxy_iso.coords, dims=oxy_iso.dims)
#     fmw = xr.DataArray(result[..., 1], coords=oxy_iso.coords, dims=oxy_iso.dims)
#     fsi = xr.DataArray(result[..., 2], coords=oxy_iso.coords, dims=oxy_iso.dims)

#     return fsw, fmw, fsi

start_year = 1991
end_year = 2025
depth_inter = 350
    # Beaufort Gyre,buffer zone,  Pan Arctic
lat_n = 80.5 #90 #90 
lat_s = 70.5 #75 #50
lon_e = -130 #-10 #180
lon_w = -170 #-130 #-180
model_years = [year for year in range(start_year, end_year + 1)]
# model_month = 10
mean_mwc = []
mean_mwc_km3 = []
months = []
ds_st = xr.open_dataset(f"/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_thetao-so_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-01-01.nc")

# depth coordinate
depth = ds_st.depth
# depth levels in meters
depth_levels = np.array([
            0,2, 4, 6, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
            125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 600, 700,
            800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000
        ])
for model_year in model_years:
    for i in range(12):
        months.append(f"{model_year}/{i+1:02d}")


        ds_o = xr.open_dataset(f"/gws/ssde/j25a/nemo/vol4/thopri/PINN/infer_output/tracer_predicted_{model_year}_{i+1:02d}.nc")
        d18O_obs = ds_o["tracer_pred"].sel(latitude=slice(lat_s,lat_n),longitude=slice(lon_w,lon_e))
        S_obs    = ds_st["so"].sel(latitude=slice(lat_s,lat_n),longitude=slice(lon_w,lon_e),
                              time=(ds_st.time.dt.year == model_year) & 
                                      (ds_st.time.dt.month == i+1)).squeeze("time")

        T_obs    = ds_st["thetao"].sel(latitude=slice(lat_s,lat_n),longitude=slice(lon_w,lon_e),
                              time=(ds_st.time.dt.year == model_year) & 
                                      (ds_st.time.dt.month == i+1)).squeeze("time")

        lats = ds_st["latitude"].sel(latitude=slice(lat_s,lat_n))
        lons = ds_st["longitude"].sel(longitude=slice(lon_w,lon_e))

        fmw_ds = xr.open_dataset("massfraction_field.nc")
        fmw = fmw_ds["field"].sel(latitude=slice(lat_s,lat_n),longitude=slice(lon_w,lon_e))


        # ============================================================
        # COMPUTE DENSITY
        # ============================================================
        
        # Build pressure from depth
        #
        # gsw.p_from_z expects NEGATIVE depth
        
        pressure = xr.apply_ufunc(
            gsw.p_from_z,
            -depth,
            lats,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # Convert Practical Salinity -> Absolute Salinity

        SA = xr.apply_ufunc(
            gsw.SA_from_SP,
            S_obs,
            pressure,
            lons,
            lats,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        
        # Conservative Temperature
        
        CT = xr.apply_ufunc(
            gsw.CT_from_pt,
            SA,
            T_obs,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # Potential density anomaly referenced to surface
        #
        # sigma0 = density - 1000 kg/m3
        
        sigma0 = xr.apply_ufunc(
            gsw.sigma0,
            SA,
            CT,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # ============================================================
        # SURFACE DENSITY
        # ============================================================
        
        # Surface density at each horizontal location
        
        sigma_surface = sigma0.isel(depth=0)
        
        # ============================================================
        # DENSITY DIFFERENCE
        # ============================================================
        
        # Compute how similar each subsurface cell
        # is to the surface density
        
        delta_sigma = np.abs(sigma0 - sigma_surface)
        
        # ============================================================
        # PROPAGATE METEORIC FRACTION DOWNWARD
        # ============================================================
        
        # Density tolerance (kg/m3)
        #
        # Smaller:
        #   tighter stratification following
        #
        # Larger:
        #   deeper penetration
        
        sigma_scale = 0.15
        
        # Weight based on density similarity
        #
        # 1 near matching density
        # decays away from matching density
        
        density_weight = np.exp(
            -(delta_sigma / sigma_scale) ** 2
        )

        # fractions = solve_mixing(d18O_obs, S_obs)   # shape (..., 3)

        # Endmember values collated from https://pmc.ncbi.nlm.nih.gov/articles/PMC5852023/
        
        # River d18O source value
        
        d18O_river = -20.0
        
        # Marine background d18O
        
        d18O_ocean = 0.3
        
        # Apply to surface meteoric fraction
        
        fmw_depth = fmw * density_weight

        # Ensure values remain between 0 and 1

        fmw_depth = fmw_depth.clip(0, 1)
        
        # # e-folding decay scale (m)
        # H = 50.0
        
        # # vertical decay weights
        # weights = xr.DataArray(
        #     np.exp(-depth / H),
        #     coords={"depth": depth},
        #     dims=["depth"],
        # )
        
        # # expand river field vertically
        # fmw_depth = fmw.expand_dims(
        #     depth=depth
        # )
        
        # # broadcast weights onto 3D field
        # weights, fmw_depth = xr.broadcast(
        #     weights,
        #     fmw_depth
        # )

        # Create spatially varying meteoric endmember
        #
        # This blends river and marine influence
        
        d18O_meteoric = (
            fmw_depth * d18O_river
            +
            (1.0 - fmw_depth) * d18O_ocean
        )

        # ============================================================
        # OTHER ENDMEMBERS
        # ============================================================
        
        # salinity endmembers
        S_ocean = 34.8
        S_meteoric = 0.0
        S_ice = 4.0
        
        # Sea ice melt d180 endmember
        d18O_ice = -2.0
        # fws,fmw,fsi = mass_fraction(d18O_obs, S_obs, dmw_depth)
        # sea ice = fractions[..., 2]
        # seawater fractions[..., 0]
        
        # f_fw = fractions[..., 1] 
        # ============================================================
        # SOLVE 3-ENDMEMBER MIXING SYSTEM
        # ============================================================
        
        # Unknowns:
        #
        # fm = meteoric fraction
        # fi = sea ice melt fraction
        # fo = ocean fraction
        
        #
        # Equations:
        #
        # fm + fi + fo = 1
        #
        # fm*Sm + fi*Si + fo*So = Sobs
        #
        # fm*dm + fi*di + fo*do = d18Oobs
        #
        # Since meteoric salinity = 0:
        #
        # Sm = 0
        

        
        # Rearranged linear system:
        #
        # [1  1 ] [fm] = [1 - fo]
        # [Si So] [fi] = [Sobs]
        #
        # Easier approach:
        #
        # Eliminate fo:
        #
        # fo = 1 - fm - fi
        
        #
        # Then solve 2x2 system
        
        A11 = S_meteoric - S_ocean
        A12 = S_ice - S_ocean
        
        A21 = d18O_meteoric - d18O_ocean
        A22 = d18O_ice - d18O_ocean
        
        B1 = S_obs - S_ocean
        B2 = d18O_obs - d18O_ocean
        
        # Determinant
        det = A11 * A22 - A12 * A21
        
        # Solve fractions
        fm = (B1 * A22 - B2 * A12) / det
        
        fi = (A11 * B2 - A21 * B1) / det
        
        fo = 1.0 - fm - fi
        
        # ============================================================
        # OPTIONAL CLEANUP
        # ============================================================
        
        # Physical bounds
        fm = fm.clip(0, 1)
        fi = fi.clip(0, 1)
        fo = fo.clip(0, 1)
        # f_mw = fmw
        # f_mw = np.clip(f_mw, 0, 1)

        
        mask = depth_levels <= depth_inter
        depth_300 = depth_levels[mask]              # depths 0–300m only
        f_mw_300  = fm[mask, :, :]        # slice depth axis of the 3D array
        # Fill NaNs with 0 so shallow columns still integrate over valid levels
        f_mw_300  = np.nan_to_num(f_mw_300, nan=0.0)
        
        MWC = np.trapezoid(f_mw_300, depth_300, axis=0)
        # lats = ds['latitude']
        # lons = ds['longitude']
        # Mask where ALL depth levels are NaN = true land/no-data
        all_nan = np.all(np.isnan(fm[mask, :, :]), axis=0)
        MWC[all_nan] = np.nan
        # ── Summary statistics ────────────────────────────────────────────────────
        cell_area_m2  = 12500 * 12500 # 12.5 km × 12.5 km in m²
        total_MWC_km3 = np.nansum(MWC * cell_area_m2) / 1e9
        mean_mwc.append(np.nanmean(MWC))
        print(f"Mean MWC:  {np.nanmean(MWC):.2f} m")
        print(f"Total MWC: {total_MWC_km3:.1f} km³")
        mean_mwc_km3.append(total_MWC_km3)
        
        # ── Plot ──────────────────────────────────────────────────────────────────
        lats     = ds_o['latitude']
        lons     = ds_o['longitude']
        proj     = ccrs.NorthPolarStereo()
        data_crs = ccrs.PlateCarree()
        
        fig = plt.figure(figsize=(28, 12))
        ax  = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([-180, 180, 50, 90], crs=data_crs)
        ax.coastlines(resolution="110m", linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        # ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)

        lats_sub = lats[(lats >= lat_s) & (lats <= lat_n)]
        lons_sub = lons[(lons >= lon_w) & (lons <= lon_e)]
        
        pcm1 = ax.pcolormesh(
            lons_sub, lats_sub, MWC,
            transform=data_crs,
            cmap=cmocean.cm.haline_r, 
            shading="auto",
            vmin=0,
            vmax=20,
        )
        
        ax.set_title(f"Meteoric Water Content (top 300 m) — {model_year}-{i+1:02d}")
        plt.colorbar(pcm1, ax=ax, orientation="horizontal", pad=0.05, shrink=0.5, fraction=0.02, label="MWC (m)")
        plt.tight_layout()
        plt.savefig(f"infer_output/MWC/MWC_{model_year}-{i+1:02d}_{lat_n}_{lat_s}_{lon_e}_{lon_w}.png")
        print("Saved.")
        plt.close(fig)

# Convert months to datetime if they are not already
months = pd.to_datetime(months)

# Create DataFrame
df = pd.DataFrame({
    "date": months,
    "mwc": mean_mwc_km3
})
df.to_parquet(f'infer_output/MWC/mwc_{start_year}_{end_year}_{lat_n}_{lat_s}_{lon_e}_{lon_w}.parquet')

print("the end!")