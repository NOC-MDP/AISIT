import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import cmocean
import matplotlib.pyplot as plt
import pandas as pd

start_year = 1991
end_year = 2025
depth_inter = 350
# Beaufort Gyre,buffer zone,  Pan Arctic
lat_n = 80.5  # 90 #90
lat_s = 70.5  # 75 #50
lon_e = -130  # -10 #180
lon_w = -170  # -130 #-180
model_years = [year for year in range(start_year, end_year + 1)]
# model_month = 10
mean_mwc = []
mean_mwc_km3 = []
mean_siwc = []
mean_siwc_km3 = []
mean_atlwc = []
mean_atlwc_km3 = []
mean_pacwc = []
mean_pacwc_km3 = []
months = []
ds_st = xr.open_dataset(
    f"/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_thetao-so_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-01-01.nc"
)

# depth coordinate
depth = ds_st.depth
# depth levels in meters
depth_levels = np.array(
    [
        0,
        2,
        4,
        6,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        125,
        150,
        175,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        600,
        700,
        800,
        900,
        1000,
        1200,
        1400,
        1600,
        1800,
        2000,
        2500,
        3000,
        3500,
        4000,
    ]
)
for model_year in model_years:
    for i in range(12):
        months.append(f"{model_year}/{i + 1:02d}")

        ds_o = xr.open_dataset(
            f"/gws/ssde/j25a/nemo/vol4/thopri/PINN/infer_output/tracer_predicted_{model_year}_{i + 1:02d}.nc"
        )
        d18O_obs = ds_o["tracer_pred"].sel(
            latitude=slice(lat_s, lat_n), longitude=slice(lon_w, lon_e)
        )
        ds_ba = xr.open_dataset(
            f"/gws/ssde/j25a/nemo/vol4/thopri/PINN_BARIUM/infer_output/tracer_predicted_{model_year}_{i + 1:02d}.nc"
        )
        dBa_obs = ds_o["tracer_pred"].sel(
            latitude=slice(lat_s, lat_n), longitude=slice(lon_w, lon_e)
        )
        S_obs = (
            ds_st["so"]
            .sel(
                latitude=slice(lat_s, lat_n),
                longitude=slice(lon_w, lon_e),
                time=(ds_st.time.dt.year == model_year)
                & (ds_st.time.dt.month == i + 1),
            )
            .squeeze("time")
        )

        # ============================================================
        # ENDMEMBERS https://pmc.ncbi.nlm.nih.gov/articles/PMC5852023
        # ============================================================

        def d18O_seaice_endmember(
            depth, z_transition=50, width=15, d18O_shallow=-2.0, d18O_deep=0.3
        ):
            """
            Smooth depth-varying sea ice melt d18O end-member.
            Sea ice melt influence is strongest near the surface and diminishes
            with depth as it mixes with ambient water masses.

            Sea ice is slightly enriched relative to its parent seawater due to
            fractionation during freezing (~+2‰), but the melt water end-member
            itself is typically close to local surface seawater or slightly
            depleted depending on the ice age and source.

            Parameters
            ----------
            depth       : float or array, depth in metres (positive downward)
            z_transition: float, centre of transition (m), default 75m
                          (sea ice melt is a shallow mixed-layer signal)
            width       : float, controls steepness of transition (m),
                          default 30m (sharper than ocean transition)
            d18O_shallow: float, end-member value near surface (‰), default -1.5
            d18O_deep   : float, end-member value at depth (‰), default -0.5
                          (converges toward ambient Atlantic/Pacific water)
            """
            # Sigmoid: 0 at surface (shallow end-member), 1 at depth (deep end-member)
            w = 1 / (1 + np.exp(-(depth - z_transition) / width))
            return d18O_shallow + w * (d18O_deep - d18O_shallow)

        # salinity endmembers
        S_pac = 32.7
        S_atl = 34.9
        S_meteoric = 0.0
        S_ice = 4.0

        # d180 endmembers
        d18O_ice = -2.0 #d18O_seaice_endmember(depth=depth)
        d18O_meteoric = -20.0
        d18O_atl = 0.3
        d18O_pac = -1.1

        # barium endmembers from https://pmc.ncbi.nlm.nih.gov/articles/PMC9285926/
        dBa_ice = 6.5
        dBa_meteoric = 130
        dBa_atl = 42
        dBa_pac = 56

        # ============================================================
        # SOLVE 4-ENDMEMBER MIXING SYSTEM
        # ============================================================
        # Unknowns:
        #
        #   fm    = meteoric fraction
        #   fi    = sea ice melt fraction
        #   f_atl = Atlantic water fraction
        #   f_pac = Pacific water fraction
        #
        # Equations:
        #
        #   fm + fi + f_atl + f_pac = 1                                       (sum)
        #   fm*Sm    + fi*Si    + f_atl*S_atl    + f_pac*S_pac    = S_obs     (salinity)
        #   fm*dm    + fi*di    + f_atl*d_atl    + f_pac*d_pac    = d18O_obs  (δ18O)
        #   fm*Bam   + fi*Bai   + f_atl*Ba_atl  + f_pac*Ba_pac   = Ba_obs    (barium)
        #
        # Eliminate f_pac = 1 - fm - fi - f_atl, subtract f_pac*X_pac
        # from each tracer row → 3×3 system in [fm, fi, f_atl]ᵀ
        
        # ── Convert xarray → numpy ────────────────────────────────────────────────
        S_obs_np     = S_obs.values        # (depth, lat, lon)
        d18O_obs_np  = d18O_obs.values     # (depth, lat, lon)
        dBa_obs_np   = dBa_obs.values      # (depth, lat, lon)
        
        shape = S_obs_np.shape             # (depth, lat, lon)
        
        # ── Build A (3×3 scalar matrix, unchanged) ────────────────────────────────
        A = np.array([
            [S_meteoric    - S_pac,    S_ice    - S_pac,    S_atl    - S_pac   ],
            [d18O_meteoric - d18O_pac, d18O_ice - d18O_pac, d18O_atl - d18O_pac],
            [dBa_meteoric  - dBa_pac,  dBa_ice  - dBa_pac,  dBa_atl  - dBa_pac ],
        ])
        
        # ── Build B as a proper (3, depth, lat, lon) numpy array ─────────────────
        B = np.stack([
            S_obs_np    - S_pac,
            d18O_obs_np - d18O_pac,
            dBa_obs_np  - dBa_pac,
        ], axis=0)                         # (3, depth*lat*lon) after reshape
        
        # ── Solve vectorised: flatten spatial dims, solve, reshape back ───────────
        B_flat = B.reshape(3, -1)          # (3, N)
        X_flat = np.linalg.solve(A, B_flat)  # (3, N)  — A broadcast over columns
        X      = X_flat.reshape(3, *shape) # (3, depth, lat, lon)
        
        fm    = X[0]
        fi    = X[1]
        f_atl = X[2]
        f_pac = 1.0 - fm - fi - f_atl

        # ==========================================================================
        # Data masking: Data below depth_inter and water columns shallower than 300m
        # ==========================================================================

        mask = depth_levels <= depth_inter
        depth_300 = depth_levels[mask]  # depths 0–300m only
        f_mw_300 = fm[mask, :, :]  # slice depth axis of the 3D array
        f_siw_300 = fi[mask, :, :]
        f_pacw_300 = f_pac[mask, :, :]
        f_atlw_300 = f_atl[mask, :, :]
        # find the deepest valid depth at each (lat, lon) point
        # fm is (depth, lat, lon) — NaN where no ocean data exists
        valid      = np.isfinite(fm)                        # True where data exists
        # broadcast depth_levels (depth,) across lat/lon, then mask invalid cells
        depth_3d   = np.where(valid, depth_levels[:, None, None], np.nan)
        bottom_depth = np.nanmax(depth_3d, axis=0)          # shape: (lat, lon)
        
        # Step 3: build 2D mask — True where seafloor >= 300m
        deep_enough = bottom_depth >= 300                   # shape: (lat, lon)
        
        # Step 4: apply to the already depth-sliced arrays
        # expand to (depth, lat, lon) to broadcast correctly
        f_mw_300  = np.where(deep_enough[None, :, :], f_mw_300,  np.nan)
        f_siw_300 = np.where(deep_enough[None, :, :], f_siw_300, np.nan)
        f_pacw_300  = np.where(deep_enough[None, :, :], f_pacw_300,  np.nan)
        f_atlw_300 = np.where(deep_enough[None, :, :], f_atlw_300, np.nan)
        
        # Fill NaNs with 0 so shallow columns still integrate over valid levels
        f_mw_300 = np.nan_to_num(f_mw_300, nan=0.0)
        f_siw_300 = np.nan_to_num(f_siw_300, nan=0.0)
        f_pacw_300 = np.nan_to_num(f_pacw_300, nan=0.0)
        f_atlw_300 = np.nan_to_num(f_atlw_300, nan=0.0)

        # ============================================================
        # Intergrate to calculate water content
        # ============================================================
        
        MWC = np.trapezoid(f_mw_300, depth_300, axis=0)
        SIWC = np.trapezoid(f_siw_300, depth_300, axis=0)
        ATLWC = np.trapezoid(f_atlw_300, depth_300, axis=0)
        PACWC = np.trapezoid(f_pacw_300, depth_300, axis=0)

        # Mask where ALL depth levels are NaN = true land/no-data
        all_nan_fm = np.all(np.isnan(fm[mask, :, :]), axis=0)
        MWC[all_nan_fm] = np.nan
        all_nan_fi = np.all(np.isnan(fi[mask, :, :]), axis=0)
        SIWC[all_nan_fi] = np.nan
        all_nan_fatl = np.all(np.isnan(f_atl[mask, :, :]), axis=0)
        ATLWC[all_nan_fatl] = np.nan
        all_nan_fpac = np.all(np.isnan(f_pac[mask, :, :]), axis=0)
        PACWC[all_nan_fpac] = np.nan
        
        # ── Summary statistics ────────────────────────────────────────────────────
        cell_area_m2 = 12500 * 12500  # 12.5 km × 12.5 km in m²
        total_MWC_km3 = np.nansum(MWC * cell_area_m2) / 1e9
        mean_mwc.append(np.nanmean(MWC))
        print(f"Mean MWC:  {np.nanmean(MWC):.2f} m")
        print(f"Total MWC: {total_MWC_km3:.1f} km³")
        mean_mwc_km3.append(total_MWC_km3)

        total_SIWC_km3 = np.nansum(SIWC * cell_area_m2) / 1e9
        mean_siwc.append(np.nanmean(SIWC))
        print(f"Mean SIWC:  {np.nanmean(SIWC):.2f} m")
        print(f"Total SIWC: {total_SIWC_km3:.1f} km³")
        mean_siwc_km3.append(total_SIWC_km3)

        total_ATLWC_km3 = np.nansum(ATLWC * cell_area_m2) / 1e9
        mean_atlwc.append(np.nanmean(ATLWC))
        print(f"Mean ATLWC:  {np.nanmean(ATLWC):.2f} m")
        print(f"Total ATLWC: {total_ATLWC_km3:.1f} km³")
        mean_atlwc_km3.append(total_ATLWC_km3)

        total_PACWC_km3 = np.nansum(PACWC * cell_area_m2) / 1e9
        mean_pacwc.append(np.nanmean(PACWC))
        print(f"Mean PACWC:  {np.nanmean(PACWC):.2f} m")
        print(f"Total PACWC: {total_PACWC_km3:.1f} km³")
        mean_pacwc_km3.append(total_PACWC_km3)

        # # ── Plot ──────────────────────────────────────────────────────────────────
        # lats     = ds_o['latitude']
        # lons     = ds_o['longitude']
        # proj     = ccrs.NorthPolarStereo()
        # data_crs = ccrs.PlateCarree()

        # fig = plt.figure(figsize=(28, 12))
        # ax  = fig.add_subplot(1, 1, 1, projection=proj)
        # ax.set_extent([-180, 180, 50, 90], crs=data_crs)
        # ax.coastlines(resolution="110m", linewidth=0.8)
        # ax.add_feature(cfeature.LAND, facecolor="lightgray")
        # # ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)

        # lats_sub = lats[(lats >= lat_s) & (lats <= lat_n)]
        # lons_sub = lons[(lons >= lon_w) & (lons <= lon_e)]

        # pcm1 = ax.pcolormesh(
        #     lons_sub, lats_sub, MWC,
        #     transform=data_crs,
        #     cmap=cmocean.cm.haline_r,
        #     shading="auto",
        #     vmin=0,
        #     vmax=20,
        # )

        # ax.set_title(f"Meteoric Water Content (top 300 m) — {model_year}-{i+1:02d}")
        # plt.colorbar(pcm1, ax=ax, orientation="horizontal", pad=0.05, shrink=0.5, fraction=0.02, label="MWC (m)")
        # plt.tight_layout()
        # plt.savefig(f"infer_output/MWC/MWC_{model_year}-{i+1:02d}_{lat_n}_{lat_s}_{lon_e}_{lon_w}.png")
        # print("Saved.")
        # plt.close(fig)

# Convert months to datetime if they are not already
months = pd.to_datetime(months)

# Create DataFrame
df = pd.DataFrame({"date": months, "mwc": mean_mwc_km3, "siwc": mean_siwc_km3,"pacwc": mean_pacwc_km3,"atlwc": mean_atlwc_km3})
df.to_parquet(
    f"infer_output/MWC/mwc_siwc_atlwc_pacwc_{start_year}_{end_year}_{lat_n}_{lat_s}_{lon_e}_{lon_w}.parquet"
)

print("the end!")
