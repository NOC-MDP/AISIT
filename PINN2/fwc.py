import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmocean
import pandas as pd

def solve_mixing(d18O_obs, S_obs):
    """Solve the 3 end-member system for scalar or array inputs."""

        # --- End member definitions ---
    d18O_sw = 0.0       # Seawater
    d18O_fw = -21.0     # Freshwater (meteoric)
    d18O_si = 2.0       # Sea ice
    
    S_sw    = 34.8      # Seawater salinity
    S_fw    = 0.0       # Freshwater salinity
    S_si    = 4.0       # Sea ice salinity
    
    A = np.array([
        [1,       1,       1      ],
        [S_sw,    S_fw,    S_si   ],
        [d18O_sw, d18O_fw, d18O_si]
    ])
    
    b = np.stack([
        np.ones_like(d18O_obs),
        S_obs,
        d18O_obs
    ], axis=-1)                          # shape (..., 3)

    # np.linalg.solve broadcasts over leading dims with (..., 3, 3) / (..., 3)
    A_inv = np.linalg.inv(A)
    fractions = b @ A_inv.T              
    # shape (..., 3)
    return fractions

start_year = 1991
end_year = 2025
depth_inter = 350
    # buffer zone, Beaufort Gyre, Pan Arctic
lat_n = 90 #80.5 #90 
lat_s = 75 #70.5 #50
lon_e = -10 #-130 #180
lon_w = -130 #-170 #-180
model_years = [year for year in range(start_year, end_year + 1)]
# model_month = 10
mean_fwc = []
mean_fwc_km3 = []
months = []
for model_year in model_years:
    for i in range(12):
        months.append(f"{model_year}/{i+1:02d}")
        ds = xr.open_dataset(f"/gws/ssde/j25a/nemo/vol4/thopri/PINN/infer_output/tracer_predicted_{model_year}_{i+1:02d}.nc")
        d18O = ds["tracer_pred"]
        ds_s = xr.open_dataset(f"/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_thetao-so_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-01-01.nc")

        d18O_obs = d18O.sel(latitude=slice(lat_s,lat_n),longitude=slice(lon_w,lon_e)).values
        S_obs    = ds_s["so"].sel(latitude=slice(lat_s,lat_n),longitude=slice(lon_w,lon_e),
                              time=(ds_s.time.dt.year == model_year) & 
                                      (ds_s.time.dt.month == i+1)).squeeze("time").values
        
        fractions = solve_mixing(d18O_obs, S_obs)   # shape (..., 3)
        
        # sea ice = fractions[..., 2]
        # seawater fractions[..., 0]
        
        f_fw = fractions[..., 1] 
        f_fw = np.clip(f_fw, 0, 1)
        
        # Select only depths <= 300m
        depth_levels = np.array([
            0,2, 4, 6, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
            125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 600, 700,
            800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000
        ])
        
        mask = depth_levels <= depth_inter
        depth_300 = depth_levels[mask]              # depths 0–300m only
        f_fw_300  = f_fw[mask, :, :]        # slice depth axis of the 3D array
        # Fill NaNs with 0 so shallow columns still integrate over valid levels
        f_fw_300  = np.nan_to_num(f_fw_300, nan=0.0)
        
        FWC = np.trapezoid(f_fw_300, depth_300, axis=0)
        lats = ds['latitude']
        lons = ds['longitude']
        # Mask where ALL depth levels are NaN = true land/no-data
        all_nan = np.all(np.isnan(f_fw[mask, :, :]), axis=0)
        FWC[all_nan] = np.nan
        # ── Summary statistics ────────────────────────────────────────────────────
        cell_area_m2  = 12500 * 12500 # 12.5 km × 12.5 km in m²
        total_FWC_km3 = np.nansum(FWC * cell_area_m2) / 1e9
        mean_fwc.append(np.nanmean(FWC))
        print(f"Mean FWC:  {np.nanmean(FWC):.2f} m")
        print(f"Total FWC: {total_FWC_km3:.1f} km³")
        mean_fwc_km3.append(total_FWC_km3)
        
        # ── Plot ──────────────────────────────────────────────────────────────────
        lats     = ds['latitude']
        lons     = ds['longitude']
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
            lons_sub, lats_sub, FWC,
            transform=data_crs,
            cmap=cmocean.cm.haline_r,   # reversed so fresh = light, salty = dark,
            shading="auto",
            vmin=0,
            vmax=20,
        )
        
        ax.set_title(f"FWC (top 300 m) — {model_year}-{i+1:02d}")
        plt.colorbar(pcm1, ax=ax, orientation="horizontal", pad=0.05, shrink=0.5, fraction=0.02, label="FWC (m)")
        plt.tight_layout()
        plt.savefig(f"infer_output/FWC/FWC_{model_year}-{i+1:02d}_{lat_n}_{lat_s}_{lon_e}_{lon_w}.png")
        print("Saved.")
        plt.close(fig)

# Convert months to datetime if they are not already
months = pd.to_datetime(months)

# Create DataFrame
df = pd.DataFrame({
    "date": months,
    "fwc": mean_fwc_km3
})
df.to_parquet(f'infer_output/FWC/fwc_{start_year}_{end_year}_{lat_n}_{lat_s}_{lon_e}_{lon_w}.parquet')

print("the end!")