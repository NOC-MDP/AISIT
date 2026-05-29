import xarray as xr
import os
import matplotlib
import cmocean
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = "/gws/ssde/j25a/nemo/vol4/thopri/PINN_BARIUM/infer_output/"

st_year = 2000
end_year = 2025

for model_year in range(st_year,end_year+1):
    for model_month in range(1,13):
        # ── OPen NetCDF ───────────────────────────────────────────────────────────
        nc_out = os.path.join(out, f"tracer_predicted_{model_year}_{model_month:02d}.nc")
        ds_out = xr.open_dataset(nc_out)
        print(f"  Opened NetCDF: {nc_out}")
        
        # ── Surface map ───────────────────────────────────────────────────────────
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        # ── Surface map ───────────────────────────────────────────────────────────
        depth_dim = next(
            (d for d in ("depth", "z", "lev", "level", "deptht") if d in ds_out.dims), None
        )
        sel_kwargs = {depth_dim: 4} if depth_dim else {}
        
        # Extract surface fields
        da_pred = ds_out["tracer_pred"].isel(**sel_kwargs)
        da_unc = ds_out["tracer_uncertainty"].isel(**sel_kwargs)
        
        # Get coordinate names
        lat_name = [c for c in da_pred.coords if "lat" in c.lower()][0]
        lon_name = [c for c in da_pred.coords if "lon" in c.lower()][0]
        
        lats = da_pred[lat_name]
        lons = da_pred[lon_name]
        
        # Create figure with Arctic projection
        proj = ccrs.NorthPolarStereo()
        data_crs = ccrs.PlateCarree()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": proj})
        
        for ax in axes:
            ax.set_extent([-180, 180, 50, 90], crs=data_crs)  # Arctic view
            ax.coastlines(resolution="110m", linewidth=0.8)
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)
        
        # ── Plot predicted tracer ─────────────────────────────────────────────────
        pcm1 = axes[0].pcolormesh(
            lons,
            lats,
            da_pred,
            transform=data_crs,
            cmap=cmocean.cm.matter, 
            shading="auto",
            vmin=40,
            vmax=70,
        )
        axes[0].set_title(f"Predicted tracer — {model_year}-{model_month:02d} (10m depth)")
        plt.colorbar(pcm1, ax=axes[0], orientation="horizontal", pad=0.05)
        
        # ── Plot uncertainty ──────────────────────────────────────────────────────
        pcm2 = axes[1].pcolormesh(
            lons, lats, da_unc, transform=data_crs, cmap="Oranges", shading="auto"
        )
        axes[1].set_title("Prediction uncertainty σ (MC-Dropout, 10m depth)")
        plt.colorbar(pcm2, ax=axes[1], orientation="horizontal", pad=0.05)
        
        plt.tight_layout()
        
        map_path = os.path.join(out, f"model_10m_map_{model_year}_{model_month:02d}.png")
        plt.savefig(map_path, dpi=150)
        plt.close()
        
        print(f"  Saved: {map_path}")
        
        # ── Zonal-mean cross-section ───────────────────────────────────────────────
        if depth_dim and "latitude" in ds_out.coords:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            ds_out["tracer_pred"].where(ds_out.depth <= 300, drop=True).mean(
                "longitude"
            ).plot(
                ax=axes[0],
                cmap = cmocean.cm.matter,
                yincrease=False,
                vmin=40,
                vmax=70,
            )
            axes[0].set_title("Zonal-mean predicted tracer")
            ds_out["tracer_uncertainty"].where(ds_out.depth <= 500, drop=True).mean(
                "longitude"
            ).plot(
                ax=axes[1],
                cmap="Oranges",
                yincrease=False,
                shading="auto"
            )
            axes[1].set_title("Zonal-mean uncertainty σ")
            plt.tight_layout()
            xsec_path = os.path.join(
                out, f"model_zonal_mean_{model_year}_{model_month:02d}.png"
            )
            plt.savefig(xsec_path, dpi=150)
            plt.close()
            print(f"  Saved: {xsec_path}")