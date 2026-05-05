"""
infer.py — PINN Conservative Tracer Inference (pre-trained model)
=================================================================
Run inference with an already-trained PINN — no retraining required.

Modes
-----
1. Ocean model inference  (--model)
   Load a NetCDF ocean model field and produce a predicted-tracer NetCDF.

2. Observation diagnostics  (--obs)
   Score the model against a CSV of labelled observations (R², RMSE,
   uncertainty, depth residuals, T-S diagram).

3. Both  (--model --obs)
   Run diagnostics first, then ocean-model inference.

At least one of --model or --obs must be supplied.

Usage examples
--------------
# Ocean model only
python infer.py \\
    --checkpoint ./output/pinn_tracer.pt \\
    --model      ocean_model.nc \\
    --model_year 2021 --model_month 8 \\
    --output_dir ./infer_output

# Observation diagnostics only
python infer.py \\
    --checkpoint ./output/pinn_tracer.pt \\
    --obs        new_obs.csv \\
    --tracer_col oxygen \\
    --date_col   date --date_format "%%d/%%m/%%Y" \\
    --output_dir ./infer_output

# Both
python infer.py \\
    --checkpoint ./output/pinn_tracer.pt \\
    --obs        new_obs.csv --tracer_col oxygen \\
    --model      ocean_model.nc \\
    --output_dir ./infer_output
"""

import argparse
import os
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd
import torch
import cmocean
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

from pinn_tracer import (
    TracerPINN,
    build_features,
    infer_on_model_field,
    plot_depth_residuals,
    plot_ts_diagram,
    predict_with_uncertainty,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference with a pre-trained PINN tracer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to saved .pt checkpoint (from main.py or prior run)",
    )

    # ── Inference targets (at least one required) ─────────────────────────────
    grp = p.add_argument_group("Inference targets (supply at least one)")
    grp.add_argument(
        "--model", default=None, help="NetCDF ocean model file for spatial inference"
    )
    grp.add_argument(
        "--obs",
        default=None,
        help="CSV of labelled observations for diagnostics/scoring",
    )

    # ── Ocean model options ────────────────────────────────────────────────────
    og = p.add_argument_group("Ocean model options (used with --model)")
    og.add_argument(
        "--model_year",
        type=int,
        default=2020,
        help="Target year for model-field inference",
    )
    og.add_argument(
        "--model_month",
        type=int,
        default=6,
        help="Target month for model-field inference (1-12)",
    )

    # ── Observation / diagnostics options ─────────────────────────────────────
    dg = p.add_argument_group("Observation options (used with --obs)")
    dg.add_argument(
        "--tracer_col",
        default="tracer",
        help="Name of the tracer column in the obs CSV",
    )
    dg.add_argument(
        "--date_col",
        default=None,
        help="Combined date column name (if not separate year/month)",
    )
    dg.add_argument(
        "--date_format",
        default=None,
        help="strptime format for --date_col, e.g. '%%d/%%m/%%Y'",
    )

    # ── Inference hyperparameters ──────────────────────────────────────────────
    ig = p.add_argument_group("Inference hyperparameters")
    ig.add_argument(
        "--mc_samples",
        type=int,
        default=100,
        help="MC-Dropout samples for uncertainty estimation",
    )
    ig.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size for forward pass (larger = faster on GPU)",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", default="./infer_output")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_checkpoint(path, device):
    """
    Load a .pt checkpoint saved by main.py and reconstruct the model + scaler.

    Checkpoint schema (written by main.py):
        model_state   : nn.Module state_dict
        feat_names    : list[str]
        hidden_dim    : int
        n_blocks      : int
        n_features    : int
        scaler_mean   : np.ndarray
        scaler_scale  : np.ndarray
    """
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] Checkpoint not found: {path}")

    ckpt = torch.load(path, weights_only=False, map_location=device)

    required_keys = {
        "model_state",
        "feat_names",
        "hidden_dim",
        "n_blocks",
        "n_features",
        "scaler_mean",
        "scaler_scale",
    }
    missing = required_keys - set(ckpt.keys())
    if missing:
        sys.exit(
            f"[ERROR] Checkpoint is missing keys: {missing}\n"
            f"        Make sure you are loading a checkpoint saved by main.py."
        )

    # Reconstruct model
    model = TracerPINN(
        n_features=ckpt["n_features"],
        hidden_dim=ckpt["hidden_dim"],
        n_blocks=ckpt["n_blocks"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Reconstruct scaler (sklearn StandardScaler shell, no re-fitting needed)
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.n_features_in_ = ckpt["n_features"]

    feat_names = ckpt["feat_names"]

    print(f"  Checkpoint   : {path}")
    print(
        f"  Architecture : hidden_dim={ckpt['hidden_dim']}, "
        f"n_blocks={ckpt['n_blocks']}, n_features={ckpt['n_features']}"
    )
    print(f"  Features     : {feat_names}")
    return model, scaler, feat_names


# ═══════════════════════════════════════════════════════════════════════════════
# DATE PARSING  (identical helper to main.py — kept local to avoid import dep)
# ═══════════════════════════════════════════════════════════════════════════════


def parse_date_column(df, date_col, date_format=None):
    if date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' not found. Available columns: {list(df.columns)}"
        )
    raw = df[date_col].astype(str)
    if date_format:
        parsed = pd.to_datetime(raw, format=date_format, errors="coerce")
        fmt_used = date_format
    else:
        parsed = pd.to_datetime(raw, dayfirst=True, errors="coerce")
        fmt_used = "auto-detected (dayfirst=True)"

    n_failed = parsed.isna().sum()
    if n_failed > 0:
        bad = raw[parsed.isna()].unique()[:5].tolist()
        raise ValueError(
            f"Could not parse {n_failed:,} dates in '{date_col}' "
            f"using format '{fmt_used}'. Examples: {bad}\n"
            f"Pass --date_format explicitly, e.g. '%d/%m/%Y'."
        )

    df = df.copy()
    df["year"] = parsed.dt.year.astype(int)
    df["month"] = parsed.dt.month.astype(int)
    n_show = min(4, len(df))
    sample = pd.DataFrame(
        {
            "raw": raw.iloc[:n_show].values,
            "parsed": parsed.iloc[:n_show].dt.strftime("%Y-%m-%d").values,
        }
    )
    print(f"  Date format : {fmt_used}")
    print(f"  Parse sample (verify day/month order):")
    print(sample.to_string(index=False))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVATION LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_observations(path, tracer_col, date_col=None, date_format=None):
    df = pd.read_csv(path)

    if date_col is not None:
        df = parse_date_column(df, date_col, date_format)
    else:
        for col in ("year", "month"):
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' missing. Pass --date_col or include "
                    "separate 'year' and 'month' columns."
                )

    required = {"temp", "salinity", "depth", "lon", "lat", "year", "month"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] CSV missing required columns: {missing}")

    if tracer_col not in df.columns:
        sys.exit(
            f"[ERROR] Tracer column '{tracer_col}' not found in CSV.\n"
            f"        Available: {list(df.columns)}\n"
            f"        Pass --tracer_col <name> to specify."
        )
    if tracer_col != "tracer":
        df = df.rename(columns={tracer_col: "tracer"})

    before = len(df)
    df = df.dropna(subset=["temp", "salinity", "depth", "lon", "lat", "tracer"])
    print(f"  Loaded {before:,} rows → {len(df):,} after dropping NaNs")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS ON OBSERVATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def run_obs_diagnostics(df, model, scaler, feat_names, mc_samples, device, out):
    """Score model on labelled observations; save plots + predictions CSV."""
    print("\n── Observation diagnostics ─────────────────────────────────")

    X, y, _feat_names, valid_mask = build_features(df)

    # Guard: feature list must match the checkpoint
    if _feat_names != feat_names:
        sys.exit(
            f"[ERROR] Feature mismatch between checkpoint and observations.\n"
            f"  Checkpoint : {feat_names}\n"
            f"  Computed   : {_feat_names}\n"
            f"  Make sure the obs CSV was prepared with the same TEOS-10 "
            f"pipeline used during training."
        )

    df_valid = df.iloc[valid_mask].reset_index(drop=True)
    X_sc = scaler.transform(X)

    y_pred_mn, y_pred_sd = predict_with_uncertainty(
        model,
        X_sc,
        n_samples=mc_samples,
        device=device,
    )

    # ── Scalar metrics ────────────────────────────────────────────────────────
    residuals = y_pred_mn - y
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(residuals))
    mean_uncert = y_pred_sd.mean()

    print(f"  N obs          : {len(y):,}")
    print(f"  RMSE           : {rmse:.4f}")
    print(f"  R²             : {r2:.4f}")
    print(f"  MAE            : {mae:.4f}")
    print(f"  Mean σ (MC)    : {mean_uncert:.4f}")

    # ── Save predictions CSV ──────────────────────────────────────────────────
    df_out = df_valid.copy()
    df_out["tracer_pred"] = y_pred_mn
    df_out["tracer_uncert"] = y_pred_sd
    df_out["residual"] = residuals
    pred_path = os.path.join(out, "obs_predictions.csv")
    df_out.to_csv(pred_path, index=False)
    print(f"  Saved: {pred_path}")

    # ── Metrics summary CSV ───────────────────────────────────────────────────
    metrics = pd.DataFrame(
        [
            {
                "n_obs": len(y),
                "rmse": rmse,
                "r2": r2,
                "mae": mae,
                "mean_uncertainty": mean_uncert,
            }
        ]
    )
    metrics_path = os.path.join(out, "obs_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"  Saved: {metrics_path}")

    # ── Scatter: observed vs predicted ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    lims = [min(y.min(), y_pred_mn.min()), max(y.max(), y_pred_mn.max())]
    ax.errorbar(
        y, y_pred_mn, yerr=y_pred_sd, fmt="none", ecolor="steelblue", alpha=0.15, lw=0.6
    )
    ax.scatter(y, y_pred_mn, s=5, alpha=0.4, c="steelblue", zorder=3)
    ax.plot(lims, lims, "r--", lw=1, label="1:1")
    ax.set(
        xlabel="Observed tracer",
        ylabel="Predicted tracer",
        title=f"Obs vs Pred  R²={r2:.3f}  RMSE={rmse:.4f}",
        xlim=lims,
        ylim=lims,
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    scatter_path = os.path.join(out, "obs_vs_pred_scatter.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"  Saved: {scatter_path}")

    # ── Residual histogram ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(residuals, bins=60, color="steelblue", edgecolor="white", lw=0.3)
    ax.axvline(0, color="r", ls="--")
    ax.set(
        xlabel="Residual (pred − obs)", ylabel="Count", title="Residual distribution"
    )
    plt.tight_layout()
    hist_path = os.path.join(out, "obs_residual_histogram.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"  Saved: {hist_path}")

    # ── Depth residual diagnostics (from pinn_tracer) ─────────────────────────
    plot_depth_residuals(y, y_pred_mn, df_valid["depth"].values)
    res_path = os.path.join(out, "obs_depth_residuals.png")
    if os.path.exists("residual_diagnostics.png"):
        os.replace("residual_diagnostics.png", res_path)
    print(f"  Saved: {res_path}")

    # ── T-S diagram ───────────────────────────────────────────────────────────
    plot_ts_diagram(df_valid, feat_names, y_pred_mn)
    ts_path = os.path.join(out, "obs_ts_diagram.png")
    if os.path.exists("ts_diagram_check.png"):
        os.replace("ts_diagram_check.png", ts_path)
    print(f"  Saved: {ts_path}")

    # ── Uncertainty vs depth ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        y_pred_sd,
        df_valid["depth"].values,
        c=np.abs(residuals),
        cmap="hot_r",
        s=4,
        alpha=0.5,
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="Prediction σ (MC-Dropout)",
        ylabel="Depth (m)",
        title="Uncertainty vs depth  (colour = |residual|)",
    )
    plt.colorbar(sc, ax=ax, label="|residual|")
    plt.tight_layout()
    unc_path = os.path.join(out, "obs_uncertainty_vs_depth.png")
    plt.savefig(unc_path, dpi=150)
    plt.close()
    print(f"  Saved: {unc_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# OCEAN MODEL INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════


def run_model_inference(
    nc_path, model, scaler, feat_names, model_year, model_month, mc_samples, device, out
):
    """Infer tracer field on a NetCDF ocean model and save results."""
    try:
        import xarray as xr
    except ImportError:
        sys.exit(
            "[ERROR] xarray is required for ocean model inference. "
            "Install with: pip install xarray netcdf4"
        )

    print(f"\n── Ocean model inference ────────────────────────────────────")
    print(f"  File   : {nc_path}")
    print(f"  Target : {model_year}-{model_month:02d}")

    if not os.path.isfile(nc_path):
        sys.exit(f"[ERROR] NetCDF file not found: {nc_path}")

    ds = xr.open_dataset(nc_path)

    # Subset to the target time step before any heavy operations
    time_str = f"{model_year}-{model_month:02d}-01"
    try:
        ds = ds.sel(time=time_str, method="nearest")
        print(
            f"  Nearest time slice selected: "
            f"{str(ds['time'].values)[:10] if 'time' in ds else 'N/A'}"
        )
    except (KeyError, ValueError):
        print("  [WARN] Could not subset by time — using full dataset as-is.")

    ds_out = infer_on_model_field(
        ds,
        model,
        scaler,
        feat_names,
        target_year=model_year,
        target_month=model_month,
        device=device,
    )

    # ── Save NetCDF ───────────────────────────────────────────────────────────
    nc_out = os.path.join(out, f"tracer_predicted_{model_year}_{model_month:02d}.nc")
    ds_out.to_netcdf(nc_out)
    print(f"  Saved NetCDF: {nc_out}")

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
        cmap=cmocean.cm.haline,   # reversed so fresh = light, salty = dark,
        shading="auto",
        vmin=-4,
        vmax=0.5,
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
            cmap = cmocean.cm.haline,   # reversed so fresh = light, salty = dark
            yincrease=False,
            vmin=-4,
            vmax=0.5,
        )
        axes[0].set_title("Zonal-mean predicted tracer")
        ds_out["tracer_uncertainty"].where(ds_out.depth <= 500, drop=True).mean(
            "longitude"
        ).plot(
            ax=axes[1],
            cmap="Oranges",
            yincrease=False,
            vmin=-4,
            vmax=0.5,
        )
        axes[1].set_title("Zonal-mean uncertainty σ")
        plt.tight_layout()
        xsec_path = os.path.join(
            out, f"model_zonal_mean_{model_year}_{model_month:02d}.png"
        )
        plt.savefig(xsec_path, dpi=150)
        plt.close()
        print(f"  Saved: {xsec_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    args = parse_args()

    if args.model is None and args.obs is None:
        sys.exit(
            "[ERROR] Nothing to do — supply at least one of:\n"
            "  --model  <ocean_model.nc>\n"
            "  --obs    <observations.csv>"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(" PINN Tracer Inference — pre-trained model")
    print(f"{'=' * 60}")
    print(f"  Device       : {device}")
    print(f"  Checkpoint   : {args.checkpoint}")
    if args.obs:
        print(f"  Obs CSV      : {args.obs}")
    if args.model:
        print(f"  Ocean model  : {args.model}")
        print(f"  Target time  : {args.model_year}-{args.model_month:02d}")
    print(f"  MC samples   : {args.mc_samples}")
    print(f"  Output dir   : {args.output_dir}")
    print()

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print("── Loading checkpoint ──────────────────────────────────────")
    model, scaler, feat_names = load_checkpoint(args.checkpoint, device)

    # ── Mode A: observation diagnostics ───────────────────────────────────────
    if args.obs is not None:
        print("\n── Loading observations ────────────────────────────────────")
        df = load_observations(
            args.obs,
            args.tracer_col,
            date_col=args.date_col,
            date_format=args.date_format,
        )
        run_obs_diagnostics(
            df,
            model,
            scaler,
            feat_names,
            mc_samples=args.mc_samples,
            device=device,
            out=args.output_dir,
        )

    # ── Mode B: ocean model inference ─────────────────────────────────────────
    if args.model is not None:
        run_model_inference(
            args.model,
            model,
            scaler,
            feat_names,
            model_year=args.model_year,
            model_month=args.model_month,
            mc_samples=args.mc_samples,
            device=device,
            out=args.output_dir,
        )

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  All outputs written to: {args.output_dir}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
