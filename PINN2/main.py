"""
main.py — PINN Conservative Tracer Inference Pipeline
======================================================
Runs the full pipeline:
  1. Load observations
  2. Engineer physics-based features (TEOS-10)
  3. Temporal cross-validation
  4. Train final PINN on all data
  5. Diagnostics & plots
  6. (Optional) Inference on ocean model NetCDF

Usage:
    python main.py --obs observations.csv [--model ocean_model.nc]
                   [--tracer_col oxygen] [--epochs 300] [--output_dir ./output]

Date handling:
    If your CSV has separate year/month columns, do nothing.
    If your CSV has a single date column (e.g. "09/06/2016"), pass:
        --date_col date
        --date_format "%d/%m/%Y"   (recommended — avoids day/month ambiguity)
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for servers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ── Local imports from the PINN module ────────────────────────────────────────
from pinn_tracer import (
    build_features,
    TracerPINN,
    train_pinn,
    temporal_cv_splits,
    predict_with_uncertainty,
    infer_on_model_field,
    plot_ts_diagram,
    plot_depth_residuals,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="PINN conservative tracer inference")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--obs", required=True,
                   help="CSV of observations. Required columns: "
                        "temp, salinity, depth, lon, lat, <tracer>, "
                        "and either: (a) separate 'year' and 'month' columns, "
                        "or (b) a single date column specified via --date_col")
    p.add_argument("--tracer_col", default="tracer",
                   help="Name of the tracer column in the CSV (default: 'tracer')")
    p.add_argument("--date_col", default=None,
                   help="Name of a combined date column, e.g. 'date'. "
                        "Use instead of separate year/month columns.")
    p.add_argument("--date_format", default=None,
                   help="strptime format string for --date_col. "
                        "e.g. '%%d/%%m/%%Y' for 09/06/2016. "
                        "Strongly recommended to avoid day/month ambiguity. "
                        "If omitted, pandas infers the format (dayfirst=True).")

    # ── Ocean model inference (optional) ──────────────────────────────────────
    p.add_argument("--model", default=None,
                   help="(Optional) NetCDF ocean model file for inference")
    p.add_argument("--model_year",  type=int, default=2020,
                   help="Target year for model inference (default: 2020)")
    p.add_argument("--model_month", type=int, default=6,
                   help="Target month for model inference (default: 6)")

    # ── Training hyperparameters ───────────────────────────────────────────────
    p.add_argument("--epochs",        type=int,   default=300)
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--hidden_dim",    type=int,   default=256)
    p.add_argument("--n_blocks",      type=int,   default=4)
    p.add_argument("--cv_folds",      type=int,   default=5,
                   help="Number of temporal CV folds (default: 5)")
    p.add_argument("--lambda_diap",   type=float, default=0.05)
    p.add_argument("--lambda_smooth", type=float, default=0.02)
    p.add_argument("--lambda_sec",    type=float, default=0.01)
    p.add_argument("--mc_samples",    type=int,   default=100,
                   help="MC-Dropout samples for uncertainty (default: 100)")

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# DATE PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_date_column(df, date_col, date_format=None):
    """
    Parse a combined date column into separate integer 'year' and 'month'
    columns.

    Supported out of the box (if date_format is None):
        09/06/2016   ->  year=2016, month=6  (DD/MM/YYYY, dayfirst=True)
        2016-06-09   ->  year=2016, month=6  (ISO 8601)
        09-Jun-2016  ->  year=2016, month=6
        June 9 2016  ->  year=2016, month=6

    WARNING: DD/MM/YYYY and MM/DD/YYYY are ambiguous when day <= 12.
    Always pass --date_format "%d/%m/%Y" for European-format dates to
    guarantee correct parsing.
    """
    if date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    raw = df[date_col].astype(str)

    if date_format:
        parsed   = pd.to_datetime(raw, format=date_format, errors="coerce")
        fmt_used = date_format
    else:
        # dayfirst=True: DD/MM/YYYY takes precedence for ambiguous dates,
        # which is standard in most oceanographic datasets (Argo, WOD, BODC).
        parsed   = pd.to_datetime(raw, dayfirst=True, errors="coerce")
        fmt_used = "auto-detected (dayfirst=True)"

    n_failed = parsed.isna().sum()
    if n_failed > 0:
        bad_examples = raw[parsed.isna()].unique()[:5].tolist()
        raise ValueError(
            f"\n  Could not parse {n_failed:,} dates in column '{date_col}' "
            f"using format '{fmt_used}'.\n"
            f"  Example unparseable values: {bad_examples}\n"
            f"  Fix: pass --date_format explicitly, e.g.:\n"
            f"    --date_format '%d/%m/%Y'   for  09/06/2016\n"
            f"    --date_format '%m/%d/%Y'   for  06/09/2016\n"
            f"    --date_format '%Y-%m-%d'   for  2016-06-09\n"
            f"    --date_format '%d-%b-%Y'   for  09-Jun-2016"
        )

    df = df.copy()
    df["year"]  = parsed.dt.year.astype(int)
    df["month"] = parsed.dt.month.astype(int)

    # Print a parse sample so the user can verify day/month order is correct
    n_show = min(4, len(df))
    sample = pd.DataFrame({
        "raw"   : raw.iloc[:n_show].values,
        "parsed": parsed.iloc[:n_show].dt.strftime("%Y-%m-%d").values,
        "year"  : df["year"].iloc[:n_show].values,
        "month" : df["month"].iloc[:n_show].values,
    })
    print(f"  Date format used : {fmt_used}")
    print(f"  Parse sample — verify month/day order is correct:")
    print(sample.to_string(index=False))
    print()

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_observations(path, tracer_col, date_col=None, date_format=None):
    """
    Load observation CSV.
    - If date_col is given, parse it into year/month columns.
    - Otherwise expect explicit 'year' and 'month' columns.
    - Rename tracer_col -> 'tracer'.
    - Drop rows with NaNs in key columns.
    """
    df = pd.read_csv(path)

    # ── Date parsing ──────────────────────────────────────────────────────────
    if date_col is not None:
        df = parse_date_column(df, date_col, date_format)
    else:
        for col in ("year", "month"):
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in CSV. "
                    f"Either include separate 'year' and 'month' columns, "
                    f"or pass --date_col <col_name> to parse a combined date."
                )

    # ── Required column check ─────────────────────────────────────────────────
    required = {"temp", "salinity", "depth", "lon", "lat", "year", "month"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if tracer_col not in df.columns:
        raise ValueError(
            f"Tracer column '{tracer_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if tracer_col != "tracer":
        df = df.rename(columns={tracer_col: "tracer"})

    # ── Drop NaNs ─────────────────────────────────────────────────────────────
    before = len(df)
    df     = df.dropna(subset=["temp", "salinity", "depth",
                                "lon", "lat", "tracer"])
    print(f"  Loaded {before:,} rows, {len(df):,} after dropping NaNs")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarise_dataset(df):
    print("\n── Dataset summary ────────────────────────────────────────")
    print(f"  Observations : {len(df):,}")
    print(f"  Year range   : {int(df['year'].min())} - {int(df['year'].max())}")
    print(f"  Depth range  : {df['depth'].min():.1f} - {df['depth'].max():.1f} m")
    print(f"  Tracer range : {df['tracer'].min():.3f} - {df['tracer'].max():.3f}")
    print(f"  Lat range    : {df['lat'].min():.1f} - {df['lat'].max():.1f} deg")
    print(f"  Lon range    : {df['lon'].min():.1f} - {df['lon'].max():.1f} deg")
    decade = df.groupby((df["year"] // 10) * 10).size()
    print("\n  Obs per decade:")
    for dec, n in decade.items():
        bar = "█" * (n // max(1, len(df) // 200))
        print(f"    {int(dec)}s  {n:>6,}  {bar}")
    print()


def save_history_plot(history, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["epoch"], history["data_loss"], label="Data")
    axes[0].plot(history["epoch"], history["phys_loss"], label="Physics")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training losses")
    axes[0].legend()
    axes[0].set_yscale("log")

    axes[1].plot(history["epoch"], history["val_rmse"])
    axes[1].set(xlabel="Epoch", ylabel="RMSE", title="Validation RMSE")

    axes[2].plot(history["epoch"], history["val_r2"])
    axes[2].axhline(0, color="r", ls="--")
    axes[2].set(xlabel="Epoch", ylabel="R2", title="Validation R2")

    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_cv_results(cv_results, output_dir):
    df_cv = pd.DataFrame(cv_results)
    path  = os.path.join(output_dir, "cv_results.csv")
    df_cv.to_csv(path, index=False)
    print(f"  Saved: {path}")

    print("\n── Temporal CV results ────────────────────────────────────")
    for row in cv_results:
        print(f"  Fold {row['fold']}  "
              f"train {int(row['train_year_min'])}-{int(row['train_year_max'])}  "
              f"val {int(row['val_year_min'])}-{int(row['val_year_max'])}  "
              f"R2={row['val_r2']:.3f}  RMSE={row['val_rmse']:.4f}  "
              f"N_val={int(row['n_val'])}")
    r2s = [r["val_r2"] for r in cv_results]
    print(f"\n  Mean R2: {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(" PINN Conservative Tracer Inference")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Obs file    : {args.obs}")
    print(f"  Tracer col  : {args.tracer_col}")
    if args.date_col:
        fmt = args.date_format if args.date_format else "auto"
        print(f"  Date col    : {args.date_col}  (format: {fmt})")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Output dir  : {args.output_dir}")
    print()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("── 1. Loading observations ─────────────────────────────────")
    df = load_observations(
        args.obs, args.tracer_col,
        date_col    = args.date_col,
        date_format = args.date_format,
    )
    summarise_dataset(df)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("── 2. Engineering features (TEOS-10) ───────────────────────")
    X, y, feat_names, valid_mask = build_features(df)
    df_valid = df.iloc[valid_mask].reset_index(drop=True)
    years    = df_valid["year"].values
    print(f"  Features : {feat_names}")
    print(f"  Valid obs: {len(y):,}")

    # ── 3. Temporal cross-validation ──────────────────────────────────────────
    print(f"\n── 3. Temporal CV ({args.cv_folds} folds) ───────────────────────────")
    splits     = temporal_cv_splits(years, n_splits=args.cv_folds)
    cv_results = []

    for fold, (tr_idx, val_idx) in enumerate(splits):
        print(f"\n  [Fold {fold+1}/{len(splits)}]  "
              f"train n={len(tr_idx):,}  val n={len(val_idx):,}")

        scaler_cv = StandardScaler().fit(X[tr_idx])
        model_cv  = TracerPINN(
            n_features = len(feat_names),
            hidden_dim = args.hidden_dim,
            n_blocks   = args.n_blocks,
        ).to(device)

        model_cv, hist_cv = train_pinn(
            model_cv,
            X[tr_idx],  y[tr_idx],
            X[val_idx], y[val_idx],
            feat_names, scaler_cv,
            n_epochs      = args.epochs,
            batch_size    = args.batch_size,
            lr            = args.lr,
            lambda_diap   = args.lambda_diap,
            lambda_smooth = args.lambda_smooth,
            lambda_sec    = args.lambda_sec,
            device        = device,
        )

        cv_results.append({
            "fold"          : fold + 1,
            "train_year_min": years[tr_idx].min(),
            "train_year_max": years[tr_idx].max(),
            "val_year_min"  : years[val_idx].min(),
            "val_year_max"  : years[val_idx].max(),
            "n_train"       : len(tr_idx),
            "n_val"         : len(val_idx),
            "val_r2"        : hist_cv["val_r2"].iloc[-1],
            "val_rmse"      : hist_cv["val_rmse"].iloc[-1],
        })

    save_cv_results(cv_results, args.output_dir)

    # ── 4. Final model (train on all data) ────────────────────────────────────
    print("── 4. Training final model on all data ─────────────────────")
    scaler      = StandardScaler().fit(X)
    model_final = TracerPINN(
        n_features = len(feat_names),
        hidden_dim = args.hidden_dim,
        n_blocks   = args.n_blocks,
    ).to(device)

    model_final, history = train_pinn(
        model_final,
        X, y, X, y,          # val == train for final loss monitoring only
        feat_names, scaler,
        n_epochs      = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        lambda_diap   = args.lambda_diap,
        lambda_smooth = args.lambda_smooth,
        lambda_sec    = args.lambda_sec,
        device        = device,
    )
    save_history_plot(history, args.output_dir)

    # Save model weights + scaler stats for future inference
    model_path = os.path.join(args.output_dir, "pinn_tracer.pt")
    torch.save({
        "model_state" : model_final.state_dict(),
        "feat_names"  : feat_names,
        "hidden_dim"  : args.hidden_dim,
        "n_blocks"    : args.n_blocks,
        "n_features"  : len(feat_names),
        "scaler_mean" : scaler.mean_,
        "scaler_scale": scaler.scale_,
    }, model_path)
    print(f"  Saved model: {model_path}")

    # ── 5. In-sample predictions + diagnostics ────────────────────────────────
    print("\n── 5. Diagnostics ──────────────────────────────────────────")
    X_sc                 = scaler.transform(X)
    y_pred_mn, y_pred_sd = predict_with_uncertainty(
        model_final, X_sc,
        n_samples = args.mc_samples,
        device    = device,
    )

    # Save predictions alongside observations
    df_pred = df_valid.copy()
    df_pred["tracer_pred"]   = y_pred_mn
    df_pred["tracer_uncert"] = y_pred_sd
    df_pred["residual"]      = y_pred_mn - y
    pred_path = os.path.join(args.output_dir, "predictions.csv")
    df_pred.to_csv(pred_path, index=False)
    print(f"  Saved: {pred_path}")

    overall_rmse = np.sqrt(np.mean((y_pred_mn - y) ** 2))
    overall_r2   = 1 - np.sum((y_pred_mn - y) ** 2) / \
                       np.sum((y - y.mean()) ** 2)
    print(f"  In-sample RMSE       : {overall_rmse:.4f}")
    print(f"  In-sample R2         : {overall_r2:.3f}")
    print(f"  Mean uncertainty (1s): {y_pred_sd.mean():.4f}")

    # T-S diagram coloured by observed vs predicted tracer
    plot_ts_diagram(df_valid, feat_names, y_pred_mn)
    ts_path = os.path.join(args.output_dir, "ts_diagram_check.png")
    if os.path.exists("ts_diagram_check.png"):
        os.replace("ts_diagram_check.png", ts_path)
    print(f"  Saved: {ts_path}")

    # Depth residual diagnostics
    plot_depth_residuals(y, y_pred_mn, df_valid["depth"].values)
    res_path = os.path.join(args.output_dir, "residual_diagnostics.png")
    if os.path.exists("residual_diagnostics.png"):
        os.replace("residual_diagnostics.png", res_path)
    print(f"  Saved: {res_path}")

    # MC-Dropout uncertainty vs depth
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(y_pred_sd, df_valid["depth"].values,
                    c=np.abs(y_pred_mn - y), cmap="hot_r", s=4, alpha=0.5)
    ax.invert_yaxis()
    ax.set(xlabel="Prediction uncertainty (1s)", ylabel="Depth (m)",
           title="MC-Dropout uncertainty vs depth\n(colour = |residual|)")
    plt.colorbar(sc, ax=ax, label="|residual|")
    plt.tight_layout()
    unc_path = os.path.join(args.output_dir, "uncertainty_vs_depth.png")
    plt.savefig(unc_path, dpi=150)
    plt.close()
    print(f"  Saved: {unc_path}")

    # ── 6. (Optional) Ocean model inference ───────────────────────────────────
    if args.model is not None:
        import xarray as xr
        print(f"\n── 6. Inferring on ocean model: {args.model} ─────────")
        ds     = xr.open_dataset(args.model)
        ds_out = infer_on_model_field(
            ds, model_final, scaler, feat_names,
            target_year  = args.model_year,
            target_month = args.model_month,
            device       = device,
        )

        nc_path = os.path.join(
            args.output_dir,
            f"tracer_predicted_{args.model_year}_{args.model_month:02d}.nc"
        )
        ds_out.to_netcdf(nc_path)
        print(f"  Saved NetCDF: {nc_path}")

        # Quick surface map
        depth_dim = "depth" if "depth" in ds_out.dims else "z"
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ds_out["tracer_pred"].isel({depth_dim: 0}).plot(
            ax=axes[0], cmap="viridis")
        axes[0].set_title(
            f"Predicted tracer — {args.model_year}-{args.model_month:02d}")
        ds_out["tracer_uncertainty"].isel({depth_dim: 0}).plot(
            ax=axes[1], cmap="Oranges")
        axes[1].set_title("Prediction uncertainty (1s)")
        plt.tight_layout()
        map_path = os.path.join(args.output_dir, "model_surface_map.png")
        plt.savefig(map_path, dpi=150)
        plt.close()
        print(f"  Saved: {map_path}")

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  All outputs written to: {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()