"""
main.py — PINN Conservative Tracer Inference Pipeline
======================================================
Runs the full pipeline:
  1. Load observations
  2. Engineer physics-based features (TEOS-10 + velocity)
  3. Temporal cross-validation
  4. Train final PINN on all data
  5. Diagnostics & plots
  6. (Optional) Inference on ocean model NetCDF

Usage:
    python main.py --obs observations.csv [--model ocean_model.nc]
                   [--tracer_col oxygen] [--epochs 300] [--output_dir ./output]

Velocity:
    If your CSV has u and v columns (m/s climatological velocity):
        --u_col u  --v_col v
    If absent, the advection loss is automatically skipped.

Date handling:
    If your CSV has separate year/month columns, do nothing.
    If your CSV has a single date column (e.g. "09/06/2016"), pass:
        --date_col date
        --date_format "%d/%m/%Y"   (recommended — avoids day/month ambiguity)
"""

# ── Imports — velocity-aware versions take priority ───────────────────────────
import argparse
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd
import torch
from pinn_tracer import (
    TracerPINN,
    build_features,
    infer_on_model_field,
    plot_depth_residuals,
    plot_ts_diagram,
    predict_with_uncertainty,
    temporal_cv_splits,
    train_pinn,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="PINN conservative tracer inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Observation data ───────────────────────────────────────────────────────
    grp_data = p.add_argument_group("Observation data")
    grp_data.add_argument(
        "--obs",
        required=True,
        help="CSV of observations. Required columns: "
        "temp, salinity, depth, lon, lat, <tracer>. "
        "Date: either separate year/month columns or --date_col. "
        "Velocity: u, v columns (optional, enables advection loss).",
    )
    grp_data.add_argument(
        "--tracer_col", default="tracer", help="Name of the tracer column."
    )
    grp_data.add_argument(
        "--u_col",
        default="u",
        help="Name of the eastward velocity column (m/s). "
        "If the column is absent the advection loss is skipped.",
    )
    grp_data.add_argument(
        "--v_col", default="v", help="Name of the northward velocity column (m/s)."
    )
    grp_data.add_argument(
        "--cast_col",
        default="cast_id",
        help="Column identifying individual casts/profiles for correct "
        "per-cast N² computation. If absent, a per-point proxy is used.",
    )
    grp_data.add_argument(
        "--date_col",
        default=None,
        help="Combined date column, e.g. 'date'. Use instead of year/month.",
    )
    grp_data.add_argument(
        "--date_format",
        default=None,
        help="strptime format for --date_col, e.g. '%%d/%%m/%%Y'.",
    )

    # ── Ocean model inference ──────────────────────────────────────────────────
    grp_model = p.add_argument_group("Ocean model inference (optional)")
    grp_model.add_argument(
        "--model", default=None, help="NetCDF ocean model file for inference."
    )
    grp_model.add_argument("--model_year", type=int, default=2020)
    grp_model.add_argument("--model_month", type=int, default=6)
    grp_model.add_argument(
        "--model_time_index",
        type=int,
        default=0,
        help="Time index to select if the model file has a time dimension.",
    )
    grp_model.add_argument(
        "--uvel_var",
        default=None,
        help="u-velocity variable name in the NetCDF (auto-detected if omitted).",
    )
    grp_model.add_argument(
        "--vvel_var",
        default=None,
        help="v-velocity variable name in the NetCDF (auto-detected if omitted).",
    )

    # ── Training hyperparameters ───────────────────────────────────────────────
    grp_train = p.add_argument_group("Training hyperparameters")
    grp_train.add_argument("--epochs", type=int, default=300)
    grp_train.add_argument("--batch_size", type=int, default=512)
    grp_train.add_argument("--lr", type=float, default=5e-4)
    grp_train.add_argument("--hidden_dim", type=int, default=256)
    grp_train.add_argument("--n_blocks", type=int, default=4)
    grp_train.add_argument(
        "--cv_folds", type=int, default=5, help="Number of temporal CV folds."
    )
    grp_train.add_argument(
        "--mc_samples",
        type=int,
        default=100,
        help="MC-Dropout forward passes for uncertainty estimation.",
    )

    # ── Physics loss weights ───────────────────────────────────────────────────
    grp_phys = p.add_argument_group("Physics loss weights")
    grp_phys.add_argument(
        "--lambda_diap",
        type=float,
        default=0.05,
        help="Diapycnal/along-isopycnal gradient ratio penalty.",
    )
    grp_phys.add_argument(
        "--lambda_smooth",
        type=float,
        default=0.02,
        help="Second-order smoothness in sigma-0.",
    )
    grp_phys.add_argument(
        "--lambda_sec", type=float, default=0.01, help="Secular temporal trend penalty."
    )
    grp_phys.add_argument(
        "--lambda_strat",
        type=float,
        default=0.05,
        help="Stratification-weighted diapycnal penalty.",
    )
    grp_phys.add_argument(
        "--lambda_advect",
        type=float,
        default=0.03,
        help="Along-stream tracer gradient penalty (requires u, v columns).",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# DATE PARSING
# ═══════════════════════════════════════════════════════════════════════════════


def parse_date_column(df, date_col, date_format=None):
    """
    Parse a combined date column into separate integer 'year' and 'month' cols.

    Supported without explicit format (dayfirst=True):
        09/06/2016  ->  year=2016, month=6
        2016-06-09  ->  year=2016, month=6
        09-Jun-2016 ->  year=2016, month=6

    WARNING: DD/MM/YYYY and MM/DD/YYYY are ambiguous when day <= 12.
    Always pass --date_format "%d/%m/%Y" to guarantee correct parsing.
    """
    if date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' not found. Available: {list(df.columns)}"
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
            f"\n  Could not parse {n_failed:,} dates in '{date_col}' "
            f"using format '{fmt_used}'.\n"
            f"  Example unparseable values: {bad}\n"
            f"  Fix: pass --date_format explicitly, e.g.:\n"
            f"    --date_format '%d/%m/%Y'   for  09/06/2016\n"
            f"    --date_format '%m/%d/%Y'   for  06/09/2016\n"
            f"    --date_format '%Y-%m-%d'   for  2016-06-09\n"
            f"    --date_format '%d-%b-%Y'   for  09-Jun-2016"
        )

    df = df.copy()
    df["year"] = parsed.dt.year.astype(int)
    df["month"] = parsed.dt.month.astype(int)

    n_show = min(4, len(df))
    sample = pd.DataFrame(
        {
            "raw": raw.iloc[:n_show].values,
            "parsed": parsed.iloc[:n_show].dt.strftime("%Y-%m-%d").values,
            "year": df["year"].iloc[:n_show].values,
            "month": df["month"].iloc[:n_show].values,
        }
    )
    print(f"  Date format used : {fmt_used}")
    print(f"  Parse sample — verify month/day order is correct:")
    print(sample.to_string(index=False))
    print()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_observations(
    path, tracer_col, u_col="u", v_col="v", date_col=None, date_format=None
):
    """
    Load observation CSV.

    Velocity columns (u_col, v_col) are optional — if absent a warning is
    printed and the advection loss will be skipped at training time.
    All other required columns are enforced.
    """
    df = pd.read_csv(path)

    # ── Date ──────────────────────────────────────────────────────────────────
    if date_col is not None:
        df = parse_date_column(df, date_col, date_format)
    else:
        for col in ("year", "month"):
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found. Either include separate "
                    f"'year' and 'month' columns, or pass --date_col."
                )

    # ── Required columns ──────────────────────────────────────────────────────
    required = {"temp", "salinity", "depth", "lon", "lat", "year", "month"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if tracer_col not in df.columns:
        raise ValueError(
            f"Tracer column '{tracer_col}' not found. Available: {list(df.columns)}"
        )
    if tracer_col != "tracer":
        df = df.rename(columns={tracer_col: "tracer"})

    # ── Velocity columns (optional) ───────────────────────────────────────────
    has_u = u_col in df.columns
    has_v = v_col in df.columns

    if has_u and has_v:
        if u_col != "u":
            df = df.rename(columns={u_col: "u"})
        if v_col != "v":
            df = df.rename(columns={v_col: "v"})
        print(
            f"  Velocity columns found: '{u_col}', '{v_col}' -> advection loss ENABLED"
        )
    else:
        missing_vel = []
        if not has_u:
            missing_vel.append(u_col)
        if not has_v:
            missing_vel.append(v_col)
        print(
            f"  WARNING: velocity column(s) {missing_vel} not found "
            f"-> advection loss DISABLED"
        )
        # Add zero-filled placeholders so build_features won't KeyError;
        # train_pinn will detect has_velocity=False and skip the loss.
        df["u"] = 0.0
        df["v"] = 0.0

    # ── Drop NaNs ─────────────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["temp", "salinity", "depth", "lon", "lat", "tracer"])
    print(f"  Loaded {before:,} rows, {len(df):,} after dropping NaNs")
    return df, (has_u and has_v)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarise_dataset(df, has_velocity):
    print("\n── Dataset summary ────────────────────────────────────────")
    print(f"  Observations : {len(df):,}")
    print(f"  Year range   : {int(df['year'].min())} - {int(df['year'].max())}")
    print(f"  Depth range  : {df['depth'].min():.1f} - {df['depth'].max():.1f} m")
    print(f"  Tracer range : {df['tracer'].min():.3f} - {df['tracer'].max():.3f}")
    print(f"  Lat range    : {df['lat'].min():.1f} - {df['lat'].max():.1f} deg")
    print(f"  Lon range    : {df['lon'].min():.1f} - {df['lon'].max():.1f} deg")
    if has_velocity:
        speed = np.sqrt(df["u"] ** 2 + df["v"] ** 2)
        print(
            f"  Speed range  : {speed.min():.4f} - {speed.max():.4f} m/s "
            f"(mean {speed.mean():.4f})"
        )
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
    path = os.path.join(output_dir, "cv_results.csv")
    df_cv.to_csv(path, index=False)
    print(f"  Saved: {path}")

    print("\n── Temporal CV results ────────────────────────────────────")
    for row in cv_results:
        print(
            f"  Fold {row['fold']}  "
            f"train {int(row['train_year_min'])}-{int(row['train_year_max'])}  "
            f"val {int(row['val_year_min'])}-{int(row['val_year_max'])}  "
            f"R2={row['val_r2']:.3f}  RMSE={row['val_rmse']:.4f}  "
            f"N_val={int(row['n_val'])}"
        )
    r2s = [r["val_r2"] for r in cv_results]
    print(f"\n  Mean R2: {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}")
    print()


def plot_residuals_vs_speed(df_valid, y_pred_mn, y, output_dir):
    """
    Residual vs current speed diagnostic.
    If the advection loss is working, residuals should be flat across speed —
    i.e. the model should not be systematically worse in fast currents.
    """
    speed = np.sqrt(df_valid["u"].values ** 2 + df_valid["v"].values ** 2)
    resid = y_pred_mn - y

    # Bin by speed quantile for a clean summary line
    n_bins = 20
    bins = np.percentile(speed, np.linspace(0, 100, n_bins + 1))
    bin_idx = np.digitize(speed, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_mid = 0.5 * (bins[:-1] + bins[1:])
    bin_rmse = np.array(
        [
            np.sqrt(np.mean(resid[bin_idx == i] ** 2))
            if (bin_idx == i).sum() > 0
            else np.nan
            for i in range(n_bins)
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(speed, resid, s=3, alpha=0.3, c="steelblue")
    axes[0].axhline(0, color="r", lw=1)
    axes[0].set(
        xlabel="Current speed (m/s)",
        ylabel="Residual",
        title="Residual vs speed\n(should be centred at all speeds)",
    )

    axes[1].plot(bin_mid, bin_rmse, "o-", color="steelblue", ms=5)
    axes[1].set(
        xlabel="Current speed (m/s)",
        ylabel="RMSE",
        title="RMSE by speed bin\n(flat = advection loss working)",
    )

    plt.tight_layout()
    path = os.path.join(output_dir, "residuals_vs_speed.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(" PINN Conservative Tracer Inference")
    print(f"{'=' * 60}")
    print(f"  Device       : {device}")
    print(f"  Obs file     : {args.obs}")
    print(f"  Tracer col   : {args.tracer_col}")
    if args.date_col:
        fmt = args.date_format if args.date_format else "auto"
        print(f"  Date col     : {args.date_col}  (format: {fmt})")
    print(f"  Velocity cols: u='{args.u_col}', v='{args.v_col}'")
    print(f"  Cast col     : '{args.cast_col}'")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"\n  Physics loss weights:")
    print(f"    lambda_diap   = {args.lambda_diap}")
    print(f"    lambda_smooth = {args.lambda_smooth}")
    print(f"    lambda_sec    = {args.lambda_sec}")
    print(f"    lambda_strat  = {args.lambda_strat}")
    print(f"    lambda_advect = {args.lambda_advect} (only applied if u/v present)")
    print()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("── 1. Loading observations ─────────────────────────────────")
    df, has_velocity = load_observations(
        args.obs,
        args.tracer_col,
        u_col=args.u_col,
        v_col=args.v_col,
        date_col=args.date_col,
        date_format=args.date_format,
    )
    summarise_dataset(df, has_velocity)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("── 2. Engineering features (TEOS-10 + velocity) ────────────")
    X, y, feat_names, valid_mask = build_features(df, cast_col=args.cast_col)
    df_valid = df.iloc[valid_mask].reset_index(drop=True)
    years = df_valid["year"].values

    # Extract raw u, v aligned to valid observations for the physics loss.
    # If velocity was absent these are all-zero arrays and has_velocity=False,
    # so train_pinn will skip the advection loss regardless.
    u_valid = df_valid["u"].values.astype(np.float32)
    v_valid = df_valid["v"].values.astype(np.float32)

    print(f"  Features : {feat_names}")
    print(f"  Valid obs: {len(y):,}")

    # ── 3. Temporal cross-validation ──────────────────────────────────────────
    print(f"\n── 3. Temporal CV ({args.cv_folds} folds) ───────────────────────────")
    splits = temporal_cv_splits(years, n_splits=args.cv_folds)
    cv_results = []

    for fold, (tr_idx, val_idx) in enumerate(splits):
        print(
            f"\n  [Fold {fold + 1}/{len(splits)}]  "
            f"train n={len(tr_idx):,}  val n={len(val_idx):,}"
        )

        scaler_cv = StandardScaler().fit(X[tr_idx])
        model_cv = TracerPINN(
            n_features=len(feat_names),
            hidden_dim=args.hidden_dim,
            n_blocks=args.n_blocks,
        ).to(device)

        model_cv, hist_cv = train_pinn(
            model_cv,
            X[tr_idx],
            y[tr_idx],
            X[val_idx],
            y[val_idx],
            feat_names,
            scaler_cv,
            u_tr=u_valid[tr_idx] if has_velocity else None,
            v_tr=v_valid[tr_idx] if has_velocity else None,
            u_val=u_valid[val_idx] if has_velocity else None,
            v_val=v_valid[val_idx] if has_velocity else None,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_diap=args.lambda_diap,
            lambda_smooth=args.lambda_smooth,
            lambda_sec=args.lambda_sec,
            lambda_strat=args.lambda_strat,
            lambda_advect=args.lambda_advect if has_velocity else 0.0,
            device=device,
        )

        cv_results.append(
            {
                "fold": fold + 1,
                "train_year_min": years[tr_idx].min(),
                "train_year_max": years[tr_idx].max(),
                "val_year_min": years[val_idx].min(),
                "val_year_max": years[val_idx].max(),
                "n_train": len(tr_idx),
                "n_val": len(val_idx),
                "val_r2": hist_cv["val_r2"].iloc[-1],
                "val_rmse": hist_cv["val_rmse"].iloc[-1],
            }
        )

    save_cv_results(cv_results, args.output_dir)

    # ── 4. Final model (train on all data) ────────────────────────────────────
    print("── 4. Training final model on all data ─────────────────────")
    scaler = StandardScaler().fit(X)
    model_final = TracerPINN(
        n_features=len(feat_names),
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
    ).to(device)

    model_final, history = train_pinn(
        model_final,
        X,
        y,
        X,
        y,  # val == train for final loss monitoring only
        feat_names,
        scaler,
        u_tr=u_valid if has_velocity else None,
        v_tr=v_valid if has_velocity else None,
        u_val=u_valid if has_velocity else None,
        v_val=v_valid if has_velocity else None,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_diap=args.lambda_diap,
        lambda_smooth=args.lambda_smooth,
        lambda_sec=args.lambda_sec,
        lambda_strat=args.lambda_strat,
        lambda_advect=args.lambda_advect if has_velocity else 0.0,
        device=device,
    )
    save_history_plot(history, args.output_dir)

    # Save model + scaler for future inference
    model_path = os.path.join(args.output_dir, "pinn_tracer.pt")
    torch.save(
        {
            "model_state": model_final.state_dict(),
            "feat_names": feat_names,
            "hidden_dim": args.hidden_dim,
            "n_blocks": args.n_blocks,
            "n_features": len(feat_names),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "has_velocity": has_velocity,
        },
        model_path,
    )
    print(f"  Saved model: {model_path}")

    # ── 5. Diagnostics ────────────────────────────────────────────────────────
    print("\n── 5. Diagnostics ──────────────────────────────────────────")
    X_sc = scaler.transform(X)
    y_pred_mn, y_pred_sd = predict_with_uncertainty(
        model_final,
        X_sc,
        n_samples=args.mc_samples,
        device=device,
    )

    # Save predictions
    df_pred = df_valid.copy()
    df_pred["tracer_pred"] = y_pred_mn
    df_pred["tracer_uncert"] = y_pred_sd
    df_pred["residual"] = y_pred_mn - y
    pred_path = os.path.join(args.output_dir, "predictions.csv")
    df_pred.to_csv(pred_path, index=False)
    print(f"  Saved: {pred_path}")

    overall_rmse = np.sqrt(np.mean((y_pred_mn - y) ** 2))
    overall_r2 = 1 - (np.sum((y_pred_mn - y) ** 2) / np.sum((y - y.mean()) ** 2))
    print(f"  In-sample RMSE       : {overall_rmse:.4f}")
    print(f"  In-sample R2         : {overall_r2:.3f}")
    print(f"  Mean uncertainty (1s): {y_pred_sd.mean():.4f}")

    # T-S diagram
    plot_ts_diagram(df_valid, feat_names, y_pred_mn)
    ts_path = os.path.join(args.output_dir, "ts_diagram_check.png")
    if os.path.exists("ts_diagram_check.png"):
        os.replace("ts_diagram_check.png", ts_path)
    print(f"  Saved: {ts_path}")

    # Depth residuals
    plot_depth_residuals(y, y_pred_mn, df_valid["depth"].values)
    res_path = os.path.join(args.output_dir, "residual_diagnostics.png")
    if os.path.exists("residual_diagnostics.png"):
        os.replace("residual_diagnostics.png", res_path)
    print(f"  Saved: {res_path}")

    # Uncertainty vs depth
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        y_pred_sd,
        df_valid["depth"].values,
        c=np.abs(y_pred_mn - y),
        cmap="hot_r",
        s=4,
        alpha=0.5,
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="Prediction uncertainty (1s)",
        ylabel="Depth (m)",
        title="MC-Dropout uncertainty vs depth\n(colour = |residual|)",
    )
    plt.colorbar(sc, ax=ax, label="|residual|")
    plt.tight_layout()
    unc_path = os.path.join(args.output_dir, "uncertainty_vs_depth.png")
    plt.savefig(unc_path, dpi=150)
    plt.close()
    print(f"  Saved: {unc_path}")

    # Residuals vs current speed (only meaningful if velocity present)
    if has_velocity:
        plot_residuals_vs_speed(df_valid, y_pred_mn, y, args.output_dir)

    # ── 6. (Optional) Ocean model inference ───────────────────────────────────
    if args.model is not None:
        import xarray as xr

        print(f"\n── 6. Inferring on ocean model: {args.model} ─────────")
        ds = xr.open_dataset(args.model)
        ds_out = infer_on_model_field(
            ds,
            model_final,
            scaler,
            feat_names,
            target_year=args.model_year,
            target_month=args.model_month,
            target_time_index=args.model_time_index,
            uvel_var=args.uvel_var,
            vvel_var=args.vvel_var,
            device=device,
        )

        nc_path = os.path.join(
            args.output_dir,
            f"tracer_predicted_{args.model_year}_{args.model_month:02d}.nc",
        )
        ds_out.to_netcdf(nc_path)
        print(f"  Saved NetCDF: {nc_path}")

        # Surface map
        dep_dim = next(
            (d for d in ["depth", "deptht", "z", "lev"] if d in ds_out.dims),
            list(ds_out.dims)[0],
        )
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ds_out["tracer_pred"].isel({dep_dim: 0}).plot(ax=axes[0], cmap="viridis")
        axes[0].set_title(
            f"Predicted tracer — {args.model_year}-{args.model_month:02d}"
        )
        ds_out["tracer_uncertainty"].isel({dep_dim: 0}).plot(ax=axes[1], cmap="Oranges")
        axes[1].set_title("Prediction uncertainty (1s)")
        plt.tight_layout()
        map_path = os.path.join(args.output_dir, "model_surface_map.png")
        plt.savefig(map_path, dpi=150)
        plt.close()
        print(f"  Saved: {map_path}")

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  All outputs written to: {args.output_dir}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
