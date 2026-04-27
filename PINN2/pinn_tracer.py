# ═══════════════════════════════════════════════════════════════════════════════
# PINN for Conservative Tracer Inference from T-S-z
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import gsw
import xarray as xr
import matplotlib.pyplot as plt

# ── 1. FEATURE ENGINEERING ─────────────────────────────────────────────────────
#
# Key idea: rotate (T, S) → (σ₀, spiciness)
#   σ₀       = potential density anomaly  → which isopycnal you're on
#   spiciness = orthogonal to σ₀ in T-S   → water mass identity ON that isopycnal
#
# This separates "across-isopycnal" from "along-isopycnal" variation,
# which is the natural coordinate system for conservative tracer physics.

def build_features(df):
    """
    Input DataFrame must have: temp, salinity, depth, lon, lat, year, month
    Returns feature array and feature names.
    """
    SA  = gsw.SA_from_SP(df["salinity"].values, df["depth"].values,
                          df["lon"].values, df["lat"].values)
    CT  = gsw.CT_from_t(SA, df["temp"].values, df["depth"].values)
    p   = gsw.p_from_z(-np.abs(df["depth"].values), df["lat"].values)

    sigma0   = gsw.sigma0(SA, CT)           # potential density (σ₀)
    spice    = gsw.spiciness0(SA, CT)       # along-isopycnal water mass index
    sigma2   = gsw.sigma2(SA, CT)           # deep water: σ₂ better than σ₀
    N2, _    = gsw.Nsquared(SA, CT, p,      # stratification: physics-informed
                             df["lat"].values)

    # Seasonal cycle: sin/cos encoding preserves periodicity
    season_sin = np.sin(2 * np.pi * df["month"].values / 12)
    season_cos = np.cos(2 * np.pi * df["month"].values / 12)

    # Normalised year: preserves secular trend information
    year_norm = (df["year"].values - 1970) / 53.0

    # Depth: log-transform compresses the surface bias (most obs <500m)
    log_depth = np.log1p(np.abs(df["depth"].values))

    feature_names = ["sigma0", "spice", "sigma2", "log_depth",
                     "year_norm", "season_sin", "season_cos"]

    X = np.column_stack([sigma0, spice, sigma2,
                         log_depth, year_norm, season_sin, season_cos])

    # Flag and drop NaNs (bad casts, land-locked etc.)
    has_tracer = "tracer" in df.columns

    valid = np.isfinite(X).all(axis=1)
    if has_tracer:
        valid = valid & np.isfinite(df["tracer"].values)
    
    y = df["tracer"].values[valid] if has_tracer else None
    
    return X[valid], y, feature_names, valid

# ── 2. NETWORK ARCHITECTURE ────────────────────────────────────────────────────
#
# Residual MLP with MC-Dropout for uncertainty.
# Skip connections help gradients flow for the physics Jacobian computation.

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))      # residual connection


class TracerPINN(nn.Module):
    def __init__(self, n_features, hidden_dim=256, n_blocks=4, dropout=0.15):
        super().__init__()
        self.embed   = nn.Sequential(nn.Linear(n_features, hidden_dim),
                                     nn.LayerNorm(hidden_dim), nn.GELU())
        self.blocks  = nn.ModuleList([ResidualBlock(hidden_dim, dropout)
                                      for _ in range(n_blocks)])
        self.head    = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h).squeeze(-1)


# ── 3. PHYSICS LOSS TERMS ──────────────────────────────────────────────────────

def jacobian_columns(model, x, col_indices):
    """
    Compute ∂C/∂x_i for selected feature columns via autograd.
    Returns tensor of shape (batch, len(col_indices)).
    """
    x = x.clone().requires_grad_(True)
    C = model(x)
    grads = torch.autograd.grad(C.sum(), x, create_graph=True)[0]
    return C, grads[:, col_indices]


def physics_losses(model, x_batch, feature_names):
    idx_sig = feature_names.index("sigma0")
    idx_spi = feature_names.index("spice")
    idx_yr  = feature_names.index("year_norm")

    x_in = x_batch.clone().requires_grad_(True)
    C = model(x_in)

    grads_1 = torch.autograd.grad(C.sum(), x_in, create_graph=True)[0]

    dC_dsig = grads_1[:, idx_sig]
    dC_dspi = grads_1[:, idx_spi]
    dC_dyr  = grads_1[:, idx_yr]

    eps = 1e-6

    # ── L1: directional constraint (stable)
    alpha = 0.2
    L_diapycnal = torch.relu(dC_dspi**2 - alpha * dC_dsig**2).mean()

    # ── L2: smoothness (second derivative)
    grads_2 = torch.autograd.grad(
        dC_dsig, x_in,
        grad_outputs=torch.ones_like(dC_dsig),
        create_graph=True
    )[0]
    d2C_dsig2 = grads_2[:, idx_sig]
    L_smooth = (d2C_dsig2**2).mean()

    # ── L3: temporal constraint
    L_secular = torch.relu(dC_dyr.abs() - 3.0).pow(2).mean()

    return L_diapycnal, L_smooth, L_secular


# ── 4. TEMPORAL CROSS-VALIDATION ───────────────────────────────────────────────
#
# NEVER use random splits with oceanographic time series.
# Use temporal blocking: train on past, evaluate on withheld future periods.

def temporal_cv_splits(years, n_splits=5):
    """
    Expanding-window temporal CV splits.
    Each fold trains on all data before a cutoff, validates on next block.

    Years 1970-2023 → ~10-year validation blocks:
      Fold 1: train <1985, val 1985-1991
      Fold 2: train <1991, val 1991-1997
      ...
    """
    year_min, year_max = years.min(), years.max()
    block = (year_max - year_min) / (n_splits + 1)
    cutoffs = [year_min + block * (i + 1) for i in range(n_splits)]
    splits = []
    for i, cut in enumerate(cutoffs[:-1]):
        val_end = cutoffs[i + 1]
        train_idx = np.where(years < cut)[0]
        val_idx   = np.where((years >= cut) & (years < val_end))[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits


# ── 5. TRAINING LOOP ───────────────────────────────────────────────────────────

def train_pinn(model, X_tr, y_tr, X_val, y_val,
               feature_names, scaler,
               n_epochs=300, batch_size=512, lr=5e-4,
               lambda_diap=0.05, lambda_smooth=0.02, lambda_sec=0.01,
               device="cpu"):

    X_tr_t  = torch.tensor(scaler.transform(X_tr),  dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr,                    dtype=torch.float32)
    X_val_t = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    y_val_t = torch.tensor(y_val,                   dtype=torch.float32)

    loader   = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                          batch_size=batch_size, shuffle=True)
    model    = model.to(device)
    opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched    = torch.optim.lr_scheduler.OneCycleLR(
                   opt, max_lr=lr, epochs=n_epochs,
                   steps_per_epoch=len(loader), pct_start=0.1)
    huber    = nn.HuberLoss(delta=1.0)    # robust to outlier casts
    history  = []

    for epoch in range(n_epochs):
        model.train()
        e_data = e_phys = 0

        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()

            # ── data loss ──
            yp    = model(Xb)
            L_dat = huber(yp, yb)

            # ── physics losses ──
            Ld, Ls, Lt = physics_losses(model, Xb, feature_names)
            L_phys = lambda_diap * Ld + lambda_smooth * Ls + lambda_sec * Lt

            (L_dat + L_phys).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            e_data += L_dat.item()
            e_phys += L_phys.item()

        # ── validation ──
        model.eval()
        with torch.no_grad():
            yv_pred = model(X_val_t.to(device)).cpu()
            rmse    = torch.sqrt(((yv_pred - y_val_t)**2).mean()).item()
            r2      = 1 - ((yv_pred - y_val_t)**2).sum().item() / \
                          ((y_val_t - y_val_t.mean())**2).sum().item()

        history.append({"epoch": epoch,
                        "data_loss": e_data / len(loader),
                        "phys_loss": e_phys / len(loader),
                        "val_rmse": rmse, "val_r2": r2})

        if epoch % 50 == 0:
            print(f"[{epoch:3d}] data={e_data/len(loader):.4f}  "
                  f"phys={e_phys/len(loader):.5f}  "
                  f"val RMSE={rmse:.4f}  R²={r2:.3f}")

    return model, pd.DataFrame(history)


# ── 6. MC-DROPOUT UNCERTAINTY ──────────────────────────────────────────────────
#
# Enable dropout at inference time → sample N forward passes → 
# mean = prediction, std = epistemic uncertainty
# Regions with no training analogues (deep water, poles) will show high σ

def predict_with_uncertainty(model, X_scaled, n_samples=100,
                             batch_size=100_000, device="cpu"):

    model.train()  # keep dropout ON

    n = X_scaled.shape[0]
    preds_mean = np.zeros(n, dtype=np.float32)
    preds_sq   = np.zeros(n, dtype=np.float32)

    for i in range(0, n, batch_size):
        X_batch = torch.tensor(
            X_scaled[i:i+batch_size],
            dtype=torch.float32
        ).to(device)

        batch_preds = []

        for _ in range(n_samples):
            with torch.no_grad():
                y = model(X_batch).cpu().numpy()
            batch_preds.append(y)

        batch_preds = np.stack(batch_preds)

        preds_mean[i:i+batch_size] = batch_preds.mean(axis=0)
        preds_sq[i:i+batch_size]   = (batch_preds**2).mean(axis=0)

        del X_batch, batch_preds
        torch.cuda.empty_cache()

    preds_std = np.sqrt(preds_sq - preds_mean**2)
    return preds_mean, preds_std


# ── 7. FULL PIPELINE ───────────────────────────────────────────────────────────

def run_pipeline(df, n_epochs=300):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device}")

    # Feature engineering
    X, y, feat_names, valid_mask = build_features(df)
    years = df["year"].values[valid_mask]
    print(f"Valid observations: {len(y):,} / {len(df):,}")
    print(f"Features: {feat_names}")

    # Temporal CV to pick λ hyperparameters
    splits = temporal_cv_splits(years, n_splits=5)
    cv_r2  = []
    for fold, (tr_idx, val_idx) in enumerate(splits):
        model_cv = TracerPINN(n_features=len(feat_names))
        scaler_cv = StandardScaler().fit(X[tr_idx])
        model_cv, hist = train_pinn(
            model_cv, X[tr_idx], y[tr_idx],
                       X[val_idx], y[val_idx],
            feat_names, scaler_cv, n_epochs=n_epochs, device=device
        )
        cv_r2.append(hist["val_r2"].iloc[-1])
        print(f"Fold {fold+1}  val R²={cv_r2[-1]:.3f}")
    print(f"\nCV R² = {np.mean(cv_r2):.3f} ± {np.std(cv_r2):.3f}")

    # Final model: train on all data
    scaler = StandardScaler().fit(X)
    model_final = TracerPINN(n_features=len(feat_names)).to(device)
    model_final, history = train_pinn(
        model_final, X, y, X, y,      # val=train for final monitoring
        feat_names, scaler, n_epochs=n_epochs, device=device
    )

    return model_final, scaler, feat_names, history


# ── 8. INFERENCE ON OCEAN MODEL ───────────────────────────────────────────────

def infer_on_model_field(ds, model, scaler, feat_names,
                         target_year, target_month, device="cpu"):
    """
    Infer tracer field on an xarray Dataset.
    Handles any (depth, lat, lon) dimension ordering and naming.
    """
    import xarray as xr

    # ── 1. Detect dimension / variable names ──────────────────────────────────
    def _find_dim(ds, candidates):
        for c in candidates:
            if c in ds.dims: return c
        raise KeyError(f"None of {candidates} found in dims {list(ds.dims)}")

    def _find_var(ds, candidates):
        for c in candidates:
            if c in ds: return c
        raise KeyError(f"None of {candidates} found in dataset {list(ds)}")

    lat_dim = _find_dim(ds, ["latitude",  "lat", "y", "nav_lat"])
    lon_dim = _find_dim(ds, ["longitude", "lon", "x", "nav_lon"])
    dep_dim = _find_dim(ds, ["depth", "deptht", "z", "lev", "level"])

    temp_var = _find_var(ds, ["thetao", "temp", "temperature", "votemper"])
    sal_var  = _find_var(ds, ["so",     "sal",  "salinity",    "vosaline"])

    # ── 2. Build 3-D coordinate meshes (guaranteed same shape) ────────────────
    lat_1d = ds[lat_dim].values.ravel()   # (nlat,)
    lon_1d = ds[lon_dim].values.ravel()   # (nlon,)
    dep_1d = ds[dep_dim].values.ravel()   # (ndep,)

    # meshgrid -> all (ndep, nlat, nlon) regardless of original coord order
    dep_3d, lat_3d, lon_3d = np.meshgrid(dep_1d, lat_1d, lon_1d,
                                          indexing="ij")  # (D, Y, X)

    # ── 3. Extract temp/sal, transposing to (depth, lat, lon) if needed ───────
    def _to_dep_lat_lon(da):
        """Transpose DataArray to (dep_dim, lat_dim, lon_dim)."""
        return da.transpose(dep_dim, lat_dim, lon_dim).values  # numpy

    temp_3d = _to_dep_lat_lon(ds[temp_var])   # (D, Y, X)
    sal_3d  = _to_dep_lat_lon(ds[sal_var])    # (D, Y, X)

    # ── 4. Ravel everything → flat vectors ────────────────────────────────────
    temp_flat = temp_3d.ravel()
    sal_flat  = sal_3d.ravel()
    dep_flat  = dep_3d.ravel()
    lat_flat  = lat_3d.ravel()
    lon_flat  = lon_3d.ravel()

    assert temp_flat.shape == sal_flat.shape == dep_flat.shape \
                           == lat_flat.shape == lon_flat.shape, \
        "Coordinate/data arrays still have mismatched lengths after meshgrid."

    # ── 5. Build DataFrame and drop NaNs (ocean land mask etc.) ──────────────
    df_model = pd.DataFrame({
        "temp"    : temp_flat,
        "salinity": sal_flat,
        "depth"   : dep_flat,
        "lat"     : lat_flat,
        "lon"     : lon_flat,
        "year"    : target_year,
        "month"   : target_month,
    })

    valid_mask  = df_model.notna().all(axis=1).values
    df_valid    = df_model[valid_mask].reset_index(drop=True)
    n_total     = len(df_model)
    n_valid     = len(df_valid)
    print(f"  Grid points: {n_total:,} total, {n_valid:,} unmasked "
          f"({100*n_valid/n_total:.1f} %)")

    # ── 6. Build physics features & scale ─────────────────────────────────────
    X_raw, _, _, feat_valid = build_features(df_valid)
    df_feat   = df_valid.iloc[feat_valid].reset_index(drop=True)
    valid_mask2 = np.where(valid_mask)[0][feat_valid]   # indices in full flat array

    X_sc = scaler.transform(X_raw)

    # ── 7. Predict ────────────────────────────────────────────────────────────
    y_pred_mn, y_pred_sd = predict_with_uncertainty(
        model, X_sc, n_samples=100, device=device
    )

    # ── 8. Place predictions back onto the 3-D grid ───────────────────────────
    pred_full  = np.full(n_total, np.nan)
    uncrt_full = np.full(n_total, np.nan)
    pred_full[valid_mask2]  = y_pred_mn
    uncrt_full[valid_mask2] = y_pred_sd

    # Reshape to (depth, lat, lon)
    D, Y, X = dep_3d.shape
    pred_3d  = pred_full.reshape(D, Y, X)
    uncrt_3d = uncrt_full.reshape(D, Y, X)

    # ── 9. Pack into xarray Dataset ───────────────────────────────────────────
    coords = {dep_dim: dep_1d, lat_dim: lat_1d, lon_dim: lon_1d}
    ds_out = xr.Dataset({
        "tracer_pred": xr.DataArray(
            pred_3d, dims=[dep_dim, lat_dim, lon_dim], coords=coords,
            attrs={"long_name": "Predicted tracer", "units": "model units"}
        ),
        "tracer_uncertainty": xr.DataArray(
            uncrt_3d, dims=[dep_dim, lat_dim, lon_dim], coords=coords,
            attrs={"long_name": "MC-Dropout uncertainty (1 sigma)"}
        ),
    })
    ds_out.attrs["source"]      = "PINN tracer inference"
    ds_out.attrs["target_date"] = f"{target_year}-{target_month:02d}"
    return ds_out


# ── 9. DIAGNOSTICS ─────────────────────────────────────────────────────────────

def plot_ts_diagram(df, feat_names, y_pred):
    """T-S diagram coloured by tracer — key sanity check.
    Predictions should follow isopycnal contours for a conservative tracer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sc1 = ax1.scatter(df["salinity"], df["temp"],
                      c=df["tracer"], cmap="plasma", s=5, alpha=0.6)
    ax1.set(xlabel="Salinity", ylabel="Temperature (°C)",
            title="Observed tracer on T-S diagram")
    plt.colorbar(sc1, ax=ax1, label="Tracer")

    sc2 = ax2.scatter(df["salinity"], df["temp"],
                      c=y_pred, cmap="plasma", s=5, alpha=0.6)
    ax2.set(xlabel="Salinity", ylabel="Temperature (°C)",
            title="Predicted tracer on T-S diagram")
    plt.colorbar(sc2, ax=ax2, label="Tracer")

    plt.tight_layout()
    plt.savefig("ts_diagram_check.png", dpi=150)


def plot_depth_residuals(y_true, y_pred, depths):
    """Residuals should be white noise vs depth — structure means missing physics."""
    resid = y_pred - y_true
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(resid, depths, s=3, alpha=0.3)
    axes[0].axvline(0, color="r"); axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Depth (m)"); axes[0].invert_yaxis()
    axes[0].set_title("Residual vs Depth — should be centred")

    axes[1].scatter(y_true, y_pred, s=3, alpha=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, "r--"); axes[1].set_xlabel("Observed")
    axes[1].set_ylabel("Predicted"); axes[1].set_title("1:1 plot")

    axes[2].hist(resid, bins=60, edgecolor="k")
    axes[2].set_xlabel("Residual"); axes[2].set_title("Residual distribution")

    plt.tight_layout()
    plt.savefig("residual_diagnostics.png", dpi=150)