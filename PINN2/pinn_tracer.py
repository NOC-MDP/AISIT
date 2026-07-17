# ═══════════════════════════════════════════════════════════════════════════════
# PINN for Conservative Tracer Inference from T-S-z
# ═══════════════════════════════════════════════════════════════════════════════

import warnings

import gsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from pyproj import Transformer
# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
N2_EPSILON = 1e-8
REF_DEPTH_SCALE = 100.0
SPEED_SCALE = 0.1  # m/s — normalises the w_flow weighting
# (typical open-ocean speed; adjust for your basin)

# ── 1. FEATURE ENGINEERING ─────────────────────────────────────────────────────
#
# Key idea: rotate (T, S) → (σ₀, spiciness)
#   σ₀       = potential density anomaly  → which isopycnal you're on
#   spiciness = orthogonal to σ₀ in T-S   → water mass identity ON that isopycnal
#
# This separates "across-isopycnal" from "along-isopycnal" variation,
# which is the natural coordinate system for conservative tracer physics.
def build_features(df, cast_col="cast_id"):
    """
    Build the full physics-informed feature matrix.

    Required DataFrame columns
    --------------------------
    temp, salinity, depth, lon, lat, year, month, tracer
    u, v          ← NEW: climatological velocity components (m/s)

    Optional
    --------
    cast_col      profile/cast identifier for correct per-cast N² computation

    Returns
    -------
    X           : np.ndarray  (n_valid, n_features)
    y           : np.ndarray  (n_valid,)  tracer values
    feat_names  : list[str]
    valid_mask  : boolean index array into the original df rows
    """
    for col in ("u", "v"):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in DataFrame. "
                f"Velocity components u and v are required. "
                f"If you only have speed and direction, derive u and v first:\n"
                f"  df['u'] = speed * np.cos(np.deg2rad(direction))\n"
                f"  df['v'] = speed * np.sin(np.deg2rad(direction))"
            )

    # ── TEOS-10 isopycnal coordinates ─────────────────────────────────────────
    SA = gsw.SA_from_SP(
        df["salinity"].values, df["depth"].values, df["lon"].values, df["lat"].values
    )
    CT = gsw.CT_from_t(SA, df["temperature"].values, df["depth"].values)
    sigma0 = gsw.sigma0(SA, CT)
    spice = gsw.spiciness0(SA, CT)
    sigma2 = gsw.sigma2(SA, CT)

    # ── Stratification (N²) per cast ──────────────────────────────────────────
    log_n2 = _compute_log_n2(df, cast_col)

    # ── Depth ─────────────────────────────────────────────────────────────────
    log_depth = np.log1p(np.abs(df["depth"].values))

    # ── Temporal ──────────────────────────────────────────────────────────────
    year_norm = (df["year"].values - 1970) / 53.0
    season_sin = np.sin(2 * np.pi * df["month"].values / 12)
    season_cos = np.cos(2 * np.pi * df["month"].values / 12)

    # ── Geographic position (periodic encoding) ───────────────────────────────
    #
    # Avoids the International Date Line discontinuity:
    #   lon = -180° and +180° map to the same location.
    #
    # Latitude:
    #   latn = sin(lat)
    #
    # Longitude:
    #   lonn_sin = sin(lon)
    #   lonn_cos = cos(lon)
    
    lat_rad = np.deg2rad(df["lat"].values)
    lon_rad = np.deg2rad(df["lon"].values)
    
    lat_norm = np.sin(lat_rad)
    
    lon_sin = np.sin(lon_rad)
    lon_cos = np.cos(lon_rad)

    # ── Velocity-derived features  ← NEW ─────────────────────────────────────
    u = df["u"].values
    v = df["v"].values

    speed = np.sqrt(u**2 + v**2)  # [m/s]  flow intensity

    # Flow direction encoded as (sin θ, cos θ) to avoid ±π discontinuity.
    # Where speed ≈ 0 the direction is undefined — use 0 (no preferred dir).
    eps = 1e-10
    flow_sin = np.where(speed > eps, v / (speed + eps), 0.0)  # sin θ = v/|u|
    flow_cos = np.where(speed > eps, u / (speed + eps), 0.0)  # cos θ = u/|u|

    # Log-speed compresses the dynamic range and is better for the network
    # as a feature.  The raw speed is kept for constructing w_flow in the loss.
    log_speed = np.log1p(speed)

    # ── Assemble ───────────────────────────────────────────────────────────────
    feature_names = [
        "sigma0",
        "spice",
        "sigma2",
        "log_depth",
        "log_n2",
        "year_norm",
        "season_sin",
        "season_cos",
        "lat_norm",
        "lon_sin",
        "lon_cos",
        "log_speed",
        "flow_sin",
        "flow_cos",  # ← NEW: velocity
    ]

    X = np.column_stack(
        [
            sigma0,
            spice,
            sigma2,
            log_depth,
            log_n2,
            year_norm,
            season_sin,
            season_cos,
            lat_norm,
            lon_sin,
            lon_cos,
            log_speed,
            flow_sin,
            flow_cos,
        ]
    )

    # Flag and drop NaNs (bad casts, land-locked etc.)
    has_tracer = "tracer" in df.columns

    valid = np.isfinite(X).all(axis=1)
    if has_tracer:
        valid = valid & np.isfinite(df["tracer"].values)

    print(
        f"  Speed range : {speed[valid].min():.4f} – "
        f"{speed[valid].max():.4f} m/s  "
        f"(mean {speed[valid].mean():.4f})"
    )
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
        return self.act(x + self.block(x))  # residual connection


class TracerPINN(nn.Module):
    def __init__(self, n_features, hidden_dim=256, n_blocks=4, dropout=0.15):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

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


def physics_losses(model, x_batch, feature_names, u_raw, v_raw):
    """
    Five physics-informed loss terms.

    Parameters
    ----------
    model        : TracerPINN (in training mode)
    x_batch      : (B, F) standardised feature tensor
    feature_names: list[str]  — in training order
    u_raw        : (B,) tensor — raw u velocity [m/s] for this batch
    v_raw        : (B,) tensor — raw v velocity [m/s]

    Returns
    -------
    L_diapycnal, L_smooth, L_secular, L_strat, L_advect
    """
    idx = {name: feature_names.index(name) for name in feature_names}
    eps = 1e-6

    x_batch = x_batch.clone().requires_grad_(True)
    C = model(x_batch)
    grads = torch.autograd.grad(C.sum(), x_batch, create_graph=True)[0]

    dC_dsig = grads[:, idx["sigma0"]]
    dC_dspi = grads[:, idx["spice"]]
    dC_dyr = grads[:, idx["year_norm"]]
    dC_dlat = grads[:, idx["lat_norm"]]
    
    dC_dlon_sin = grads[:, idx["lon_sin"]]
    dC_dlon_cos = grads[:, idx["lon_cos"]]
    dC_dlon = dC_dlon_sin + dC_dlon_cos # this may not be best way
    log_n2 = x_batch[:, idx["log_n2"]]

    # ── L1: diapycnal / along-isopycnal ratio ─────────────────────────────────
    L_diapycnal = (dC_dspi**2 / (dC_dsig.detach() ** 2 + eps)).mean()

    # ── L2: second-order smoothness in σ₀ ─────────────────────────────────────
    d2C_dsig2 = torch.autograd.grad(
        dC_dsig.sum(), x_batch, create_graph=True, retain_graph=True
    )[0][:, idx["sigma0"]]
    L_smooth = (d2C_dsig2**2).mean()

    # ── L3: temporal smoothness ───────────────────────────────────────────────
    L_secular = (torch.clamp(dC_dyr.abs() - 3.0, min=0) ** 2).mean()

    # ── L4: stratification-weighted diapycnal penalty ─────────────────────────
    w_mix = torch.sigmoid(-log_n2)  # high N² → small weight
    L_strat = (w_mix.detach() * dC_dsig**2).mean()

    # ── L5: along-stream tracer gradient  ← NEW ──────────────────────────────
    #
    # Physical basis: u · ∇_h C ≈ 0  for conservative tracer in steady state.
    #
    # Implementation:
    #   along_grad = û · ∂C/∂lon_norm + v̂ · ∂C/∂lat_norm
    #
    # where û = u/|u|, v̂ = v/|u| are the unit-vector components.
    #
    # The penalty is weighted by w_flow = speed / (SPEED_SCALE + speed) so
    # that:
    #   • Strong currents (boundary currents, jets): w_flow ≈ 1 → tight constraint
    #   • Weak flow (gyre interior, deep water): w_flow ≈ 0 → relaxed constraint
    #
    # We use the RAW u, v (passed in from the DataLoader, not from x_batch)
    # so the unit vectors are in physical m/s, not standardised space.
    # This avoids the scaler distorting the direction information.
    # ─────────────────────────────────────────────────────────────────────────

    speed_raw = torch.sqrt(u_raw**2 + v_raw**2 + eps)

    # Unit velocity components
    u_hat = u_raw / speed_raw  # cos θ
    v_hat = v_raw / speed_raw  # sin θ

    # Along-stream component of the tracer gradient (in normalised lon/lat space)
    along_grad = u_hat * dC_dlon + v_hat * dC_dlat

    # Speed-based weighting: saturates at 1 for fast currents
    w_flow = speed_raw / (SPEED_SCALE + speed_raw)

    L_advect = (w_flow.detach() * along_grad**2).mean()

    return L_diapycnal, L_smooth, L_secular, L_strat, L_advect


class TracerDataset(torch.utils.data.Dataset):
    """
    Wraps (X_scaled, y, u_raw, v_raw) so the DataLoader can serve velocity
    alongside features without shuffling them out of sync.
    """

    def __init__(self, X, y, u, v):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.u = torch.tensor(u, dtype=torch.float32)
        self.v = torch.tensor(v, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.u[i], self.v[i]


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
        val_idx = np.where((years >= cut) & (years < val_end))[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits


def _compute_log_n2(df, cast_col):
    """Thin wrapper — delegates to compute_n2_per_cast from pinn_tracer_N2."""
    try:
        from pinn_tracer_N2 import compute_n2_per_cast

        return compute_n2_per_cast(df, cast_col=cast_col)
    except ImportError:
        # Fallback: per-point proxy if N2 module not available
        warnings.warn("pinn_tracer_N2 not found — using per-point N² proxy.")
        SA = gsw.SA_from_SP(
            df["salinity"].values,
            df["depth"].values,
            df["lon"].values,
            df["lat"].values,
        )
        CT = gsw.CT_from_t(SA, df["temperature"].values, df["depth"].values)
        p = gsw.p_from_z(-np.abs(df["depth"].values), df["lat"].values)
        alpha = gsw.alpha(SA, CT, p)
        beta = gsw.beta(SA, CT, p)
        rho = gsw.rho(SA, CT, p)
        n2 = np.maximum(
            9.7963 / rho * rho * (alpha + beta) / REF_DEPTH_SCALE, N2_EPSILON
        )
        return np.log10(n2)


# ── 5. TRAINING LOOP ───────────────────────────────────────────────────────────


def train_pinn(
    model,
    X_tr,
    y_tr,
    X_val,
    y_val,
    feat_names,
    scaler,
    u_tr=None,
    v_tr=None,  # ← NEW: raw velocities for training
    u_val=None,
    v_val=None,  # ← NEW: raw velocities for validation
    n_epochs=300,
    batch_size=512,
    lr=5e-4,
    lambda_diap=0.05,
    lambda_smooth=0.02,
    lambda_sec=0.01,
    lambda_strat=0.05,
    lambda_advect=0.03,  # ← NEW
    device="cpu",
):
    """
    Training loop with 5-term physics loss.

    If u_tr / v_tr are None the advection loss is skipped (backward compatible
    with datasets that lack velocity).
    """
    has_velocity = (u_tr is not None) and (v_tr is not None)

    if not has_velocity and lambda_advect > 0:
        print("  [train] No velocity data supplied — advection loss will be skipped.")

    X_tr_sc = scaler.transform(X_tr)
    X_val_sc = scaler.transform(X_val)

    # Use TracerDataset if velocity present, plain TensorDataset otherwise
    if has_velocity:
        tr_ds = TracerDataset(X_tr_sc, y_tr, u_tr, v_tr)
    else:
        from torch.utils.data import TensorDataset

        tr_ds = TensorDataset(
            torch.tensor(X_tr_sc, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )

    loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val_sc, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=len(loader), pct_start=0.1
    )
    huber = nn.HuberLoss(delta=1.0)
    history = []

    for epoch in range(n_epochs):
        model.train()
        e_data = e_phys = 0.0

        for batch in loader:
            if has_velocity:
                Xb, yb, ub, vb = batch
                ub = ub.to(device)
                vb = vb.to(device)
            else:
                Xb, yb = batch
                ub = vb = None

            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()

            # Data loss
            yp = model(Xb)
            L_dat = huber(yp, yb)

            # Physics losses
            Ld, Ls, Lt, Lstr, Ladv = physics_losses(
                model,
                Xb,
                feat_names,
                u_raw=ub if has_velocity else torch.zeros_like(yb),
                v_raw=vb if has_velocity else torch.zeros_like(yb),
            )

            L_phys = (
                lambda_diap * Ld
                + lambda_smooth * Ls
                + lambda_sec * Lt
                + lambda_strat * Lstr
                + (lambda_advect * Ladv if has_velocity else 0.0)
            )

            (L_dat + L_phys).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            e_data += L_dat.item()
            e_phys += L_phys.item()

        # Validation
        model.eval()
        with torch.no_grad():
            yv_pred = model(X_val_t.to(device)).cpu()
            rmse = torch.sqrt(((yv_pred - y_val_t) ** 2).mean()).item()
            r2 = (
                1
                - ((yv_pred - y_val_t) ** 2).sum().item()
                / ((y_val_t - y_val_t.mean()) ** 2).sum().item()
            )

        history.append(
            {
                "epoch": epoch,
                "data_loss": e_data / len(loader),
                "phys_loss": e_phys / len(loader),
                "val_rmse": rmse,
                "val_r2": r2,
            }
        )

        if epoch % 50 == 0:
            print(
                f"  [{epoch:3d}] data={e_data / len(loader):.4f}  "
                f"phys={e_phys / len(loader):.5f}  "
                f"val RMSE={rmse:.4f}  R²={r2:.3f}"
            )

    return model, pd.DataFrame(history)


# ── 6. MC-DROPOUT UNCERTAINTY ──────────────────────────────────────────────────
#
# Enable dropout at inference time → sample N forward passes →
# mean = prediction, std = epistemic uncertainty
# Regions with no training analogues (deep water, poles) will show high σ


def predict_with_uncertainty(
    model, X_scaled, n_samples=100, batch_size=100_000, device="cpu"
):

    model.train()  # keep dropout ON

    n = X_scaled.shape[0]
    preds_mean = np.zeros(n, dtype=np.float32)
    preds_sq = np.zeros(n, dtype=np.float32)

    for i in range(0, n, batch_size):
        X_batch = torch.tensor(X_scaled[i : i + batch_size], dtype=torch.float32).to(
            device
        )

        batch_preds = []

        for _ in range(n_samples):
            with torch.no_grad():
                y = model(X_batch).cpu().numpy()
            batch_preds.append(y)

        batch_preds = np.stack(batch_preds)

        preds_mean[i : i + batch_size] = batch_preds.mean(axis=0)
        preds_sq[i : i + batch_size] = (batch_preds**2).mean(axis=0)

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
    cv_r2 = []
    for fold, (tr_idx, val_idx) in enumerate(splits):
        model_cv = TracerPINN(n_features=len(feat_names))
        scaler_cv = StandardScaler().fit(X[tr_idx])
        model_cv, hist = train_pinn(
            model_cv,
            X[tr_idx],
            y[tr_idx],
            X[val_idx],
            y[val_idx],
            feat_names,
            scaler_cv,
            n_epochs=n_epochs,
            device=device,
        )
        cv_r2.append(hist["val_r2"].iloc[-1])
        print(f"Fold {fold + 1}  val R²={cv_r2[-1]:.3f}")
    print(f"\nCV R² = {np.mean(cv_r2):.3f} ± {np.std(cv_r2):.3f}")

    # Final model: train on all data
    scaler = StandardScaler().fit(X)
    model_final = TracerPINN(n_features=len(feat_names)).to(device)
    model_final, history = train_pinn(
        model_final,
        X,
        y,
        X,
        y,  # val=train for final monitoring
        feat_names,
        scaler,
        n_epochs=n_epochs,
        device=device,
    )

    return model_final, scaler, feat_names, history


# ── 8. INFERENCE ON OCEAN MODEL ───────────────────────────────────────────────


def infer_on_model_field(
    ds,
    model,
    scaler,
    feat_names,
    target_year=2020,
    target_month=6,
    target_time_index=0,
    temp_var=None,
    salt_var=None,
    uvel_var=None,
    vvel_var=None,  # ← NEW
    device="cpu",
):
    """
    Run the trained PINN on a gridded ocean model Dataset.

    Extends the original infer_on_model_field to read u and v from the
    NetCDF and compute the velocity features required by the network.

    Additional parameters
    ─────────────────────
    uvel_var : str  name of the u-velocity variable (default: auto-detect)
    vvel_var : str  name of the v-velocity variable (default: auto-detect)
    """
    import xarray as xr

    # ── Candidate variable names ───────────────────────────────────────────────THETA', 'SALT', 'EVEL', 'NVEL', 'WVEL'
    _TEMP_VARS = ["temp", "temperature", "thetao", "votemper", "potential_temperature","THETA"]
    _SALT_VARS = ["salinity", "salt", "so", "vosaline", "practical_salinity","SALT"]
    _UVEL_VARS = [
        # "u",
        # "uo",
        "vxo",
        "EVEL",
        # "vozocrtx",
        # "u_velocity",
        # "UVEL",
        # "eastward_sea_water_velocity",
    ]
    _VVEL_VARS = [
        # "v",
        # "vo",
        "vyo",
        "NVEL",
        # "vomecrty",
        # "v_velocity",
        # "VVEL",
        # "northward_sea_water_velocity",
    ]
    _DEPTH_NAMES = ["depth", "deptht", "depthu", "depthv", "Z","z", "lev", "level"]
    _LAT_NAMES = ["lat", "latitude", "nav_lat", "yt_ocean", "nlat", "y"]
    _LON_NAMES = ["lon", "longitude", "nav_lon", "xt_ocean", "nlon", "x"]
    _TIME_NAMES = ["time", "time_counter", "t", "time_0", "time_centered"]

    def _find_var(ds, candidates, label, override):
        if override and override in ds:
            return override
        low = {v.lower(): v for v in ds.data_vars}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        raise ValueError(
            f"Cannot find {label} variable. "
            f"Available: {list(ds.data_vars)}. "
            f"Pass {label.lower()}_var='your_name' to override."
        )

    def _find_dim(da, candidates, label):
        low = {d.lower(): d for d in da.dims}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        raise ValueError(f"Cannot find {label} dimension in {list(da.dims)}.")

    def _to_3d(da, dep_dim, lat_dim, lon_dim):
        """Squeeze extra dims then transpose to (D, Y, X)."""
        extra = [d for d in da.dims if d not in {dep_dim, lat_dim, lon_dim}]
        for d in extra:
            da = da.isel({d: 0}) if da.sizes[d] > 1 else da.squeeze(d)
        return da.transpose(dep_dim, lat_dim, lon_dim).values

    # ── Variable detection ────────────────────────────────────────────────────
    temp_var = _find_var(ds, _TEMP_VARS, "temperature", temp_var)
    salt_var = _find_var(ds, _SALT_VARS, "salinity", salt_var)
    uvel_var = _find_var(ds, _UVEL_VARS, "u-velocity", uvel_var)
    vvel_var = _find_var(ds, _VVEL_VARS, "v-velocity", vvel_var)
    print(
        f"  [infer] temp='{temp_var}', salt='{salt_var}', "
        f"u='{uvel_var}', v='{vvel_var}'"
    )

    # ── Dimension detection ───────────────────────────────────────────────────
    ref = ds[temp_var]
    dep_dim = _find_dim(ref, _DEPTH_NAMES, "depth")
    lat_dim = _find_dim(ref, _LAT_NAMES, "latitude")
    lon_dim = _find_dim(ref, _LON_NAMES, "longitude")

    # Handle time
    time_dim = next((d for d in _TIME_NAMES if d in ref.dims), None)
    if time_dim and ref.sizes[time_dim] > 1:
        print(
            f"  [infer] Slicing time index {target_time_index} "
            f"from dim '{time_dim}' (size {ref.sizes[time_dim]})"
        )
        ds = ds.isel({time_dim: target_time_index})
        ref = ds[temp_var]

    # ── Extract 3-D arrays ────────────────────────────────────────────────────
    temp_3d = _to_3d(ds[temp_var], dep_dim, lat_dim, lon_dim)
    salt_3d = _to_3d(ds[salt_var], dep_dim, lat_dim, lon_dim)
    u_3d = _to_3d(ds[uvel_var], dep_dim, lat_dim, lon_dim)
    v_3d = _to_3d(ds[vvel_var], dep_dim, lat_dim, lon_dim)

    n_dep, n_lat, n_lon = temp_3d.shape

    # Depth, lat, lon grids
    depth_vals = (
        np.abs(ds.coords[dep_dim].values)
        if dep_dim in ds.coords
        else np.arange(n_dep, dtype=float)
    )
    lat_1d = ds.coords[lat_dim].values if lat_dim in ds.coords else np.zeros(n_lat)
    lon_1d = ds.coords[lon_dim].values if lon_dim in ds.coords else np.zeros(n_lon)

    if lat_1d.ndim == 2:  # curvilinear grid
        lat_3d = np.broadcast_to(lat_1d[np.newaxis], temp_3d.shape).copy()
        lon_3d = np.broadcast_to(lon_1d[np.newaxis], temp_3d.shape).copy()
    else:
        lat_3d = np.broadcast_to(
            lat_1d[np.newaxis, :, np.newaxis], temp_3d.shape
        ).copy()
        lon_3d = np.broadcast_to(
            lon_1d[np.newaxis, np.newaxis, :], temp_3d.shape
        ).copy()

    depth_3d = np.broadcast_to(
        depth_vals[:, np.newaxis, np.newaxis], temp_3d.shape
    ).copy()

    # 2. Define the Target Polar Stereographic Projection (EPSG:3413)
    # Central meridian (lon_0) for EPSG:3413 is -45 degrees (Greenland)
    lon_0 = -45.0 
    
    # Define coordinate transformer (WGS84 lat-lon to NSIDC Polar Stereographic North)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    x_grid, y_grid = transformer.transform(lon_3d, lat_3d)
    
    # 3. Calculate the Grid Convergence Angle (gamma)
    # Convert angles to radians for numpy trigonometric functions
    gamma = np.radians(lon_3d - lon_0)
    
    # 4. Perform the Vector Rotation
    u_3d = u_3d * np.cos(gamma) - v_3d * np.sin(gamma)
    v_3d = u_3d * np.sin(gamma) + v_3d * np.cos(gamma)

    # ── Flatten ───────────────────────────────────────────────────────────────
    T_flat = temp_3d.ravel()
    S_flat = salt_3d.ravel()
    z_flat = depth_3d.ravel()
    lat_f = lat_3d.ravel()
    lon_f = lon_3d.ravel()
    u_flat = u_3d.ravel()
    v_flat = v_3d.ravel()

    # # rotate velocities
    # X, Y = u_flat, v_flat
    # X_src_crs = X / np.cos(lat_f/ 180 * np.pi)
    # Y_src_crs = Y
    # magnitude = np.sqrt(X**2 + Y**2)
    # magn_src_crs = np.sqrt(X_src_crs**2 + Y_src_crs**2)
    # u_flat = X_src_crs * magnitude / magn_src_crs
    # v_flat = Y_src_crs * magnitude / magn_src_crs
        

    # ── TEOS-10 ───────────────────────────────────────────────────────────────
    SA = gsw.SA_from_SP(S_flat, z_flat, lon_f, lat_f)
    CT = gsw.CT_from_t(SA, T_flat, z_flat)
    p_flat = gsw.p_from_z(-z_flat, lat_f)
    sigma0 = gsw.sigma0(SA, CT)
    spice = gsw.spiciness0(SA, CT)
    sigma2 = gsw.sigma2(SA, CT)

    # N² proxy (per-point; no cast structure in gridded data)
    alpha = gsw.alpha(SA, CT, p_flat)
    beta = gsw.beta(SA, CT, p_flat)
    rho = gsw.rho(SA, CT, p_flat)
    n2_proxy = np.maximum(9.7963 / rho * rho * (alpha + beta) / 100.0, 1e-8)
    log_n2 = np.log10(n2_proxy)

    # ── Velocity features ─────────────────────────────────────────────────────
    eps_v = 1e-10
    speed = np.sqrt(u_flat**2 + v_flat**2)
    flow_sin = np.where(speed > eps_v, v_flat / (speed + eps_v), 0.0)
    flow_cos = np.where(speed > eps_v, u_flat / (speed + eps_v), 0.0)
    log_speed = np.log1p(speed)

    # ── Scalar temporal/seasonal features ────────────────────────────────────
    n_pts = T_flat.shape[0]
    year_norm = (target_year - 1970) / 53.0
    season_sin = np.sin(2 * np.pi * target_month / 12)
    season_cos = np.cos(2 * np.pi * target_month / 12)

    # ── Feature map — must match training order ───────────────────────────────
    feature_map = {
        "sigma0": sigma0,
        "spice": spice,
        "sigma2": sigma2,
        "log_depth": np.log1p(z_flat),
        "log_n2": log_n2,
        "year_norm": np.full(n_pts, year_norm),
        "season_sin": np.full(n_pts, season_sin),
        "season_cos": np.full(n_pts, season_cos),
        "lat_norm": np.sin(np.deg2rad(lat_f)),
        "lon_sin": np.sin(np.deg2rad(lon_f)),
        "lon_cos": np.cos(np.deg2rad(lon_f)),
        "log_speed": log_speed,
        "flow_sin": flow_sin,
        "flow_cos": flow_cos,
    }

    missing = [f for f in feat_names if f not in feature_map]
    if missing:
        raise ValueError(
            f"Features {missing} not computed. "
            f"Add them to feature_map in infer_on_model_field."
        )

    X_model = np.column_stack([feature_map[f] for f in feat_names])

    # ── Mask land / ice ───────────────────────────────────────────────────────
    valid = np.isfinite(X_model).all(axis=1)
    n_valid = valid.sum()
    print(
        f"  [infer] Valid ocean points: {n_valid:,} / {valid.size:,} "
        f"({100 * n_valid / valid.size:.1f}%)"
    )

    X_valid = scaler.transform(X_model[valid])

    # ── MC-Dropout inference ──────────────────────────────────────────────────
    model.train()

    batch_size = 100_000  # tune this (start lower if needed)
    n_mc = 20  # reduce from 50 → big win

    n = X_valid.shape[0]
    mean = np.zeros(n, dtype=np.float32)
    sq_mean = np.zeros(n, dtype=np.float32)

    for i in range(0, n, batch_size):
        X_batch_np = X_valid[i : i + batch_size]

        X_batch = torch.tensor(X_batch_np, dtype=torch.float32).to(device)

        mc_preds = []

        for _ in range(n_mc):
            with torch.no_grad(), torch.cuda.amp.autocast():
                y = model(X_batch).cpu().numpy()
            mc_preds.append(y)

        mc_preds = np.stack(mc_preds)  # (n_mc, batch)

        mean[i : i + batch_size] = mc_preds.mean(axis=0)
        sq_mean[i : i + batch_size] = (mc_preds**2).mean(axis=0)

        del X_batch, mc_preds
        torch.cuda.empty_cache()

    tracer_mean = mean
    tracer_std = np.sqrt(sq_mean - mean**2)

    tracer_field = np.full(valid.size, np.nan)
    uncert_field = np.full(valid.size, np.nan)
    tracer_field[valid] = tracer_mean
    uncert_field[valid] = tracer_std

    # ── Output Dataset ────────────────────────────────────────────────────────
    out_dims = (dep_dim, lat_dim, lon_dim)
    out_coords = {
        dep_dim: ds.coords.get(dep_dim, np.arange(n_dep)),
        lat_dim: ds.coords.get(lat_dim, np.arange(n_lat)),
        lon_dim: ds.coords.get(lon_dim, np.arange(n_lon)),
    }

    import xarray as xr

    ds_out = xr.Dataset(
        {
            "tracer_pred": (out_dims, tracer_field.reshape(n_dep, n_lat, n_lon)),
            "tracer_uncertainty": (out_dims, uncert_field.reshape(n_dep, n_lat, n_lon)),
        },
        coords=out_coords,
        attrs={
            "description": "PINN tracer inference with velocity loss",
            "target_year": target_year,
            "target_month": target_month,
            "temp_var_used": temp_var,
            "salt_var_used": salt_var,
            "uvel_var_used": uvel_var,
            "vvel_var_used": vvel_var,
        },
    )
    return ds_out


# ── 9. DIAGNOSTICS ─────────────────────────────────────────────────────────────


def plot_ts_diagram(df, feat_names, y_pred):
    """T-S diagram coloured by tracer — key sanity check.
    Predictions should follow isopycnal contours for a conservative tracer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sc1 = ax1.scatter(
        df["salinity"], df["temperature"], c=df["tracer"], cmap="plasma", s=5, alpha=0.6
    )
    ax1.set(
        xlabel="Salinity",
        ylabel="Temperature (°C)",
        title="Observed tracer on T-S diagram",
    )
    plt.colorbar(sc1, ax=ax1, label="Tracer")

    sc2 = ax2.scatter(
        df["salinity"], df["temperature"], c=y_pred, cmap="plasma", s=5, alpha=0.6
    )
    ax2.set(
        xlabel="Salinity",
        ylabel="Temperature (°C)",
        title="Predicted tracer on T-S diagram",
    )
    plt.colorbar(sc2, ax=ax2, label="Tracer")

    plt.tight_layout()
    plt.savefig("ts_diagram_check.png", dpi=150)


def plot_depth_residuals(y_true, y_pred, depths):
    """Residuals should be white noise vs depth — structure means missing physics."""
    resid = y_pred - y_true
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(resid, depths, s=3, alpha=0.3)
    axes[0].axvline(0, color="r")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()
    axes[0].set_title("Residual vs Depth — should be centred")

    axes[1].scatter(y_true, y_pred, s=3, alpha=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, "r--")
    axes[1].set_xlabel("Observed")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("1:1 plot")

    axes[2].hist(resid, bins=60, edgecolor="k")
    axes[2].set_xlabel("Residual")
    axes[2].set_title("Residual distribution")
    axes[2].set_xlabel("Residual")
    axes[2].set_title("Residual distribution")

    plt.tight_layout()
    plt.savefig("residual_diagnostics.png", dpi=150)
