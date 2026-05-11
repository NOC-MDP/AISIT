"""
pinn_tracer_ssh_sic.py  —  adds SSH-contour alignment (L6) and sea-ice
concentration (L7) physics losses to the PINN tracer pipeline.

NEW OBSERVATIONS REQUIRED
─────────────────────────
Your CSV must now include:

  ssh      Sea surface height (m), e.g. from AVISO/CMEMS climatology.
           Used as a feature (captures dynamic height / mesoscale structure)
           and its gradient drives L6.

  deta_dx  Eastward SSH gradient component (m deg⁻¹ lon).
  deta_dy  Northward SSH gradient component (m deg⁻¹ lat).
           Compute once from your SSH climatology via finite difference:
               deta_dx = np.gradient(ssh_field, lon_axis, axis=-1)
               deta_dy = np.gradient(ssh_field, lat_axis, axis=-2)
           then interpolate to observation locations.
           These are NOT added as network features — they are raw physical
           values passed only to the loss function (same pattern as u/v).

  sic      Sea ice concentration (0–1 fraction), e.g. from OSI-SAF or NSIDC.
           Added as a feature AND used raw in L7.

PHYSICAL MOTIVATION
───────────────────
L6  SSH contour alignment
    Geostrophic balance: u_geo = -(g/f) ∂η/∂y,  v_geo = (g/f) ∂η/∂x
    Flow is perpendicular to ∇η, so SSH contours ARE the streamlines.
    A conservative tracer is constant along streamlines, so ∇C ∥ ∇η.
    We penalise the component of ∇C that is tangent to SSH contours:

        tangent direction  = (-∂η/∂y, ∂η/∂x) / |∇η|   (rotated 90° from ∇η)
        L_ssh = mean( w_ssh · (τ̂ · ∇_h C)² )
        w_ssh = |∇η| / (|∇η|_scale + |∇η|)

    Weighting by |∇η| concentrates the constraint where SSH fronts are
    sharp (boundary currents, ACC, mesoscale eddies) and relaxes it in
    the sluggish gyre interior.

    NOTE: this is complementary to L5 (advection loss), not redundant.
    L5 uses full u/v (geostrophic + ageostrophic + tidal residuals).
    L6 uses only the geostrophic streamline direction derived from SSH.
    Both can be active simultaneously.

L7  Sea-ice concentration
    Under sea ice:
      • Wind mixing is suppressed → no surface-driven homogenisation.
      • Air–sea tracer exchange is blocked → tracer decoupled from atmosphere.
      • The ice-covered mixed layer acts as an isolated reservoir.
    Result: under high SIC, horizontal tracer gradients should be small.

        L_ice = mean( w_ice · (|∂C/∂lon|² + |∂C/∂lat|²) )
        w_ice = sic_raw²   (quadratic: punishes near-100% ice strongly)

    The quadratic weight means marginal ice (<50%) gets only ~25% of the
    penalty of full ice cover — physically reasonable since partial ice
    still allows some wind mixing through leads.

INTEGRATION
───────────
Drop this file alongside pinn_tracer_velocity.py and update imports in
main.py (see bottom of file for the updated train call signature).
"""

import numpy as np
import pandas as pd
import gsw
import torch
import torch.nn as nn
import warnings


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SPEED_SCALE   = 0.1      # m/s    — advection loss saturation speed
SSH_GRAD_SCALE = 0.01    # m/deg  — SSH gradient loss saturation scale
                          # typical open-ocean |∇η| ≈ 0.005–0.05 m/deg;
                          # boundary currents reach ~0.1 m/deg


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  UPDATED build_features
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(df, cast_col="cast_id"):
    """
    Build the full physics-informed feature matrix.

    Required DataFrame columns
    --------------------------
    temp, salinity, depth, lon, lat, year, month, tracer
    u, v          eastward/northward velocity (m/s)
    ssh           sea surface height (m)                    ← NEW
    sic           sea ice concentration (0–1)               ← NEW

    deta_dx, deta_dy are NOT features — pass them as raw arrays to
    physics_losses separately (same pattern as u, v).

    Optional
    --------
    cast_col  profile identifier for per-cast N² computation.
    """
    # ── Velocity check ────────────────────────────────────────────────────────
    for col in ("u", "v"):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' missing. Velocity components u and v required."
            )

    # ── SSH / SIC checks ──────────────────────────────────────────────────────
    has_ssh = "ssh" in df.columns
    has_sic = "sic" in df.columns

    if not has_ssh:
        warnings.warn(
            "Column 'ssh' not found — SSH feature and L6 loss will be skipped. "
            "Provide co-located SSH (m) from AVISO/CMEMS for best results."
        )
    if not has_sic:
        warnings.warn(
            "Column 'sic' not found — SIC feature and L7 loss will be skipped. "
            "Provide co-located sea ice concentration (0–1) from OSI-SAF/NSIDC."
        )

    # ── TEOS-10 ───────────────────────────────────────────────────────────────
    SA     = gsw.SA_from_SP(df["salinity"].values, df["depth"].values,
                             df["lon"].values,      df["lat"].values)
    CT     = gsw.CT_from_t(SA, df["temp"].values, df["depth"].values)
    sigma0 = gsw.sigma0(SA, CT)
    spice  = gsw.spiciness0(SA, CT)
    sigma2 = gsw.sigma2(SA, CT)

    # ── Stratification ────────────────────────────────────────────────────────
    log_n2 = _compute_log_n2(df, cast_col)

    # ── Temporal ──────────────────────────────────────────────────────────────
    year_norm  = (df["year"].values - 1970) / 53.0
    season_sin = np.sin(2 * np.pi * df["month"].values / 12)
    season_cos = np.cos(2 * np.pi * df["month"].values / 12)

    # ── Spatial ───────────────────────────────────────────────────────────────
    log_depth = np.log1p(np.abs(df["depth"].values))
    lon_norm  = df["lon"].values / 180.0
    lat_norm  = df["lat"].values /  90.0

    # ── Velocity ──────────────────────────────────────────────────────────────
    u = df["u"].values;  v = df["v"].values
    speed     = np.sqrt(u ** 2 + v ** 2)
    eps       = 1e-10
    flow_sin  = np.where(speed > eps, v / (speed + eps), 0.0)
    flow_cos  = np.where(speed > eps, u / (speed + eps), 0.0)
    log_speed = np.log1p(speed)

    # ── SSH  ← NEW ────────────────────────────────────────────────────────────
    # Normalise by 1 m (typical mesoscale SSH amplitude).
    # The StandardScaler will further standardise this, but pre-normalising
    # keeps the raw feature in a sensible range before fitting.
    ssh_norm = df["ssh"].values / 1.0 if has_ssh else np.zeros(len(df))

    # ── SIC  ← NEW ────────────────────────────────────────────────────────────
    # Already in [0, 1]; clip to guard against data quality issues.
    sic_feat = (np.clip(df["sic"].values, 0.0, 1.0)
                if has_sic else np.zeros(len(df)))

    # ── Assemble ──────────────────────────────────────────────────────────────
    feature_names = [
        "sigma0", "spice", "sigma2",
        "log_depth", "log_n2",
        "year_norm", "season_sin", "season_cos",
        "lon_norm", "lat_norm",
        "log_speed", "flow_sin", "flow_cos",
        "ssh_norm",                              # ← NEW
        "sic",                                   # ← NEW
    ]

    X = np.column_stack([
        sigma0, spice, sigma2,
        log_depth, log_n2,
        year_norm, season_sin, season_cos,
        lon_norm, lat_norm,
        log_speed, flow_sin, flow_cos,
        ssh_norm,
        sic_feat,
    ])

    valid = np.isfinite(X).all(axis=1) & np.isfinite(df["tracer"].values)

    if has_ssh:
        print(f"  SSH range   : {ssh_norm[valid].min():.3f} – "
              f"{ssh_norm[valid].max():.3f} m")
    if has_sic:
        ice_frac = (sic_feat[valid] > 0.15).mean() * 100
        print(f"  SIC > 15%%  : {ice_frac:.1f}%% of valid obs")

    return X[valid], df["tracer"].values[valid], feature_names, valid


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  UPDATED physics_losses  (7 terms)
# ═══════════════════════════════════════════════════════════════════════════════

def physics_losses(model, x_batch, feature_names,
                   u_raw, v_raw,
                   ssh_grad_x_raw, ssh_grad_y_raw,   # ← NEW: SSH gradient
                   sic_raw):                           # ← NEW: sea ice conc.
    """
    Seven physics-informed loss terms.

    Parameters
    ----------
    model            : TracerPINN (training mode — dropout active)
    x_batch          : (B, F) standardised feature tensor, requires_grad will
                       be set internally
    feature_names    : list[str] in training order
    u_raw            : (B,) tensor, eastward velocity [m/s]
    v_raw            : (B,) tensor, northward velocity [m/s]
    ssh_grad_x_raw   : (B,) tensor, ∂η/∂lon [m deg⁻¹]  ← NEW
    ssh_grad_y_raw   : (B,) tensor, ∂η/∂lat [m deg⁻¹]  ← NEW
    sic_raw          : (B,) tensor, sea ice concentration [0–1]  ← NEW

    Returns
    -------
    L_diapycnal, L_smooth, L_secular, L_strat, L_advect,
    L_ssh, L_ice
    """
    idx = {name: feature_names.index(name) for name in feature_names}
    eps = 1e-6

    x_batch = x_batch.clone().requires_grad_(True)
    C       = model(x_batch)
    grads   = torch.autograd.grad(C.sum(), x_batch,
                                   create_graph=True)[0]

    dC_dsig = grads[:, idx["sigma0"]]
    dC_dspi = grads[:, idx["spice"]]
    dC_dyr  = grads[:, idx["year_norm"]]
    dC_dlon = grads[:, idx["lon_norm"]]
    dC_dlat = grads[:, idx["lat_norm"]]
    log_n2  = x_batch[:, idx["log_n2"]]

    # ── L1: diapycnal / along-isopycnal ratio ─────────────────────────────────
    L_diapycnal = (dC_dspi ** 2 / (dC_dsig.detach() ** 2 + eps)).mean()

    # ── L2: second-order smoothness in σ₀ ─────────────────────────────────────
    d2C_dsig2 = torch.autograd.grad(
        dC_dsig.sum(), x_batch,
        create_graph=True, retain_graph=True
    )[0][:, idx["sigma0"]]
    L_smooth = (d2C_dsig2 ** 2).mean()

    # ── L3: temporal smoothness ───────────────────────────────────────────────
    L_secular = (torch.clamp(dC_dyr.abs() - 3.0, min=0) ** 2).mean()

    # ── L4: stratification-weighted diapycnal penalty ─────────────────────────
    w_mix   = torch.sigmoid(-log_n2)           # high N² → small weight
    L_strat = (w_mix.detach() * dC_dsig ** 2).mean()

    # ── L5: along-stream tracer gradient (velocity advection) ─────────────────
    speed_raw  = torch.sqrt(u_raw ** 2 + v_raw ** 2 + eps)
    u_hat      = u_raw / speed_raw
    v_hat      = v_raw / speed_raw
    along_grad = u_hat * dC_dlon + v_hat * dC_dlat
    w_flow     = speed_raw / (SPEED_SCALE + speed_raw)
    L_advect   = (w_flow.detach() * along_grad ** 2).mean()

    # ── L6: SSH contour alignment  ← NEW ─────────────────────────────────────
    #
    # Geostrophic flow is perpendicular to ∇η, so SSH contours are streamlines.
    # Conservative tracer → constant along streamlines → ∇C ∥ ∇η.
    # Penalty: component of ∇C tangent to SSH contours.
    #
    # SSH contour tangent direction (unit vector, rotated 90° from ∇η):
    #     τ̂ = (-∂η/∂y, ∂η/∂x) / |∇η|
    #
    # Along-contour tracer gradient:
    #     along_ssh = τ̂_x · ∂C/∂lon + τ̂_y · ∂C/∂lat
    #              = (-∂η/∂y · ∂C/∂lon + ∂η/∂x · ∂C/∂lat) / |∇η|
    #
    # Weight by |∇η| so that sharp SSH fronts enforce the constraint strongly.
    # The w_ssh weight saturates at 1 for |∇η| >> SSH_GRAD_SCALE.
    # ─────────────────────────────────────────────────────────────────────────
    ssh_grad_mag = torch.sqrt(
        ssh_grad_x_raw ** 2 + ssh_grad_y_raw ** 2 + eps
    )
    # Unit contour tangent: τ̂ = (-∂η/∂y, ∂η/∂x) / |∇η|
    tau_x = -ssh_grad_y_raw / ssh_grad_mag    # eastward component of τ̂
    tau_y =  ssh_grad_x_raw / ssh_grad_mag    # northward component of τ̂

    along_ssh_contour = tau_x * dC_dlon + tau_y * dC_dlat

    # Saturating weight — concentrate constraint where SSH fronts are strong
    w_ssh = ssh_grad_mag / (SSH_GRAD_SCALE + ssh_grad_mag)

    L_ssh = (w_ssh.detach() * along_ssh_contour ** 2).mean()

    # ── L7: sea-ice concentration horizontal gradient suppression  ← NEW ──────
    #
    # Under sea ice: wind mixing suppressed, air-sea exchange blocked →
    # horizontal tracer gradients should be small.
    #
    # Penalty: sum of squared horizontal gradients, weighted by SIC².
    # Quadratic weight: 100% ice → full penalty; 50% ice → 25% penalty;
    # 0% ice → no penalty (open ocean: unconstrained horizontal gradients).
    #
    # We use lon/lat gradients (already in normalised space via autograd).
    # Note: this is a SURFACE constraint. For sub-surface obs, the ice above
    # still reduces horizontal mixing through its effect on the mixed layer.
    # The loss correctly applies to all depths since it is data-driven — if
    # your obs are deep and unaffected by surface ice, the model will learn
    # this from the data loss overriding the (small) physics penalty.
    # ─────────────────────────────────────────────────────────────────────────
    w_ice  = sic_raw ** 2                     # quadratic: 0→0, 0.5→0.25, 1→1
    L_ice  = (w_ice.detach() * (dC_dlon ** 2 + dC_dlat ** 2)).mean()

    return L_diapycnal, L_smooth, L_secular, L_strat, L_advect, L_ssh, L_ice


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  UPDATED TracerDataset  (carries all raw physical fields)
# ═══════════════════════════════════════════════════════════════════════════════

class TracerDataset(torch.utils.data.Dataset):
    """
    Serves (X_scaled, y, u, v, deta_dx, deta_dy, sic) per batch,
    keeping all raw physical arrays in sync with shuffled feature rows.
    """
    def __init__(self, X, y, u, v, deta_dx, deta_dy, sic):
        self.X       = torch.tensor(X,       dtype=torch.float32)
        self.y       = torch.tensor(y,       dtype=torch.float32)
        self.u       = torch.tensor(u,       dtype=torch.float32)
        self.v       = torch.tensor(v,       dtype=torch.float32)
        self.deta_dx = torch.tensor(deta_dx, dtype=torch.float32)
        self.deta_dy = torch.tensor(deta_dy, dtype=torch.float32)
        self.sic     = torch.tensor(sic,     dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.X[i], self.y[i],
                self.u[i], self.v[i],
                self.deta_dx[i], self.deta_dy[i],
                self.sic[i])


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  UPDATED train_pinn  (7-term physics loss)
# ═══════════════════════════════════════════════════════════════════════════════

def train_pinn(model, X_tr, y_tr, X_val, y_val,
               feat_names, scaler,
               # Raw physical arrays (training set)
               u_tr=None,       v_tr=None,
               deta_dx_tr=None, deta_dy_tr=None,
               sic_tr=None,
               # Raw physical arrays (validation set — for consistent signature)
               u_val=None,       v_val=None,
               deta_dx_val=None, deta_dy_val=None,
               sic_val=None,
               # Hyperparameters
               n_epochs=300, batch_size=512, lr=5e-4,
               lambda_diap=0.05,    lambda_smooth=0.02,
               lambda_sec=0.01,     lambda_strat=0.05,
               lambda_advect=0.03,
               lambda_ssh=0.04,     # ← NEW
               lambda_ice=0.03,     # ← NEW
               device="cpu"):
    """
    Training loop with 7-term physics loss.

    Graceful degradation
    --------------------
    Each physical array is optional.  If absent the corresponding loss
    weight is forced to zero so the remaining terms are unaffected:

        u/v absent       → lambda_advect = 0
        deta_dx/dy absent → lambda_ssh    = 0
        sic absent        → lambda_ice    = 0
    """
    has_vel = (u_tr is not None) and (v_tr is not None)
    has_ssh = (deta_dx_tr is not None) and (deta_dy_tr is not None)
    has_sic = sic_tr is not None

    if not has_vel and lambda_advect > 0:
        print("  [train] No velocity — advection loss skipped.")
    if not has_ssh and lambda_ssh > 0:
        print("  [train] No SSH gradient — SSH loss skipped.")
    if not has_sic and lambda_ice > 0:
        print("  [train] No SIC — ice loss skipped.")

    lam_adv = lambda_advect if has_vel else 0.0
    lam_ssh = lambda_ssh    if has_ssh else 0.0
    lam_ice = lambda_ice    if has_sic else 0.0

    n = len(y_tr)
    zeros = np.zeros(n, dtype=np.float32)

    ds = TracerDataset(
        X      = scaler.transform(X_tr),
        y      = y_tr,
        u      = u_tr       if has_vel else zeros,
        v      = v_tr       if has_vel else zeros,
        deta_dx= deta_dx_tr if has_ssh else zeros,
        deta_dy= deta_dy_tr if has_ssh else zeros,
        sic    = sic_tr     if has_sic else zeros,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True
    )

    X_val_t = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    y_val_t = torch.tensor(y_val,                   dtype=torch.float32)

    model  = model.to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.OneCycleLR(
                 opt, max_lr=lr, epochs=n_epochs,
                 steps_per_epoch=len(loader), pct_start=0.1)
    huber  = nn.HuberLoss(delta=1.0)
    history = []

    for epoch in range(n_epochs):
        model.train()
        e_data = e_phys = 0.0

        for Xb, yb, ub, vb, dx_b, dy_b, sic_b in loader:
            Xb   = Xb.to(device);    yb  = yb.to(device)
            ub   = ub.to(device);    vb  = vb.to(device)
            dx_b = dx_b.to(device);  dy_b = dy_b.to(device)
            sic_b = sic_b.to(device)

            opt.zero_grad()

            L_dat = huber(model(Xb), yb)

            (Ld, Ls, Lt, Lstr,
             Ladv, Lssh, Lice) = physics_losses(
                model, Xb, feat_names,
                u_raw          = ub,
                v_raw          = vb,
                ssh_grad_x_raw = dx_b,
                ssh_grad_y_raw = dy_b,
                sic_raw        = sic_b,
            )

            L_phys = (lambda_diap   * Ld    +
                      lambda_smooth * Ls    +
                      lambda_sec    * Lt    +
                      lambda_strat  * Lstr  +
                      lam_adv       * Ladv  +
                      lam_ssh       * Lssh  +   # ← NEW
                      lam_ice       * Lice)      # ← NEW

            (L_dat + L_phys).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            e_data += L_dat.item()
            e_phys += L_phys.item()

        model.eval()
        with torch.no_grad():
            yv  = model(X_val_t.to(device)).cpu()
            rmse = torch.sqrt(((yv - y_val_t) ** 2).mean()).item()
            r2   = (1 - ((yv - y_val_t) ** 2).sum().item() /
                        ((y_val_t - y_val_t.mean()) ** 2).sum().item())

        history.append({
            "epoch"    : epoch,
            "data_loss": e_data / len(loader),
            "phys_loss": e_phys / len(loader),
            "val_rmse" : rmse,
            "val_r2"   : r2,
        })

        if epoch % 50 == 0:
            print(f"  [{epoch:3d}]  data={e_data/len(loader):.4f}  "
                  f"phys={e_phys/len(loader):.5f}  "
                  f"RMSE={rmse:.4f}  R²={r2:.3f}")

    return model, pd.DataFrame(history)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  HELPER: compute log N² (unchanged — included for self-containment)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_log_n2(df, cast_col):
    try:
        from pinn_tracer_N2 import compute_n2_per_cast
        return compute_n2_per_cast(df, cast_col=cast_col)
    except ImportError:
        warnings.warn("pinn_tracer_N2 not found — using per-point N² proxy.")
        SA    = gsw.SA_from_SP(df["salinity"].values, df["depth"].values,
                               df["lon"].values,      df["lat"].values)
        CT    = gsw.CT_from_t(SA, df["temp"].values, df["depth"].values)
        p     = gsw.p_from_z(-np.abs(df["depth"].values), df["lat"].values)
        alpha = gsw.alpha(SA, CT, p)
        beta  = gsw.beta(SA, CT, p)
        rho   = gsw.rho(SA, CT, p)
        n2    = np.maximum(9.7963 / rho * rho * (alpha + beta) / 100.0, 1e-8)
        return np.log10(n2)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN.PY INTEGRATION NOTES
# ═══════════════════════════════════════════════════════════════════════════════
#
# (a) New CLI args to add in parse_args():
#
#     grp_data.add_argument("--ssh_col",     default="ssh",
#         help="Sea surface height column (m).")
#     grp_data.add_argument("--deta_dx_col", default="deta_dx",
#         help="Eastward SSH gradient column (m/deg).")
#     grp_data.add_argument("--deta_dy_col", default="deta_dy",
#         help="Northward SSH gradient column (m/deg).")
#     grp_data.add_argument("--sic_col",     default="sic",
#         help="Sea ice concentration column (0-1).")
#
#     grp_phys.add_argument("--lambda_ssh",  type=float, default=0.04)
#     grp_phys.add_argument("--lambda_ice",  type=float, default=0.03)
#
# (b) In load_observations(), handle each column the same way as u/v:
#     check presence, rename to standard name, fill zeros + warn if absent.
#
# (c) Extract raw arrays after build_features():
#
#     deta_dx_valid = df_valid["deta_dx"].values.astype(np.float32)
#     deta_dy_valid = df_valid["deta_dy"].values.astype(np.float32)
#     sic_valid     = df_valid["sic"].values.astype(np.float32)
#
# (d) Pass to train_pinn() for both CV folds and final model:
#
#     model_cv, hist_cv = train_pinn(
#         model_cv,
#         X[tr_idx], y[tr_idx], X[val_idx], y[val_idx],
#         feat_names, scaler_cv,
#         u_tr        = u_valid[tr_idx]       if has_velocity else None,
#         v_tr        = v_valid[tr_idx]       if has_velocity else None,
#         deta_dx_tr  = deta_dx_valid[tr_idx] if has_ssh      else None,
#         deta_dy_tr  = deta_dy_valid[tr_idx] if has_ssh      else None,
#         sic_tr      = sic_valid[tr_idx]     if has_sic      else None,
#         ...
#         lambda_ssh  = args.lambda_ssh,
#         lambda_ice  = args.lambda_ice,
#     )
#
# (e) New diagnostic plot: residuals vs SIC (add alongside residuals_vs_speed):
#
#     if has_sic:
#         fig, ax = plt.subplots(figsize=(7, 5))
#         ax.scatter(df_valid["sic"], y_pred_mn - y, s=3, alpha=0.3)
#         ax.axhline(0, color="r")
#         ax.set(xlabel="Sea ice concentration", ylabel="Residual",
#                title="Residual vs SIC — should be centred under ice too")
#         plt.tight_layout()
#         plt.savefig(os.path.join(args.output_dir, "residuals_vs_sic.png"),
#                     dpi=150)
#         plt.close()
#
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# SANITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("SSH + SIC physics-loss sanity check...")
    rng = np.random.default_rng(0)
    n   = 256

    df = pd.DataFrame({
        "temp"    : rng.uniform(4,  20,  n),
        "salinity": rng.uniform(33, 36,  n),
        "depth"   : rng.uniform(5,  300, n),
        "lon"     : rng.uniform(-60, 0,  n),
        "lat"     : rng.uniform( 40, 80, n),
        "year"    : rng.integers(1990, 2020, n),
        "month"   : rng.integers(1, 13,  n),
        "tracer"  : rng.uniform(200, 300, n),
        "u"       : rng.uniform(-0.3, 0.3, n),
        "v"       : rng.uniform(-0.3, 0.3, n),
        "ssh"     : rng.uniform(-0.5, 0.5, n),
        "sic"     : rng.uniform(0.0,  1.0, n),
    })

    X, y, names, valid = build_features(df)
    print(f"\nFeatures ({len(names)}): {names}")
    print(f"X shape: {X.shape}")

    # Simulate SSH gradient arrays (normally interpolated from a climatology)
    deta_dx = rng.uniform(-0.02, 0.02, valid.sum()).astype(np.float32)
    deta_dy = rng.uniform(-0.02, 0.02, valid.sum()).astype(np.float32)
    sic     = df["sic"].values[valid].astype(np.float32)

    from pinn_tracer import TracerPINN
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    model  = TracerPINN(n_features=len(names))

    bs = 64
    Xb    = torch.tensor(scaler.transform(X[:bs]),  dtype=torch.float32)
    ub    = torch.tensor(df["u"].values[:bs],        dtype=torch.float32)
    vb    = torch.tensor(df["v"].values[:bs],        dtype=torch.float32)
    dx_b  = torch.tensor(deta_dx[:bs],              dtype=torch.float32)
    dy_b  = torch.tensor(deta_dy[:bs],              dtype=torch.float32)
    sic_b = torch.tensor(sic[:bs],                  dtype=torch.float32)

    Ld, Ls, Lt, Lstr, Ladv, Lssh, Lice = physics_losses(
        model, Xb, names, ub, vb, dx_b, dy_b, sic_b
    )

    print("\nLoss terms:")
    print(f"  L_diapycnal      : {Ld.item():.5f}")
    print(f"  L_smooth         : {Ls.item():.5f}")
    print(f"  L_secular        : {Lt.item():.5f}")
    print(f"  L_stratification : {Lstr.item():.5f}")
    print(f"  L_advection      : {Ladv.item():.5f}")
    print(f"  L_ssh            : {Lssh.item():.5f}  ← new")
    print(f"  L_ice            : {Lice.item():.5f}  ← new")
    print("\nSanity check passed.")