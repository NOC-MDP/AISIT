"""
Quasi-conservative Barium PINN loss functions — Level 2 parameterisation.

Physics: barite (BaSO4) saturation-state-driven remineralisation, coupled
with precipitation tied to particle export, and soft constraint terms for
mixed-layer suppression, column conservation, and the Ba-Si* correlation.

Governing PDE:
    dBa/dt + u·∇Ba = ∇·(K·∇Ba) + R_remin(Ba, SO4, T, P) - R_precip(Ba, POC)

All inputs are assumed to be PyTorch tensors of shape (N,) or (N, 1),
where N is the number of collocation points in the batch.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# K_sp parameterisation
# ---------------------------------------------------------------------------

def ksp_barite(T_C: torch.Tensor, P_dbar: torch.Tensor) -> torch.Tensor:
    """
    Solubility product of barite (BaSO4) as a function of temperature and
    pressure, after Monnin et al. (1999).

    Args:
        T_C:     Temperature in degrees Celsius.
        P_dbar:  Pressure in dbar (approx. equivalent to depth in metres).

    Returns:
        K_sp in (mol/kg)^2.
    """
    T_K = T_C + 273.15

    # Temperature dependence (log10 Ksp fit, simplified Monnin form)
    log10_Ksp0 = -9.87 + 0.0174 * (T_C - 25.0) - 3.5e-5 * (T_C - 25.0) ** 2

    # Pressure correction via the partial molar volume of dissolution
    # dln(Ksp)/dP = -delta_V / (RT)
    delta_V  = -26.0e-6   # m^3 / mol  (partial molar volume change)
    R_gas    = 8.314       # J / (mol K)
    P_pascal = P_dbar * 1e4  # dbar -> Pa  (1 dbar ≈ 10^4 Pa)

    ln_Ksp_pressure_correction = -(delta_V * P_pascal) / (R_gas * T_K)

    # Convert log10 base to natural log, apply correction, convert back
    ln_Ksp0  = log10_Ksp0 * torch.log(torch.tensor(10.0))
    ln_Ksp   = ln_Ksp0 + ln_Ksp_pressure_correction
    return torch.exp(ln_Ksp)


# ---------------------------------------------------------------------------
# Saturation index
# ---------------------------------------------------------------------------

def saturation_index(
    Ba:    torch.Tensor,
    SO4:   torch.Tensor,
    T_C:   torch.Tensor,
    P_dbar: torch.Tensor,
) -> torch.Tensor:
    """
    Ion activity product divided by K_sp.

        Omega = [Ba2+][SO4 2-] / K_sp(T, P)

    Omega < 1  ->  undersaturated  ->  barite dissolves  ->  R_remin > 0
    Omega > 1  ->  oversaturated   ->  barite precipitates -> R_precip > 0

    Args:
        Ba:     Dissolved [Ba2+] in nmol/kg (will be converted internally).
        SO4:    Dissolved [SO4 2-] in mmol/kg.
        T_C:    Temperature in degrees Celsius.
        P_dbar: Pressure in dbar.

    Returns:
        Dimensionless saturation index Omega, shape (N,).
    """
    Ba_mol  = Ba  * 1e-9   # nmol/kg -> mol/kg
    SO4_mol = SO4 * 1e-3   # mmol/kg -> mol/kg
    Ksp     = ksp_barite(T_C, P_dbar)
    return (Ba_mol * SO4_mol) / (Ksp + 1e-30)  # epsilon avoids divide-by-zero


# ---------------------------------------------------------------------------
# Source / sink terms
# ---------------------------------------------------------------------------

def R_remineralisation(
    Ba:         torch.Tensor,
    SO4:        torch.Tensor,
    T_C:        torch.Tensor,
    P_dbar:     torch.Tensor,
    k_diss:     torch.Tensor,
) -> torch.Tensor:
    """
    Barite dissolution rate driven by local undersaturation.

        R_remin = k_diss * relu(1 - Omega)

    The relu enforces zero dissolution in oversaturated water — the saturation
    horizon emerges naturally from the data without a hard-coded depth cutoff.

    Args:
        Ba, SO4, T_C, P_dbar: State variables (see saturation_index).
        k_diss:  Learned dissolution rate constant (scalar or broadcastable).
                 Should be passed through softplus before this call to ensure
                 it is non-negative.

    Returns:
        R_remin in nmol/kg/s, shape (N,).
    """
    omega = saturation_index(Ba, SO4, T_C, P_dbar)
    undersaturation = torch.relu(1.0 - omega)
    return k_diss * undersaturation


def R_precipitation(
    Ba:       torch.Tensor,
    POC:      torch.Tensor,
    k_precip: torch.Tensor,
    n_poc:    float = 1.0,
) -> torch.Tensor:
    """
    Barite precipitation within sinking organic aggregates.

        R_precip = k_precip * POC^n_poc * Ba

    The Ba term captures the concentration dependence of nucleation;
    n_poc controls the non-linearity of the export flux coupling.

    Args:
        Ba:       Dissolved [Ba2+] in nmol/kg.
        POC:      Particulate organic carbon flux proxy (e.g. mg C / m^2 / d,
                  normalised to dimensionless [0, 1] range recommended).
        k_precip: Learned precipitation rate constant (passed through softplus
                  externally to ensure non-negativity).
        n_poc:    POC flux exponent (default 1.0, can be learned).

    Returns:
        R_precip in nmol/kg/s, shape (N,).
    """
    return k_precip * (POC ** n_poc) * Ba


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------

def loss_pde_residual(
    dBa_dt:    torch.Tensor,
    advection: torch.Tensor,
    diffusion: torch.Tensor,
    remin:     torch.Tensor,
    precip:    torch.Tensor,
) -> torch.Tensor:
    """
    MSE of the PDE residual at interior collocation points.

    Residual form:
        dBa/dt + u·∇Ba - ∇·(K·∇Ba) - R_remin + R_precip = 0

    All terms must be in consistent units (nmol/kg/s recommended).
    The advection and diffusion terms should be computed externally via
    automatic differentiation through your PINN forward pass.
    """
    residual = dBa_dt + advection - diffusion - remin + precip
    return torch.mean(residual ** 2)


def loss_data(
    Ba_pred: torch.Tensor,
    Ba_obs:  torch.Tensor,
) -> torch.Tensor:
    """
    MSE against observed dissolved Ba concentrations.

    Args:
        Ba_pred: PINN output at observation locations (nmol/kg).
        Ba_obs:  Measured Ba (nmol/kg).
    """
    return torch.mean((Ba_pred - Ba_obs) ** 2)


def loss_mixed_layer(
    remin_mld: torch.Tensor,
) -> torch.Tensor:
    """
    Soft constraint: remineralisation must vanish in the surface mixed layer.

    Pass only the R_remin values at collocation points shallower than the
    local mixed layer depth (MLD). The saturation-state formulation already
    suppresses this physically, but this penalty term adds an explicit
    gradient signal during early training when Omega predictions are noisy.

    Args:
        remin_mld: R_remin evaluated at points with depth < MLD.
    """
    if remin_mld.numel() == 0:
        return torch.tensor(0.0)
    return torch.mean(remin_mld ** 2)


def loss_column_conservation(
    remin_col:  torch.Tensor,
    precip_col: torch.Tensor,
) -> torch.Tensor:
    """
    Soft column-integrated Ba conservation constraint.

    In steady state, depth-integrated remineralisation ≈ depth-integrated
    precipitation within a water column. This is approximate (lateral
    transport can violate it locally) so weight this term lightly.

    Args:
        remin_col:  R_remin values over a full water column (nmol/kg/s).
        precip_col: R_precip values over the same column (nmol/kg/s).
    """
    return (torch.mean(remin_col) - torch.mean(precip_col)) ** 2


def loss_ba_si_correlation(
    Ba_pred: torch.Tensor,
    Si_obs:  torch.Tensor,
    a:       torch.Tensor,
    b:       torch.Tensor,
) -> torch.Tensor:
    """
    Ba–Si* linear correlation constraint.

    Dissolved Ba and silicic acid co-vary linearly across most water masses
    (Sillen & Dymond, 1992; Jeandel et al., 1996):

        [Ba] ≈ a * [Si] + b

    This is a cheap, high-information constraint because Si is measured far
    more widely than Ba. a and b can be fixed from literature values or
    learned as additional PINN parameters.

    Typical open-ocean values (nmol/kg vs µmol/kg):
        a ≈ 0.27  (nmol Ba per µmol Si)
        b ≈ 45    (nmol/kg preformed Ba)

    Args:
        Ba_pred: PINN Ba predictions at collocated Si observation sites.
        Si_obs:  Measured silicic acid (µmol/kg).
        a, b:    Slope and intercept (fixed or learned).
    """
    Ba_from_Si = a * Si_obs + b
    return torch.mean((Ba_pred - Ba_from_Si) ** 2)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def barium_pinn_loss(
    # ----- PDE terms (from autodiff through the PINN forward pass) -----
    dBa_dt:    torch.Tensor,   # time derivative of Ba prediction
    advection: torch.Tensor,   # u·∇Ba
    diffusion: torch.Tensor,   # ∇·(K·∇Ba)

    # ----- State variables at collocation points -----
    Ba_pred:   torch.Tensor,   # PINN Ba output (nmol/kg)
    SO4:       torch.Tensor,   # sulphate (mmol/kg)
    T_C:       torch.Tensor,   # temperature (°C)
    P_dbar:    torch.Tensor,   # pressure (dbar)
    POC:       torch.Tensor,   # POC flux proxy (dimensionless, normalised)

    # ----- Learned rate constants (raw, pre-softplus) -----
    k_diss_raw:   torch.Tensor,   # dissolution rate constant
    k_precip_raw: torch.Tensor,   # precipitation rate constant

    # ----- Data constraint -----
    Ba_obs:    torch.Tensor,   # observed Ba at data points (nmol/kg)
    Ba_pred_at_obs: torch.Tensor,  # PINN output at observation locations

    # ----- Mixed-layer mask -----
    mld_mask:  torch.Tensor,   # boolean, True where depth < MLD

    # ----- Ba-Si* constraint -----
    Ba_pred_at_si: torch.Tensor,  # PINN output at Si observation sites
    Si_obs:    torch.Tensor,      # observed Si (µmol/kg)
    a_basi:    torch.Tensor,      # Ba-Si slope  (fixed or learned)
    b_basi:    torch.Tensor,      # Ba-Si intercept (fixed or learned)

    # ----- Loss weights -----
    lambda_data: float = 1.0,
    lambda_mld:  float = 0.1,
    lambda_col:  float = 0.01,
    lambda_basi: float = 0.5,
) -> dict:
    """
    Compute all PINN loss terms for the quasi-conservative Barium tracer.

    Returns a dict with individual loss components and the weighted total,
    so you can log each term separately during training and adjust weights
    without rewriting the loss call.

    Usage example
    -------------
    losses = barium_pinn_loss(
        dBa_dt=dBa_dt, advection=adv, diffusion=diff,
        Ba_pred=Ba, SO4=SO4, T_C=T, P_dbar=P, POC=poc,
        k_diss_raw=model.k_diss, k_precip_raw=model.k_precip,
        Ba_obs=ba_obs, Ba_pred_at_obs=Ba_at_obs,
        mld_mask=mld_mask,
        Ba_pred_at_si=Ba_at_si, Si_obs=si_obs,
        a_basi=model.a_basi, b_basi=model.b_basi,
    )
    losses["total"].backward()
    """
    # Enforce non-negativity of rate constants via softplus
    k_diss   = nn.functional.softplus(k_diss_raw)
    k_precip = nn.functional.softplus(k_precip_raw)

    # Source / sink terms at all collocation points
    remin  = R_remineralisation(Ba_pred, SO4, T_C, P_dbar, k_diss)
    precip = R_precipitation(Ba_pred, POC, k_precip)

    # --- Individual loss components ---
    L_phys = loss_pde_residual(dBa_dt, advection, diffusion, remin, precip)
    L_data = loss_data(Ba_pred_at_obs, Ba_obs)
    L_mld  = loss_mixed_layer(remin[mld_mask])
    L_col  = loss_column_conservation(remin, precip)
    L_basi = loss_ba_si_correlation(Ba_pred_at_si, Si_obs, a_basi, b_basi)

    L_total = (
        L_phys
        + lambda_data * L_data
        + lambda_mld  * L_mld
        + lambda_col  * L_col
        + lambda_basi * L_basi
    )

    return {
        "total":       L_total,
        "pde":         L_phys,
        "data":        L_data,
        "mld":         L_mld,
        "column":      L_col,
        "ba_si":       L_basi,
        # Diagnostics — detached so they don't affect the graph
        "k_diss":      k_diss.detach(),
        "k_precip":    k_precip.detach(),
        "omega_mean":  saturation_index(Ba_pred, SO4, T_C, P_dbar).mean().detach(),
    }