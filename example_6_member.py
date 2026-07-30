import numpy as np
from scipy.optimize import lsq_linear

# -------------------------------------------------------------------------
# 1. Base End-Member Definitions & Standard Deviations (Variability)
# -------------------------------------------------------------------------
# Vector Format: [Mass, Salinity, delta18O (‰), Barium (nmol/kg), Total Alkalinity (µmol/kg)]
base_end_members = {
    "ATL": np.array([1.0, 34.80, 0.30, 45.0, 2300.0]),  # Atlantic Marine
    "PAC": np.array([1.0, 32.50, -1.10, 78.0, 2220.0]),  # Pacific Marine
    "NAM": np.array(
        [1.0, 0.00, -19.50, 130.0, 1600.0]
    ),  # North American Rivers (High Ba & TA)
    "EUR": np.array(
        [1.0, 0.00, -19.00, 45.0, 800.0]
    ),  # Eurasian Rivers (Low Ba, Moderate TA)
    "SIM": np.array(
        [1.0, 4.00, 2.00, 10.0, 300.0]
    ),  # Sea Ice Melt / Brine (Low TA)
    "GLAC": np.array(
        [1.0, 0.00, -30.00, 2.0, 20.0]
    ),  # Glacier Melt (Ultra-light isotope, near-zero TA/Ba)
}

# Standard deviations capturing natural end-member variability/freshet noise
end_member_std = {
    "ATL": np.array([0.0, 0.05, 0.05, 3.0, 10.0]),
    "PAC": np.array([0.0, 0.30, 0.15, 5.0, 15.0]),
    "NAM": np.array([0.0, 0.00, 1.50, 20.0, 150.0]),
    "EUR": np.array([0.0, 0.00, 1.00, 8.0, 80.0]),
    "SIM": np.array([0.0, 1.00, 0.50, 3.0, 40.0]),
    "GLAC": np.array([0.0, 0.00, 2.00, 1.0, 15.0]),
}

# Analytical measurement uncertainties (Lab/ML Model Prediction Error)
# [Mass, Sal, d18O, Ba, TA]
obs_uncertainty = np.array([0.0, 0.01, 0.05, 1.5, 10.0])

# Solver Weights: [Mass, Sal, d18O, Ba, TA]
# Note: TA weighted at 4.0 to balance tracer sensitivity without over-fitting biological photic noise
weights = np.array([100.0, 25.0, 10.0, 5.0, 4.0])
water_masses = ["ATL", "PAC", "NAM", "EUR", "SIM", "GLAC"]


# -------------------------------------------------------------------------
# 2. Single Bounded 6-Component OMP Solver
# -------------------------------------------------------------------------
def solve_single_6comp_omp(A_raw, obs_raw, min_sim=-0.20):
    """Solves one 6-component OMP realization with bounds using lsq_linear."""
    mean_vals = np.mean(A_raw, axis=1)
    std_vals = np.std(A_raw, axis=1)
    std_vals[0] = 1.0  # Keep mass conservation row unscaled
    mean_vals[0] = 0.0

    # Normalization & Weighting
    A_norm = (A_raw - mean_vals[:, None]) / std_vals[:, None]
    A_weighted = A_norm * weights[:, None]

    obs_norm = (obs_raw - mean_vals) / std_vals
    obs_weighted = obs_norm * weights

    # Bounds for [ATL, PAC, NAM, EUR, SIM, GLAC]
    # All marine, river, and glacier fractions >= 0. SIM lower bound permits brine rejection.
    lower_bounds = [0.0, 0.0, 0.0, 0.0, min_sim, 0.0]
    upper_bounds = [1.0, 1.0, 1.0, 1.0, 1.00, 1.0]

    res = lsq_linear(
        A_weighted, obs_weighted, bounds=(lower_bounds, upper_bounds)
    )
    x = res.x
    f_sum = np.sum(x)

    return x / f_sum if f_sum != 0 else x


# -------------------------------------------------------------------------
# 3. Monte Carlo Simulation Engine
# -------------------------------------------------------------------------
def run_6comp_monte_carlo(
    obs_sal, obs_d18O, obs_ba, obs_ta, n_iter=2000, min_sim=-0.20
):
    """Runs N-iteration Monte Carlo OMP simulation for 6 end-members.

    Parameters:
      obs_sal  : Salinity
      obs_d18O : delta18O (‰)
      obs_ba   : Barium (nmol/kg)
      obs_ta   : Total Alkalinity (µmol/kg)
      n_iter   : Number of Monte Carlo iterations
      min_sim  : Lower bound allowed for negative SIM (brine rejection)
    """
    base_obs = np.array([1.0, obs_sal, obs_d18O, obs_ba, obs_ta])
    results = np.zeros((n_iter, 6))

    for i in range(n_iter):
        # A. Perturb 6 End-Members with Gaussian distribution
        A_perturbed = np.zeros((5, 6))
        for j, wm in enumerate(water_masses):
            noise = np.random.normal(0, end_member_std[wm])
            A_perturbed[:, j] = base_end_members[wm] + noise

        # B. Perturb Sample Observations (Analytical / ML prediction noise)
        obs_noise = np.random.normal(0, obs_uncertainty)
        obs_perturbed = base_obs + obs_noise

        # C. Solve OMP for perturbed realization
        fractions = solve_single_6comp_omp(
            A_perturbed, obs_perturbed, min_sim=min_sim
        )
        results[i, :] = fractions

    # D. Summary Statistics
    summary = {}
    for j, wm in enumerate(water_masses):
        summary[wm] = {
            "mean": np.mean(results[:, j]),
            "std": np.std(results[:, j]),
            "ci_95": np.percentile(results[:, j], [2.5, 97.5]),
        }

    return summary, results


# -------------------------------------------------------------------------
# 4. Example Run: Fjord/Shelf Profile with Glacial Runoff
# -------------------------------------------------------------------------
# Sample Observation: Mixed surface layer with Pacific, North American River, and Greenland Glacial Discharge
sample_sal = 28.5
sample_d18O = -4.5
sample_ba = 68.0
sample_ta = 1350.0  # Reduced TA driven by zero-alkalinity glacier melt dilution

stats, mc_runs = run_6comp_monte_carlo(
    obs_sal=sample_sal,
    obs_d18O=sample_d18O,
    obs_ba=sample_ba,
    obs_ta=sample_ta,
    n_iter=2000,
    min_sim=-0.20,
)

# Print Summary
print(
    f"{'Water Mass':<14} | {'Mean Fraction':<15} | {'Std Dev (±)':<12} | {'95% CI Range':<20}"
)
print("-" * 69)
for wm in water_masses:
    m = stats[wm]["mean"] * 100
    s = stats[wm]["std"] * 100
    ci_low, ci_high = stats[wm]["ci_95"] * 100
    print(
        f"{wm:<14} | {m:>6.2f}%         | ±{s:>5.2f}%      | [{ci_low:>6.2f}%, {ci_high:>6.2f}%]"
    )
