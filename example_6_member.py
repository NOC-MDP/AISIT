import numpy as np
from scipy.optimize import lsq_linear

# -------------------------------------------------------------------------
# 1. Base Reference End-Members & Standard Deviations
# -------------------------------------------------------------------------
base_end_members = {
    "ATL": np.array([1.0, 34.80,  0.30,  45.0, 2300.0]),
    "PAC": np.array([1.0, 32.50, -1.10,  78.0, 2220.0]),
    "NAM": np.array([1.0,  0.00,-19.50, 130.0, 1600.0]),
    "EUR": np.array([1.0,  0.00,-19.00,  45.0,  800.0]),
    "SIM": np.array([1.0,  4.00,  2.00,  10.0,  300.0]),
    "GLAC": np.array([1.0, 0.00,-30.00,   2.0,   20.0]),
}

end_member_std = {
    "ATL": np.array([0.0, 0.05, 0.05,  3.0,  10.0]),
    "PAC": np.array([0.0, 0.30, 0.15,  5.0,  15.0]),
    "NAM": np.array([0.0, 0.00, 1.50, 20.0, 150.0]),
    "EUR": np.array([0.0, 0.00, 1.00,  8.0,  80.0]),
    "SIM": np.array([0.0, 1.00, 0.50,  3.0,  40.0]),
    "GLAC": np.array([0.0, 0.00, 2.00,  1.0,  15.0]),
}

obs_uncertainty = np.array([0.0, 0.01, 0.05, 1.5, 10.0])
weights = np.array([100.0, 25.0, 10.0, 5.0, 4.0])
water_masses = ["ATL", "PAC", "NAM", "EUR", "SIM", "GLAC"]

# -------------------------------------------------------------------------
# 2. Dynamic End-Member Parameterization
# -------------------------------------------------------------------------
def get_dynamic_end_members(lat, lon, depth):
    """
    Adjusts base end-member tracer signatures dynamically as a function
    of geographical coordinates and depth.
    """
    # Deep copy base signatures to modify per sample
    em = {k: v.copy() for k, v in base_end_members.items()}

    # --- A. Atlantic Water Depth / Regional Gradient ---
    # Framing Fram Strait / Norwegian Sea inflow vs Deep Eurasian Basin Atlantic Water
    if depth > 500:
        em["ATL"][1] = 34.91   # Deep Atlantic is slightly saltier
        em["ATL"][2] = 0.35    # d18O shift in deep basin
        em["ATL"][4] = 2310.0  # Higher TA due to remineralization/dissolution
    elif lon < 0: # Atlantic sector / Fram Strait surface
        em["ATL"][1] = 35.00   # Core North Atlantic Water

    # --- B. Pacific Water Regional Shifts ---
    # Bering Strait / Chukchi Sea vs Beaufort Gyre Pacific Summer Water (PSW) / Winter Water (PWW)
    if depth > 100 and depth <= 250:
        # PWW (Pacific Winter Water) core: colder, nutrient/Ba rich, saltier
        em["PAC"][1] = 33.10
        em["PAC"][2] = -0.80
        em["PAC"][3] = 85.0    # Higher Barium signature in PWW
    elif depth <= 50:
        # PSW (Pacific Summer Water): fresher
        em["PAC"][1] = 31.80
        em["PAC"][2] = -1.30

    # --- C. River End-Member Regional Gradients ---
    # Eurasian Rivers: Lena/Yenisei vs Ob (Ob has higher alkalinity)
    # Longitude check for Western Siberian rivers vs Eastern Siberian rivers
    if 60 <= lon <= 90: # Ob River dominance zone
        em["EUR"][4] = 1100.0  # Ob has higher TA (~1100 µmol/kg)
    elif 120 <= lon <= 140: # Lena River dominance zone
        em["EUR"][4] = 700.0   # Lena has lower TA (~700 µmol/kg)
        em["EUR"][3] = 60.0    # Higher Ba in Lena

    # North American Rivers: Mackenzie River vs Alaskan rivers
    if lon < -120: # Western North American / Mackenzie influence
        em["NAM"][4] = 1750.0  # High carbonate runoff from Mackenzie Basin
        em["NAM"][2] = -20.5   # Isotopic depletion increases inland

    return em


# -------------------------------------------------------------------------
# 3. Solver with Dynamic Signatures & Bounds
# -------------------------------------------------------------------------
def calculate_dynamic_bounds(depth, lon,month=None, dist_chukchi=9999, dist_ru=9999, dist_na=9999):
    """
    Adjusts solver bounds dynamically based on depth, location, and seasonality.
    """
    # Default minimum SIM based on seasonality if month is provided
    if month is not None:
        if month in [6, 7, 8]:  # Summer (June-August): Melt season, no brine rejection
            min_sim = 0.0
            glac_upper = 0.15  # Allow glacial melt in summer
        elif month in [5, 9, 10]:  # Spring/Autumn (Transition shoulders)
            min_sim = -0.05
            glac_upper = 0.05  # Shoulder seasons
        else:  # Winter (Nov-May): Active freezing and brine rejection
            min_sim = -0.20
            glac_upper = 0.001 # Winter: Shut off glacial melt!
    else:
        min_sim = -0.02  # Default fallback if month is unknown
        glac_upper = 1.0

    lower_bounds = [0.0, 0.0, 0.0, 0.0, min_sim, 0.0]
    upper_bounds = [1.0, 1.0, 1.0, 1.0, 1.0, glac_upper]

    # 1. Depth & Regional Constraint for Atlantic Water
    # Atlantic water does not occupy the surface/shelf layers of the Canada Basin / Beaufort Sea (lon < -100)
    if depth < 150 and lon < -100:
        upper_bounds[0] = 0.05  # Restrict ATL to < 5% in the surface Beaufort/Canada Basin

    # Depth constraints
    if depth > 300:
        upper_bounds[2] = 0.02  # NAM
        upper_bounds[3] = 0.02  # EUR
        upper_bounds[4] = 0.05  # SIM
        upper_bounds[5] = 1e-5  # GLAC (near zero)

    # Geographic constraints
    if dist_chukchi > 2000:
        upper_bounds[1] = 0.15  # PAC

    # Allow higher PAC fraction in the surface Beaufort/Chukchi shelf
    if depth <= 50 and lon < -120:
        upper_bounds[1] = 1.0  # Allow up to 100% Pacific Surface Water

    return lower_bounds, upper_bounds


def solve_single_6comp_omp(A_raw, obs_raw, lower_bounds, upper_bounds):
    mean_vals = np.mean(A_raw, axis=1)
    std_vals = np.std(A_raw, axis=1)
    std_vals[0] = 1.0
    mean_vals[0] = 0.0

    A_norm = (A_raw - mean_vals[:, None]) / std_vals[:, None]
    A_weighted = A_norm * weights[:, None]

    obs_norm = (obs_raw - mean_vals) / std_vals
    obs_weighted = obs_norm * weights

    res = lsq_linear(A_weighted, obs_weighted, bounds=(lower_bounds, upper_bounds))
    x = res.x
    f_sum = np.sum(x)

    return x / f_sum if f_sum != 0 else x


# -------------------------------------------------------------------------
# 4. Monte Carlo Engine Using Dynamic End-Members
# -------------------------------------------------------------------------
def run_6comp_monte_carlo(
    obs_sal, obs_d18O, obs_ba, obs_ta, lat, lon, depth, month=None, n_iter=2000
):
    base_obs = np.array([1.0, obs_sal, obs_d18O, obs_ba, obs_ta])
    results = np.zeros((n_iter, 6))

    # Get local dynamic end-members for this specific (Lat, Lon, Depth)
    dynamic_end_members = get_dynamic_end_members(lat, lon, depth)
    lb, ub = calculate_dynamic_bounds(depth=depth, lon=lon,month=month)

    for i in range(n_iter):
        A_perturbed = np.zeros((5, 6))
        for j, wm in enumerate(water_masses):
            noise = np.random.normal(0, end_member_std[wm])
            # Perturb around the dynamically computed mean tracer values
            A_perturbed[:, j] = dynamic_end_members[wm] + noise

        obs_noise = np.random.normal(0, obs_uncertainty)
        obs_perturbed = base_obs + obs_noise

        fractions = solve_single_6comp_omp(A_perturbed, obs_perturbed, lb, ub)
        results[i, :] = fractions

    summary = {}
    for j, wm in enumerate(water_masses):
        summary[wm] = {
            "mean": np.mean(results[:, j]),
            "std": np.std(results[:, j]),
            "ci_95": np.percentile(results[:, j], [2.5, 97.5]),
        }

    return summary, results


# -------------------------------------------------------------------------
# 5. Example Execution: Mackenzie River Delta Surface Sample
# -------------------------------------------------------------------------
# Sample collected near the Mackenzie River mouth in July (Lat: 70°N, Lon: -135°W, Depth: 10m)
sample_sal = 22.0
sample_d18O = -12.0
sample_ba = 110.0
sample_ta = 1700.0
month = 7



stats, mc_runs = run_6comp_monte_carlo(
    obs_sal=sample_sal,
    obs_d18O=sample_d18O,
    obs_ba=sample_ba,
    obs_ta=sample_ta,
    lat=70.0,
    lon=-135.0,
    depth=10.0,
    n_iter=2000,
    month=month
)
print(f"{'Month':<14} {month}")
print(f"{'Water Mass':<14} | {'Mean Fraction':<15} | {'Std Dev (±)':<12} | {'95% CI Range':<20}")
print("-" * 69)
for wm in water_masses:
    m = stats[wm]["mean"] * 100
    s = stats[wm]["std"] * 100
    ci_low, ci_high = stats[wm]["ci_95"] * 100
    print(f"{wm:<14} | {m:>6.2f}%         | ±{s:>5.2f}%      | [{ci_low:>6.2f}%, {ci_high:>6.2f}%]")

# Sample collected near the Mackenzie River mouth in January (Lat: 70°N, Lon: -135°W, Depth: 10m)
sample_sal = 31.5       # Much higher salinity due to winter mixing and lack of summer freshet
sample_d18O = -2.5      # Much less depleted (river input is minimal, mostly marine/winter baseflow)
sample_ba = 55.0        # Lower barium, reflecting lower winter river discharge
sample_ta = 2150.0      # Higher alkalinity, closer to marine water values
month = 1               # Triggers winter rules (min_sim = -0.20)

stats, mc_runs = run_6comp_monte_carlo(
    obs_sal=sample_sal,
    obs_d18O=sample_d18O,
    obs_ba=sample_ba,
    obs_ta=sample_ta,
    lat=70.0,
    lon=-135.0,
    depth=10.0,
    n_iter=2000,
    month=month
)
print(f"{'Month':<14} {month}")
print(f"{'Water Mass':<14} | {'Mean Fraction':<15} | {'Std Dev (±)':<12} | {'95% CI Range':<20}")
print("-" * 69)
for wm in water_masses:
    m = stats[wm]["mean"] * 100
    s = stats[wm]["std"] * 100
    ci_low, ci_high = stats[wm]["ci_95"] * 100
    print(f"{wm:<14} | {m:>6.2f}%         | ±{s:>5.2f}%      | [{ci_low:>6.2f}%, {ci_high:>6.2f}%]")
