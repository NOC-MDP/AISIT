import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset(
    "cmems_mod_arc_phy_my_topaz4_P1M_stfbaro_180.00W-179.88E_50.00N-90.00N_1991-01-01-2025-10-01.nc"
)
var = "stfbaro"

def extract_point(ds, var, lat, lon):
    return ds[var].sel(latitude=lat, longitude=lon, method="nearest")

p_east = extract_point(ds, var, 80, 8.7)
p_west = extract_point(ds, var, 80, -13)

diff = p_east - p_west
diff.plot(figsize=(18,9))
plt.show()

