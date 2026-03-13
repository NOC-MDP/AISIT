"""Function for calculating the freshwater percentages from oxygen isotopes"""
import numpy as np
import xarray as xr
def mass_fraction(oxy_iso, sal, dmw):
    """
    Function for computing the mass fraction of freshwater oxygen isotopes.
    Parameters
    ----------
    oxy_iso, sal, dmw : xarray.DataArray
        DataArrays for oxygen isotopes, salinity, and meteoric water d18O
    Returns
    -------
    fsw, fmw, fsi : xarray.DataArray
    """
    ssw, smw, ssi = 34.88, 0.0, 3.0
    dsw, dsi = 0.34, 2.1

    # Convert everything to numpy — avoids ALL xarray alignment issues
    d18o_np = np.asarray(oxy_iso)
    sal_np  = np.asarray(sal)
    dmw_np  = np.asarray(dmw)

    # Broadcast dmw to match oxy_iso/sal shape if needed (e.g. no depth dim)
    dmw_np = np.broadcast_to(dmw_np, d18o_np.shape)

    def _solve(d18o_val, sal_val, dmw_val):
        k = np.array([[1,   1,       1  ],
                      [dsw, dmw_val, dsi],
                      [ssw, smw,     ssi]])
        rhs = np.array([1.0, d18o_val, sal_val])
        return np.linalg.solve(k, rhs)

    result = np.vectorize(_solve, signature="(),(),()->(n)")(d18o_np, sal_np, dmw_np)
    # result shape: (*original_dims, 3)

    # Wrap back into DataArrays using oxy_iso as the coordinate template
    fsw = xr.DataArray(result[..., 0], coords=oxy_iso.coords, dims=oxy_iso.dims)
    fmw = xr.DataArray(result[..., 1], coords=oxy_iso.coords, dims=oxy_iso.dims)
    fsi = xr.DataArray(result[..., 2], coords=oxy_iso.coords, dims=oxy_iso.dims)

    return fsw, fmw, fsi