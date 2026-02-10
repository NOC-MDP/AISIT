"""Function for calculating the freshwater percentages from oxygen isotopes"""

import numpy as np


def mass_fraction(oxy_iso, sal):
    """
    Function for computing the mass fraction of freshwater oxygen isotopes

    Parameters
    ----------
    oxy_iso, sal : xarray.DataArray
        DataArrays for oxygen isotopes and salinity

    Returns
    -------
    fir, fme, fsi: xarray.DataArray
    """
    # Constants
    sir = 34.88
    ssi = 3
    sme = 0
    dir_ = 0.34
    dme = -21
    dsi = 2.1  # Using values from Jones et al., 2008

    # Build K matrix
    k = np.array([1, dir_, sir], [1, dme, sme], [1, dsi, ssi])

    # Take the inverse and separate into components
    kinv = np.linalg.inv(k)
    a1, b1, c1 = kinv[0]
    a2, b2, c2 = kinv[1]
    a3, b3, c3 = kinv[2]

    # Compute the mass fractions
    fir = a1 + b1 * oxy_iso + c1 * sal
    fme = a2 + b2 * oxy_iso + c2 * sal
    fsi = a3 + b3 * oxy_iso + c3 * sal

    return fir, fme, fsi
