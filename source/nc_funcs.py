"""Functions for loading & processing .nc files"""

from datetime import datetime
from warnings import warn
from bisect import bisect

import os

import polars as pl
import pandas as pd
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.path as mpath

import cartopy.crs as ccrs
import cartopy.feature as cart


## -- Data processing -- ##

## Load the data


def coord_check(da, check_list, err_coord_name="coord"):
    """
    Checks if the DataArray's coordinate is in the default list

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray
    check_list : list of str
        Default check list
    err_coord_name : str, default "coord"
        Name of variable to be updated if KeyError raised

    Returns
    -------
    i : str
        Returns the coordinate name, or None if coordinate isn't in the list
    """
    check_list = {c.lower() for c in check_list}
    for i in da.coords:
        if i.lower() in check_list:
            return i
    raise KeyError(
        f"Coordinate not in default list. Please enter the coordinates for your DataArray manually in the `{err_coord_name}` variable."
    )


def read_single(path, engine="netcdf4", chunks=None):
    """
    Reads a .nc file.

    Parameters
    ----------
    path : str
        Path to file
    engine : str, default 'netcdf4'
        File type to be read. Options: ({"netcdf4", "scipy", "pydap", "h5netcdf", "zarr", None} , installed backend or subclass of xarray.backends.BackendEntrypoint, optional). If not provided, the default engine is chosen based on available dependencies, with a preference for “netcdf4”. A custom backend class (a subclass of BackendEntrypoint) can also be used.
    chunks : int, dict, 'auto' or None, default None
        If provided, used to load the data into dask arrays.
            chunks="auto" will use dask auto chunking taking into account the engine preferred chunks.
            chunks=None skips using dask, which is generally faster for small arrays.
            chunks=-1 loads the data with dask using a single chunk for all arrays.
            chunks={} loads the data with dask using the engine’s preferred chunk size, generally identical to the format’s chunk size. If not available, a single chunk for all arrays.
        See dask chunking for more details.

    Returns
    -------
    ds : xarray.DataSet
    """
    ds = xr.open_dataset(path, engine=engine, chunks=chunks)
    return ds


def read_multiple_dirs(
    paths, tcoord, engine="netcdf4", endswith=".nc", subdir_search=True, chunks=None
):
    """
    Reads multiple files from within a list of files, a single root directory, a list of directories, or some combination of all.

    Parameters
    ----------
    paths : str or tuple
        Input a single directory or a list of files/directories
    tcoord : str
        Coordinate along which to combine the files to produce a single DataSet
    engine : str, default 'netcdf4'
        File type to be read. Options: ({"netcdf4", "scipy", "pydap", "h5netcdf", "zarr", None} , installed backend or subclass of xarray.backends.BackendEntrypoint, optional). If not provided, the default engine is chosen based on available dependencies, with a preference for “netcdf4”. A custom backend class (a subclass of BackendEntrypoint) can also be used.
    endswith : str, default '.nc'
        Choose the str at the end of the file for listing the files.
    subdir_search : bool, default True
        Choose whether to list files in sub directories within a called directory. False means only files in the directory (not subdirectories) are listed
    chunks : int, dict, 'auto' or None, default None
        If provided, used to load the data into dask arrays.
            chunks="auto" will use dask auto chunking taking into account the engine preferred chunks.
            chunks=None skips using dask, which is generally faster for small arrays.
            chunks=-1 loads the data with dask using a single chunk for all arrays.
            chunks={} loads the data with dask using the engine’s preferred chunk size, generally identical to the format’s chunk size. If not available, a single chunk for all arrays.
        See dask chunking for more details.

    Returns
    -------
    ds : xarray.DataSet
    """
    fullpath_list = []
    if isinstance(
        paths, (list, tuple)
    ):  # Allows you to input multiple directories as a tuple
        for p in paths:
            if os.path.isfile(p):
                fullpath_list.append(p)
            else:
                list_files(
                    p,
                    fullpath_list,
                    endswith=endswith,
                    subdir_search=subdir_search,
                )
    else:
        list_files(paths, fullpath_list, endswith=endswith, subdir_search=subdir_search)
    fullpath_list = sorted(
        fullpath_list
    )  # Sorts in ascending order - is there a better way to do this with time?
    ds = xr.open_mfdataset(
        fullpath_list, combine="nested", concat_dim=tcoord, engine=engine, chunks=chunks
    )
    # concat_dim creates a dimension named 'tcoord' if there isn't already one
    return ds


def get_var(ds, var):
    """
    Chooses a specific variable from within a DataSet.

    Parameters
    ----------
    ds : xarray.DataSet
        Input DataSet
    var : str
        Name of chosen variable

    Returns
    -------
    davar : xarray.DataArray
    """
    try:
        davar = getattr(ds, var)  # DataArray for input variable
    except AttributeError:
        raise AttributeError(
            "Variable not recognised. Try again with one of the following: ",
            list(ds.keys()),
        )  # Raises an error if unrecognised variable is input
    davar.attrs.update(ds.attrs)  # Includes global attributes from ds in davar
    return davar


def climdata(
    file,
    var=None,
    t=None,
    x=None,
    y=None,
    engine="netcdf4",
    endswith=".nc",
    subdir_search=True,
    chunks=None,
):
    """
    Creates a DataArray analysing an input NetCDF file with climate data. Returns analysed DataArray.

    Parameters
    ----------
    file : str
        Path to file containing the climate data.
    var : str, optional
        Name of variable
    t, x, y : str, optional
        Time, latitude, longitude coordinates. Enter as a tuple (time,longitude,latitude). Optional, if the coordinate names are standard: ('time','t'),('longitude','lon'),('latitude','lat'). Only needed if doing any regridding or subset selection, or if your concatenating along a dimension that is not called "t" or "time"
    engine : str, default 'netcdf4'
        File type to be read. Options: ({"netcdf4", "scipy", "pydap", "h5netcdf", "zarr", None} , installed backend or subclass of xarray.backends.BackendEntrypoint, optional). If not provided, the default engine is chosen based on available dependencies, with a preference for “netcdf4”. A custom backend class (a subclass of BackendEntrypoint) can also be used.
    endswith : str, default '.nc'
        Choose the str at the end of the files which are to be read.
    subdir_search : bool, default True
        Choose whether to list files in sub directories within a called directory. False means only files in the directory (not subdirectories) are listed
    chunks : int, dict, 'auto' or None, default None
        If provided, used to load the data into dask arrays.
            chunks="auto" will use dask auto chunking taking into account the engine preferred chunks.
            chunks=None skips using dask, which is generally faster for small arrays.
            chunks=-1 loads the data with dask using a single chunk for all arrays.
            chunks={} loads the data with dask using the engine’s preferred chunk size, generally identical to the format’s chunk size. If not available, a single chunk for all arrays.
        See dask chunking for more details.
    Returns
    -------
    davar: xarray.DataArray
    """

    coords = (t, x, y)

    ## 1. Open the NetCDF file and choose a variable (e.g. wind speed). Build a DataArray

    if isinstance(file, str) and os.path.isfile(
        file
    ):  # Open file(s) and combine to create DataSet
        dset = read_single(file, engine=engine, chunks=chunks)
    elif isinstance(file, (tuple, list)) and [
        os.path.isfile(file[i]) for i in range(0, len(file) - 1)
    ]:
        if not coords[0]:
            try:
                dset = read_multiple_dirs(
                    file,
                    "time",
                    engine=engine,
                    endswith=endswith,
                    subdir_search=subdir_search,
                    chunks=chunks,
                )
            except UnboundLocalError:
                try:
                    dset = read_multiple_dirs(
                        file,
                        "t",
                        engine=engine,
                        endswith=endswith,
                        subdir_search=subdir_search,
                        chunks=chunks,
                    )
                except UnboundLocalError:
                    raise KeyError(
                        "Time coordinate not recognised. Please enter coords explicitly"
                    )
        else:
            t = coords[0]
            dset = read_multiple_dirs(
                file,
                t,
                engine=engine,
                endswith=endswith,
                subdir_search=subdir_search,
                chunks=chunks,
            )
    elif os.path.isdir(file):
        if not coords[0]:
            try:
                dset = read_multiple_dirs(
                    file,
                    "time",
                    engine=engine,
                    endswith=endswith,
                    subdir_search=subdir_search,
                    chunks=chunks,
                )
            except UnboundLocalError:
                try:
                    dset = read_multiple_dirs(
                        file,
                        "t",
                        engine=engine,
                        endswith=endswith,
                        subdir_search=subdir_search,
                        chunks=chunks,
                    )
                except UnboundLocalError:
                    raise KeyError(
                        "Time coordinate not recognised. Please enter coords explicitly"
                    )
        else:
            t = coords[0]
            dset = read_multiple_dirs(
                file,
                t,
                engine=engine,
                endswith=endswith,
                subdir_search=subdir_search,
                chunks=chunks,
            )
    else:
        raise FileNotFoundError(f"No such file or directory: {os.path.abspath(file)}")

    if var:  # Option to select specific variable to create DataArray
        davar = get_var(dset, var)  # Create DataArray with the specified variable
    else:
        davar = dset

    return davar  # Returns the DataArray


## Select time segments


class SortedWarning(Warning):
    """Warning class for Sortedness"""

    pass


class SortedError(Exception):
    """Error class for Sortedness"""

    pass


def _check_sorted(vals):
    return all(vals[i + 1] >= vals[i] for i in range(len(vals) - 1)) or all(
        vals[i + 1] <= vals[i] for i in range(len(vals) - 1)
    )


def _find_nearest(vals, test):
    i = bisect(vals, test)  # Position that test would be inserted

    # Handle edges
    if i == 0 and test <= vals[0]:
        return vals[0]
    elif i == len(vals) and test >= vals[-1]:
        return vals[-1]

    test_idx = [i - 1, i]
    return vals[test_idx[np.argmin([abs(test - vals[j]) for j in test_idx])]]


def find_nearest(vals, test, check_sorted=True):
    """
    Find the nearest value in a list of values for each test value.

    Uses bisection for speediness!

    Parameters
    ----------
    vals : list
        List of values - this is the pool of values for which we are looking
        for a nearest match. This list MUST be sorted. Sortedness is not
        checked, nor is the list sorted.
    test : list
        List of query values
    check_sorted : bool, default True
        Optionally check that the input vals is sorted. Raises an error if set
        to True (default), displays a warning if set to False.

    Returns
    -------
    A list containing the index of the nearest neighbour in vals for each value
    in test. Or the index of the nearest neighbour if test is a single value.
    """
    if check_sorted:
        s = _check_sorted(vals)
        if not s:
            raise SortedError("Input values are not sorted")
    else:
        warn("Not checking sortedness of data", SortedWarning)

    if not isinstance(test, list):
        return _find_nearest(vals, test)

    return [_find_nearest(vals, t) for t in test]


def daterange(tstart, tend, tfreq, skip_29feb=False):
    """
    Generates a list of dates, starting from tstart with a frequency of tfreq, and ending as close to tend as possible

    Parameters
    ----------
    tstart : str
        Start time
    tend : str
        End time
    tfreq : str
        Frequency. Enter using polars.datetime_range intervals (see Notes)
    skip_29feb : bool, default False
        Choose whether to skip 29th Feb when calculating date ranges on a n-daily scale. Only works with the "d" interval. Default is to not skip

    Returns
    -------
    range : numpy.ndarray

    Notes
    -----
    - 1ns (1 nanosecond)
    - 1us (1 microsecond)
    - 1ms (1 millisecond)
    - 1s (1 second)
    - 1m (1 minute)
    - 1h (1 hour)
    - 1d (1 calendar day)
    - 1w (1 calendar week)
    - 1mo (1 calendar month)
    - 1q (1 calendar quarter)
    - 1y (1 calendar year)
    """
    # Change tstart,tend to datetime.datetime objects
    tstart, tend = np.datetime64(tstart).astype("M8[ms]").astype(
        datetime
    ), np.datetime64(tend).astype("M8[ms]").astype(datetime)
    # Polars range
    if not skip_29feb:
        polars_range = pl.datetime_range(
            tstart,
            tend,
            tfreq,
            eager=True,
        )
    else:
        n = int("".join(ch for ch in tfreq if ch.isdigit()))
        # Daterange for daily, then drop 29th Feb
        polars_range_day = pl.datetime_range(
            tstart,
            tend,
            "1d",
            eager=True,
        )
        polars_range_day_no29feb = polars_range_day.filter(
            ~((polars_range_day.dt.month() == 2) & (polars_range_day.dt.day() == 29))
        )
        # Create new daterange that skips 29th Feb
        polars_range = polars_range_day_no29feb[::n]
    # Numpy range
    numpy_range = polars_range.to_numpy()
    return numpy_range


def timeseg(da, t, Tseg, Tstep=False, skip_29feb=False):
    """
    Select a time segment for the DataArray. Relies on the existence of a time coordinate and assumes they are numpy.datetime64 objects.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray
    t: str
        Name of time coordinate
    Tseg : str or tuple
        Time region to select. Enter as a single date or a tuple of dates (start_date,end_date) in the form of numpy.datetime64 values.
    Tstep: str, optional
        Time step to select. Enter using polars.datetime_range intervals (see Notes). Default is input data frequency.
    skip_29feb : bool, default False
        Choose whether to skip 29th Feb when calculating date ranges on a n-daily scale. Only works with the "d" interval. Default is to not skip

    Returns
    -------
    da : xarray.DataArray

    Notes
    -----
    - 1ns (1 nanosecond)
    - 1us (1 microsecond)
    - 1ms (1 millisecond)
    - 1s (1 second)
    - 1m (1 minute)
    - 1h (1 hour)
    - 1d (1 calendar day)
    - 1w (1 calendar week)
    - 1mo (1 calendar month)
    - 1q (1 calendar quarter)
    - 1y (1 calendar year)
    """
    # If entered as a tuple
    if isinstance(Tseg, tuple) or not Tseg:
        if Tseg:
            Tstart, Tend = Tseg
        else:
            Tstart, Tend = False, False
        if not Tstart:
            Tstart = getattr(da, t).values[0]
        if not Tend:
            Tend = getattr(da, t).values[-1]
        Tseg = (Tstart, Tend)
        # Here we actually select (everything leading is just making sure the time coords work properly)
        if not Tstep:
            da = da.sel({t: slice(Tstart, Tend)})
        else:
            Tstart_test = getattr(da.sel({t: Tstart}), t).values
            if isinstance(Tstart_test, np.ndarray) and len(Tstart_test) > 1:
                Tstart = Tstart_test[0]
            else:
                Tstart = getattr(da.sel({t: Tstart}, method="nearest"), t).values

            Tend_test = getattr(da.sel({t: Tend}), t).values
            if isinstance(Tend_test, np.ndarray) and len(Tend_test) > 1:
                Tend = Tend_test[0]
            else:
                Tend = getattr(da.sel({t: Tend}, method="nearest"), t).values
            Tlist = list(daterange(Tstart, Tend, Tstep, skip_29feb=skip_29feb))
            Tlist = find_nearest(getattr(da, t).values, Tlist)
            ## Add threshold ##
            da = da.sel({t: Tlist})
    # If one time value entered
    else:
        da = da.sel({t: Tseg})
    return da


def create_seasonal_da(da, t="time"):
    """
    Create a seasonally-averaged dataset based off higher resolution, with dates at the start of the season

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray
    t : str, default 'time'
        Time coordinate

    Returns
    -------
    da_season : xarray.DataArray
    """
    da_season = da.resample({t: "QS-DEC"}).mean(t)
    return da_season


def timestep_da(
    da, T, Tseg=False, t="time", average_method="mean", skip_29feb=False, skipna=True
):
    """
    Takes an average over a given timestep T e.g. average monthly data annually.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray
    t : str
        Time coordinate
    T : str
        Desired time frequency. Enter using polars.datetime_range intervals (see Notes)
    Tseg : tuple, optional
        Time segment. Enter as a tuple (tstart,tend), where they are both np.datetime64. Default is to use input data's full time span.
    average_method : str, default 'mean'
        Chosen method to average over the timestep.
    skip_29feb : bool, default False
        Choose whether to skip 29th Feb when calculating date ranges on a n-daily scale. This groups 29th Feb with previous grouping e.g. for "5d", there will be a 6 day group from 24th-29th Feb in a leap year. Default is to not skip
    skipna : bool, default True
        Whether to skip NaN values when calculating the average.

    Returns
    -------
    grouped : xarray.DataArray

    Notes
    -----
    - 1ns (1 nanosecond)
    - 1us (1 microsecond)
    - 1ms (1 millisecond)
    - 1s (1 second)
    - 1m (1 minute)
    - 1h (1 hour)
    - 1d (1 calendar day)
    - 1w (1 calendar week)
    - 1mo (1 calendar month)
    - 1q (1 calendar quarter)
    - 1y (1 calendar year)
    - 1sn (1 season) -- This is NOT a polars.datetime_range interval. Only use '1sn' (not '2sn'...). Returns in ['DJF','MAM','JJA','SON'] format
    """

    # Creates seasonal average from higher res data
    if T == "1sn":
        grouped = create_seasonal_da(da, t)

    # Creates other averages using polars.datetime_range interval
    else:
        # da is the time segmented array; da_subset is the time segmented array with the timestep included (NOT regridded yet)
        da = timeseg(da, t, Tseg, Tstep=False)
        da_subset = timeseg(da, t, Tseg, T, skip_29feb=skip_29feb)

        # Define custom bins with which to group the data, according to the t values of da_subset
        bins = []
        j = 0
        for i in da[t].values:
            try:
                if i < da_subset[t].values[j + 1]:
                    bins.append(da_subset[t].values[j])
                else:
                    j = j + 1
                    bins.append(da_subset[t].values[j])
            except IndexError:
                bins.append(da_subset[t].values[j])

        group_labels = xr.DataArray(bins, coords={t: da[t].values}, dims=t, name=t)

        # Group by this custom variable and compute the average method
        grouped = getattr(da.groupby(group_labels), average_method)(
            dim=t, skipna=skipna
        )

    return grouped


## Select spatial regions


def reg_sel(da, coord, Reg, step=False):
    """
    Selects a region of the data along a chosen coordinate, with the option to choose the frequency of the coordinate too. This does NOT regrid the data.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray
    coord : str
        Coordinate along which you will choose the region
    Reg : float or tuple
        Chosen region. Enter as either a single point or a tuple (start,end). If tuple, it will select a range of values for the coordinate, from the start point to end point, with a given frequency (see step parameter).
    step : float, optional
        Frequency of the data. Only relevant if Reg is a tuple. Default is for the frequency to be the frequency of the input DataArray.

    Returns
    -------
    da : xarray.DataArray
    """
    # if statement for whether we want single point or a start to end region
    da = da.rename({coord: "coord"})
    if (
        da.coord.values[0] > da.coord.values[1]
    ):  # Makes sure we go from smallest to largest number
        cvals, cvale = da.coord.values[-1], da.coord.values[0]
        if isinstance(Reg, tuple):
            Reg = Reg[1], Reg[0]
        else:
            Reg = Reg
    else:
        cvals, cvale = da.coord.values[0], da.coord.values[-1]

    if isinstance(Reg, tuple):  # Subregion
        # if cvals<=Reg[0]<=cvale and cvals<=Reg[1]<=cvale:
        start, end = Reg
        if not start in da.coord.values:
            start = da.sel(coord=start, method="nearest").coord.values
        if not step:
            da = da.sel(coord=slice(start, end))
        else:
            istep = int(step / (abs(da.coord[1] - da.coord[0])))
            da = da.sel(coord=slice(start, end, istep))

        #     else:
        #         raise ValueError("Start values not in list of coordinate values.")
        # else:
        #     raise ValueError("Selected region outside bounds of the data")
    else:
        if cvals <= Reg <= cvale:
            da = da.sel(
                coord=Reg, method="nearest"
            )  # Single point. N.B. If chosen point isn't in data, takes nearest value
        else:
            raise ValueError("Selected value is not a data point")
    da = da.rename({"coord": coord})
    return da


def xy_region(da, X, Y, Xreg, Yreg, Xstep=False, Ystep=False):
    """
    Select a x-y region from your DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray
    X : float
        Longitude coordinate
    Y : float
        Latitude coordinate
    Xreg : float or tuple
        Longitude region. Enter as a single point or tuple (x_start,x_end).
    Yreg : float or tuple
        Latitude region. Enter as a single point or tuple (y_start,y_end).
    Xstep : float, optional
        Longtiudinal frequency. Default is to keep the frequency of the input DataArray.
    Ystep : float, optional
        Latitudinal frequency. Default is to keep the frequency of the input DataArray.

    Returns
    -------
    da : xarray.DataArray
    """
    if Xreg:
        da = reg_sel(da, X, Xreg, Xstep)

    if Yreg:
        da = reg_sel(da, Y, Yreg, Ystep)

    return da


## Climatology & anomaly functions


def weighted_average(da, X, Y):
    """
    Performs a weighted average over longitude-latitude. Takes the mean of all longitude values and the weighted mean of all latitude values, where the weight is cos(latitude). Returns a timeseries.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray
    X : str
        Longitude coordinate
    Y : str
        Latitude coordinate

    Returns
    -------
    da : xarray.DataArray
    """
    weights = np.cos(
        np.deg2rad(getattr(da, Y))
    )  # Create DataArray "weights" for cos(latitude)
    weights.name = "weights"
    da_weighted = da.weighted(
        weights
    )  # Weights DataArray da along latitude dimension by cos(latitude)
    da_out = da_weighted.mean(
        (X, Y), skipna=True
    )  # Takes weighted mean and mean over latitude and longitude respectively
    da_out.attrs.update(da.attrs)
    return da_out


def climatology(
    da,
    tstep,
    coords=("time", "longitude", "latitude"),
    average_method="mean",
    drop_leap_yr=True,
):
    """
    Returns a DataArray for the climatology of a dataset. Use this function in conjunction with climdata().

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray
    tstep : str
        Time step you wish to take e.g. "day","month"... Use a DatetimeAccessor. Note: use "dayofyear" for daily climatology, not "day"!
    coords : tuple, default ("time","longitude","latitude")
        Coordinates
    average_method : str, default 'mean'
        Choice of averaging method over the time step. If None, it will return the grouped by object
    harm_fit : tuple, optional
        Parameters for fitting climatology to a Fourier series. Enter as (n,T), where n & T are the order of the series and period of the data respectively. Uses the 'harm_fit()' function.
    drop_leap_yr : bool, default True
        Option to drop leap year from a daily climatology

    Returns
    -------
    da_step : xarray.DataArray
    """
    dtattrs = (
        "year",
        "season",
        "quarter",
        "month",
        "week",
        "day",
        "dayofyear",
        "hour",
        "minute",
        "second",
        "microsecond",
        "nanosecond",
    )  # List of time steps

    t, x, y = coords
    da = da.rename({t: "t"})
    davt = da.t
    if tstep == "dayofyear":
        if drop_leap_yr:
            da = da.sel(t=~((da.t.dt.month == 2) & (da.t.dt.day == 29)))
            # Create reference for non-leap year
            leap_ref = pd.date_range("2001-01-01", "2001-12-31", freq="D")
        else:
            # Create reference for leap year
            leap_ref = pd.date_range("2000-01-01", "2000-12-31", freq="D")
        # Apply mapping to create a new coordinate
        month_day_to_doy = {(d.month, d.day): i + 1 for i, d in enumerate(leap_ref)}
        doy = xr.DataArray(
            [month_day_to_doy[(s.month, s.day)] for s in pd.to_datetime(da.t.values)],
            coords=[da.t],
            dims=["t"],
        )
        da.coords["dayofyear"] = doy
        da_step = da.groupby("dayofyear")
    else:
        try:
            tstp = getattr(davt.dt, tstep)
        except AttributeError:
            raise AttributeError(
                "Unrecognised time step. Please select from the following list: ",
                dtattrs,
            )  # Raises error if incorrect time step is input
        # Group by the timestep
        da_step = da.groupby(tstp)
    # Retain attributes
    da_step.attrs = da.attrs
    # Take the chosen average method of the data values, according to the chosen timestep.
    if average_method:
        da_step = getattr(da_step, average_method)(dim="t")
    else:
        da_step = da_step.rename({"t": t})

    return da_step


def anomaly(da, daclim, tstep, t="time"):
    """
    Generates a DataArray for the climate anomaly of a variable.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray
    daclim : xarray.DataArray
        DataArray of a climatology
    tstep : str
        Time frequency of data
    t : str, default 'time'
        Time coordinate

    Returns
    -------
    anom : xarray.DataArray
    """
    da2 = da[f"{t}.{tstep}"]
    daclim2 = daclim.sel({tstep: da2})  # Creates array with month
    anom = da - daclim2  # Subtracts appropriate climatology
    anom = anom.drop_vars(tstep)  # Drops month
    anom.attrs.update(da.attrs)
    return anom


## -- Plotting -- ##

## Timeseries functions


def timeseriesplot(da, var, tcoord, tseg=False, **figkwargs):
    """
    Creates a time series plot of climate data.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing climate data
    var : str
        Variable name. This will label the y-axis.
    tcoord : str
        Time coordinate. This will label the x-axis.
    tseg : tuple, optional
        time span of the plot. Enter as tuple (tstart,tend). If data is a DateTime, enter as (np.datetime64(tstart),np.datetime64(tend)). Default is full range of data.
    **figkwargs : optional
        List of kwargs relating to figure options.

    Returns
    -------
    fig, axes, p : matplotlib.Figure.figure, matplotlib.Axes.axes, matplotlib.pyplot.plot

    Other parameters (figkwargs)
    ----------------------------
    fig_title : str, optional
        Title of figure
    title_font_size : int, default 20
        Font size of title
    legend_labels : str or tuple of str, optional
        Labels for legend
    legend_location : str or int, optional
        Location of legend. 9 locations, combining keywords 'upper','center' & 'lower' with 'left' & 'right'. Use 'best' to put the legend in the default best location.
    line_colors : str, optional
        Color for the plot's lines
    line_style : str, optional
        Style for the plot's lines
    alpha : float, default 1
        Opacity of plot's lines
    figsize : (float,float), optional
        Figure width, height in inches.
    axes_label_fontsize : int
        Fontsize for axes labels
    legend_fontsize : int
        Fontsize for legend labels
    grid : bool, default False
        Choose whether to have gridlines
    """

    defaultkwargs = {
        "fig_title": None,
        "title_font_size": 20,
        "legend_labels": None,
        "legend_location": None,
        "line_colors": None,
        "line_style": None,
        "alpha": None,
        "figsize": (15, 7),
        "axes_label_fontsize": 12,
        "legend_fontsize": 12,
        "grid": False,
    }

    figkwargs = {**defaultkwargs, **figkwargs}

    t = tcoord

    fig, axes = plt.subplots(
        1, 1, figsize=figkwargs["figsize"]
    )  # Sets up figure & axes
    fig.suptitle(figkwargs["fig_title"], fontsize=figkwargs["title_font_size"])
    # if statement to give the option to plot multiple DataArrays on the same plot
    if isinstance(da, tuple):
        da_dict = {}
        for _, i in enumerate(da):
            da_dict[f"da{i}"] = da[i]
            if figkwargs["line_colors"]:
                p = da_dict[f"da{i}"].plot.line(
                    x=t,
                    ax=axes,
                    color=(
                        figkwargs["line_colors"]
                        if not figkwargs["line_colors"]
                        else figkwargs["line_colors"][i]
                    ),
                    linestyle=(
                        figkwargs["line_style"]
                        if not figkwargs["line_style"]
                        else figkwargs["line_style"][i]
                    ),
                    alpha=(
                        figkwargs["alpha"]
                        if not figkwargs["alpha"]
                        else figkwargs["alpha"][i]
                    ),
                )
            else:
                p = da_dict[f"da{i}"].plot.line(x=t, ax=axes)
        if figkwargs["legend_labels"]:
            axes.legend(
                [*figkwargs["legend_labels"]],
                loc=figkwargs["legend_location"],
                fontsize=figkwargs["legend_fontsize"],
            )
    else:
        p = da.plot.line(
            x=t,
            ax=axes,
            color=figkwargs["line_colors"],
            linestyle=figkwargs["line_style"],
        )
        if figkwargs["legend_labels"]:
            axes.legend(
                [figkwargs["legend_labels"]],
                loc=figkwargs["legend_location"],
                fontsize=figkwargs["legend_fontsize"],
            )

    axes.grid(figkwargs["grid"])
    axes.set_title(None)
    axes.set_xlabel(f"{t}", fontsize=figkwargs["axes_label_fontsize"])
    axes.set_ylabel(f"{var}", fontsize=figkwargs["axes_label_fontsize"])
    axes.tick_params(
        axis="both", which="major", labelsize=figkwargs["axes_label_fontsize"]
    )
    if tseg:
        axes.set_xlim(tseg)

    return fig, axes, p


## Lat-lon plot functions


def circle():  # For creating circular boundaries in lat-lon plots
    """
    Defines a circle in matplotlib
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circl = mpath.Path(verts * radius + center)
    return circl


def add_colorbar(
    fig,
    plot,
    ax_pos=(0.85, 0.15, 0.025, 0.7),
    cbar_label=None,
    cbar_label_size=18,
    cbar_ticklabel_size=16,
    **cbarkws,
):  # For adding colorbars to plots, specifically in multiplot
    """
    Add a colorbar to a matplotlib.figure.Figure. Recommend using in conjunction with multiplots function.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure into which the colorbar will be added
    plot : matplotlib.pyplot.plot
        Plot which the colorbar is assigned to
    ax_pos : tuple of floats (left, bottom, width, height), default (0.85,0.15,0.025,0.7)
        Axes into which the colorbar will be drawn.
    cbar_label : str, optional
        Label for the colorbar
    cbar_kws : dict, optional
        Dictionary of keyword arguments for colorbar. See matplotlib.pyplot.colorbar documentation for more information

    Returns
    -------
    cbar : matplotlib.pyplot.colorbar
    """
    cbar_ax = fig.add_axes(ax_pos)
    cbar = fig.colorbar(plot, cax=cbar_ax, **cbarkws)
    cbar.ax.tick_params(labelsize=cbar_ticklabel_size)
    if cbar_label:
        cbar.set_label(label=cbar_label, size=cbar_label_size)
    return cbar


def add_region_highlight(reg, axs, **figkwargs):
    """
    Add a highlighted region to a cartopy plot.

    Parameters
    ----------
    reg : tuple
        Corners of highlighted area. Enter as (lon_min, lon_max, lat_min, lat_max)
    axs : matplotlib.Axes.axes
    figkwargs : dict, optional
        Dictionary of keyword arguments for region lines. See matplotlib.pyplot.plot for more information
    """

    x0, x1, y0, y1 = reg

    axs.plot([x0, x0], [y0, y1], transform=ccrs.Geodetic(), **figkwargs)

    axs.plot([x1, x1], [y0, y1], transform=ccrs.Geodetic(), **figkwargs)

    lons = np.linspace(x0, x1, 100)

    axs.plot(lons, np.full_like(lons, y0), transform=ccrs.Geodetic(), **figkwargs)

    axs.plot(lons, np.full_like(lons, y1), transform=ccrs.Geodetic(), **figkwargs)

    return None


def multiplot(
    da,
    tcoord=None,
    ncol=1,
    nrow=1,
    x=None,
    y=None,
    xyreg=(False, False),
    cax_pos=(0.9, 0.21, 0.025, 0.5),
    cbar_label=None,
    cbar_kws=None,
    **figkwargs,
):
    """
    Produce a single plot or an array of plots using matplotlib.pyplot.subplots, integrated with Cartopy.

    Option to produce a colorbar for the whole figure. This can be integrated with other matplotlib figure features.

    Default is a single plot on a PlateCarree map with a colorbar.

    da : xarray.DataArray
        Input DataArray
    tcoord : str, optional
        Time coordinate. Set to None for single plot.
    ncol : int, default 1
        Number of columns
    nrow : int, default 1
        Number of rows
    x : str, optiona
        Longitude coordinate name
    y : str, optional
        Latitude coordinate name
    xyreg : tuple, optional
        Lon-lat region. Enter as tuple (lon_point,lat_point) or tuple of tuples ((lon_start,lon_end),(lat_start,lat_end)). Default is the full region from the input DataArray.
    cax_pos : tuple (left, bottom, width, height), default (0.9,0.21,0.025,0.5)
        Dimensions (left, bottom, width, height) of the colorbar axes. All quantities are in fractions of figure width and height. Default is to not produce a colorbar. Note that if a colorbar is produced for an array of plots, use vmin, vmax kwargs to ensure colorbar aligns with all plots. Set cax_pos=None for no colorbar.
    cbar_label : str, optional
        Label for colorbar
    cbar_kws : dict, optional
        Dictionary of keyword arguments for colorbar. Includes cbar_kws={'label':None,'label_size':18,ticklabel_size':16}. See add_colorbar function for more information.
    figkwargs : optional
        An assortment of figure keyword arguments (see below)

    Returns
    -------
    fig, axs, p : matplotlib.figure.Figure, matplotlib.axes.Axes or array of Axes, matplotlib.pyplot.plot or list of matplotlib.pyplot.plot

    Other Parameters (figkwargs)
    ----------------------------
    figsize : (float,float), optional
        Figure width, height in inches.
    left : float, optional
        The position of the left edge of the subplots, as a fraction of the figure width.
    right : float, default 0.85
        The position of the right edge of the subplots, as a fraction of the figure width.
    bottom : float, default 0.05
        The position of the bottom edge of the subplots, as a fraction of the figure height.
    top : float, optional
        The position of the top edge of the subplots, as a fraction of the figure height.
    wspace : float, optional
        The width of the padding between subplots, as a fraction of the average Axes width.
    hspace : float, optional
        The height of the padding between subplots, as a fraction of the average Axes height.
    suptitle : str, optional
        Title for the whole figure
    suptitle_fontsize : float, default 12
        Fontsize for figure title
    subplot_title : str or tuple, optional
        Titles of each subplots. Enter as a string 'title' for a single plot and a tuple of strings ('title1','title2',...) for an array of plots. Set as ' ' for no subplot titles. Default to the time coordinate chosen.
    subplot_fontsize : int, default 10
        Subplot label fontsize
    axes_label_fontsize : int, default 8
        Axes label fontsize
    map_type : str, default 'PlateCaree'
        Type of Cartopy map to be used
    cmap : str, default 'RdBu'
        Color scale to be used
    set_mask_color : str, optional
        Color to be used for the nan values (i.e. mask)
    levels : int or array-like, optional
        Determines the number and positions of the contour lines / regions. If array-like, draw contour lines at the specified levels. The values must be in increasing order.
    vmin, vmax : float, optional
        Define the data range that the colormap covers. By default, the colormap covers the complete value range of the supplied data.
    land_edge_color : str, default 'k'
        Color used for land edge
    land_face_color : str, default 'None'
        Color used for land face
    central_longitude : float, default 0
        Central longitudinal position on plot
    central_latitude : float, optional
        Central latitudinal position on plot
    gridlines : bool, default False
        Choose to dispay gridlines
    circular_boundary : bool, default False
        Choose to display circular (as opposed to rectangular) boundaries on the map. Only works if chosen map type is circular.

    Useful settings
    ---------------
    Here are some useful settings for various grids of figures. Assumes a global plot on a PlateCarree map with a vertical colorbar that has a label.
    Single plot (ncol=1,nrow=1) : figsize=(10,7),right=0.85,bottom=0.05,cax_pos=(0.9,0.21,0.025,0.5)
    2x2 plot (ncol=2,nrow=2) : figsize=(10,7),right=0.85,bottom=0.05,cax_pos=(0.9,0.21,0.025,0.5)
    4x3 plot (ncol=4,nrow=3) : figsize=(17,8),right=.95,cax_pos=(0.97,.1,.025,.7)
    """

    defaultkwargs = {
        "figsize": (10, 7),
        "left": None,
        "right": 0.85,
        "top": None,
        "bottom": 0.05,
        "hspace": None,
        "wspace": None,
        "suptitle": None,
        "suptitle_fontsize": 12,
        "subplot_title": None,
        "subplot_fontsize": 10,
        "axes_label_fontsize": 8,
        "map_type": "PlateCarree",
        "cmap": "RdBu",
        "set_mask_color": None,
        "levels": None,
        "vmin": None,
        "vmax": None,
        "land_edge_color": "k",
        "land_face_color": "None",
        "central_longitude": 0,
        "central_latitude": None,
        "gridlines": False,
        "circular_boundary": False,
    }

    figkwargs = {**defaultkwargs, **figkwargs}

    xreg, yreg = xyreg

    # da = change_lon(da, x, to0360=False)

    if (
        not xreg
    ):  # This makes the longitude centre around the westerly xreg & outputs correct longitude map region. Default is to centre at 0 deg
        clon = figkwargs["central_longitude"]
        xdiff = 360
    else:
        xdiff = xreg[1] - xreg[0]
        clon = xdiff / 2 + xreg[0]

    if not yreg:  # This makes latitude axes correct extent on map
        yregN, yregS = -90, 90
    else:
        yregN, yregS = yreg[0], yreg[1]

    if not figkwargs["central_latitude"]:
        proj = getattr(ccrs, figkwargs["map_type"])(central_longitude=clon)
    else:
        proj = getattr(ccrs, figkwargs["map_type"])(
            central_longitude=clon, central_latitude=figkwargs["central_latitude"]
        )  # Central latitude required for some map projections e.g. Orthographic. Add with fig_options

    fig, axs = plt.subplots(
        nrow, ncol, subplot_kw={"projection": proj}, figsize=figkwargs["figsize"]
    )

    fig.subplots_adjust(
        left=figkwargs["left"],
        right=figkwargs["right"],
        top=figkwargs["top"],
        bottom=figkwargs["bottom"],
        hspace=figkwargs["hspace"],
        wspace=figkwargs["wspace"],
    )

    if figkwargs["suptitle"]:
        fig.suptitle(figkwargs["suptitle"], fontsize=figkwargs["suptitle_fontsize"])

    if figkwargs["set_mask_color"]:
        cmap = getattr(plt.cm, figkwargs["cmap"]).copy()
        cmap.set_bad((0, 0, 0, 0))  # Set nan values to be transparent (white)
    else:
        cmap = figkwargs["cmap"]

    if ncol > 1 or nrow > 1:
        if not tcoord:
            tcoord = coord_check(da, ("time", "t"), err_coord_name="tcoord")
        for m, ax in enumerate(axs.flat):
            p = []
            p.append(
                da.isel({tcoord: m}).plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    levels=figkwargs["levels"],
                    vmin=figkwargs["vmin"],
                    vmax=figkwargs["vmax"],
                    add_colorbar=False,
                )
            )
            pcbar = p[0]  # The plot on which to scale the colorbar

            if figkwargs["set_mask_color"]:
                # Check the x, y coordinates for common ones
                if not x:
                    x = coord_check(da, ("longitude", "lon", "x"))
                if not y:
                    y = coord_check(da, ("latitude", "lat", "y"))
                # Set a colour for the mask by overlaying on top of plot
                nan_mask = np.isnan(da.isel({tcoord: m}))
                custom_cmap = mcolors.ListedColormap(
                    [(0, 0, 0, 0), figkwargs["set_mask_color"]]
                )  # Custom cmap for overlaying nan value colour mask
                ax.pcolormesh(
                    getattr(da.isel({tcoord: m}), x),
                    getattr(da.isel({tcoord: m}), y),
                    nan_mask,
                    cmap=custom_cmap,
                    transform=ccrs.PlateCarree(),
                )

            if xreg == False and yreg == False:
                ax.set_global()
            else:
                ax.set_extent(
                    (clon - xdiff / 2, clon + xdiff / 2, yregS, yregN),
                    crs=ccrs.PlateCarree(),
                )
            ax.add_feature(
                cart.feature.LAND,
                zorder=1,
                edgecolor=figkwargs["land_edge_color"],
                facecolor=figkwargs["land_face_color"],
            )
            gl = ax.gridlines(draw_labels=True, alpha=figkwargs["gridlines"] + 0)
            gl.top_labels, gl.right_labels = False, False
            gl.xlabel_style = {"size": figkwargs["axes_label_fontsize"]}
            gl.ylabel_style = {"size": figkwargs["axes_label_fontsize"]}

            if figkwargs["circular_boundary"]:
                ax.set_boundary(circle(), transform=ax.transAxes)

            if figkwargs["subplot_title"]:
                ax.set_title(
                    figkwargs["subplot_title"][m], size=figkwargs["subplot_fontsize"]
                )
    else:
        p = da.plot(
            ax=axs,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            levels=figkwargs["levels"],
            vmin=figkwargs["vmin"],
            vmax=figkwargs["vmax"],
            add_colorbar=False,
        )
        pcbar = p

        if figkwargs["set_mask_color"]:
            # Check the x,y coordinates for common names
            if not x:
                x = coord_check(da, ("longitude", "lon", "x"))
            if not y:
                y = coord_check(da, ("latitude", "lat", "y"))
            # Set a colour for the mask by overlaying on top of plot
            nan_mask = np.isnan(da)
            custom_cmap = mcolors.ListedColormap(
                [(0, 0, 0, 0), figkwargs["set_mask_color"]]
            )  # Custom cmap for overlaying nan value colour mask
            axs.pcolormesh(
                getattr(da, x),
                getattr(da, y),
                nan_mask,
                cmap=custom_cmap,
                transform=ccrs.PlateCarree(),
            )

        if xreg == False and yreg == False:
            axs.set_global()
        else:
            axs.set_extent(
                (clon - xdiff / 2, clon + xdiff / 2, yregS, yregN),
                crs=ccrs.PlateCarree(),
            )
        axs.add_feature(
            cart.feature.LAND,
            zorder=1,
            edgecolor=figkwargs["land_edge_color"],
            facecolor=figkwargs["land_face_color"],
        )
        gl = axs.gridlines(draw_labels=True, alpha=figkwargs["gridlines"] + 0)
        gl.top_labels, gl.right_labels = False, False
        gl.xlabel_style = {"size": figkwargs["axes_label_fontsize"]}
        gl.ylabel_style = {"size": figkwargs["axes_label_fontsize"]}

        if figkwargs["circular_boundary"]:
            axs.set_boundary(circle(), transform=axs.transAxes)

        if figkwargs["subplot_title"]:
            axs.set_title(
                figkwargs["subplot_title"], size=figkwargs["subplot_fontsize"]
            )

    if cax_pos:
        if cbar_kws:
            add_colorbar(
                fig,
                pcbar,
                ax_pos=cax_pos,
                cbar_label=cbar_label,
                cbar_label_size=figkwargs["subplot_fontsize"],
                cbar_ticklabel_size=figkwargs["subplot_fontsize"],
                **cbar_kws,
            )
        else:
            add_colorbar(
                fig,
                pcbar,
                ax_pos=cax_pos,
                cbar_label=cbar_label,
                cbar_label_size=figkwargs["subplot_fontsize"],
                cbar_ticklabel_size=figkwargs["subplot_fontsize"],
            )

    return fig, axs, p
