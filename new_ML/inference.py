"""
inference.py — Apply the trained model to the full ocean model grid.

The output is a proper xr.DataArray with the same coordinate metadata as the
input ocean model, ready to be written to NetCDF or used in further analysis.
"""

import logging
import numpy as np
import xarray as xr
import torch
from pathlib import Path
from typing import Optional

from config import Config
from data import Normaliser, build_grid_features
from model import TracerMLP

log = logging.getLogger(__name__)


def infer_on_grid(
    cfg: Config,
    model: TracerMLP,
    norm: Normaliser,
    device: torch.device,
    time_slice: Optional[str] = None,
    output_path: Optional[str] = None,
) -> xr.DataArray:
    """
    Run the trained model over the full ocean model grid.

    Parameters
    ----------
    cfg         : Config
    model       : trained TracerMLP
    norm        : fitted Normaliser (to transform inputs and inverse-transform output)
    device      : torch device
    time_slice  : optional ISO time string to select a single snapshot
    output_path : if provided, save result to NetCDF

    Returns
    -------
    tracer_da : xr.DataArray (lon × lat × depth) in original tracer units
    """
    dc = cfg.data

    log.info(f"Loading grid from {dc.grid_path}")
    ds = xr.open_dataset(dc.grid_path, chunks="auto")

    # Build feature matrix for every grid cell
    log.info("Building grid feature matrix …")
    X_grid, stacked = build_grid_features(ds, dc, time_slice=time_slice)

    # Drop cells with any NaN (e.g. land mask) — track indices to put them back
    valid_mask = np.isfinite(X_grid).all(axis=1)
    X_valid    = X_grid[valid_mask]
    log.info(f"Grid cells: {len(X_grid):,} total, {valid_mask.sum():,} ocean (non-NaN)")

    # Normalise inputs
    X_valid_n = norm.transform_X(X_valid)

    # Predict
    log.info("Running inference …")
    y_norm = model.predict_numpy(X_valid_n, device=device)

    # Inverse-transform to physical units
    y_phys = norm.inverse_transform_y(y_norm).astype(np.float32)

    # Reconstruct full array (NaN for land)
    y_full = np.full(len(X_grid), np.nan, dtype=np.float32)
    y_full[valid_mask] = y_phys

    # Unstack back to (lon × lat × depth) DataArray
    tracer_da = xr.DataArray(
        data   = y_full,
        coords = stacked.coords,
        dims   = ["cell"],
        name   = "tracer_predicted",
    ).unstack("cell")

    tracer_da.attrs.update({
        "long_name":  "Predicted Conservative Tracer",
        "units":      "model_units",   # update as appropriate
        "source":     "TracerMLP inference",
        "grid_vars":  ", ".join(dc.grid_vars),
    })

    log.info(
        f"Inference complete. "
        f"Min={float(np.nanmin(y_phys)):.3f}  "
        f"Max={float(np.nanmax(y_phys)):.3f}  "
        f"Mean={float(np.nanmean(y_phys)):.3f}"
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        tracer_da.to_netcdf(output_path)
        log.info(f"Saved to {output_path}")

    return tracer_da


def infer_time_series(
    cfg: Config,
    model: TracerMLP,
    norm: Normaliser,
    device: torch.device,
    times: list,
    output_path: Optional[str] = None,
) -> xr.DataArray:
    """
    Run inference for a sequence of time slices and concatenate into a
    (time × lon × lat × depth) DataArray.
    """
    slices = []
    for t in times:
        log.info(f"  Processing time slice: {t}")
        da = infer_on_grid(cfg, model, norm, device, time_slice=t)
        da = da.assign_coords(time=t).expand_dims("time")
        slices.append(da)

    full_da = xr.concat(slices, dim="time")
    full_da.name = "tracer_predicted"

    if output_path:
        full_da.to_netcdf(output_path)
        log.info(f"Time series saved to {output_path}")

    return full_da
