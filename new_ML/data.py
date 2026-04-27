"""
data.py — Co-location, feature engineering, normalisation, and PyTorch Datasets.

Co-location strategy
--------------------
For each sparse observation (lon, lat, depth, time) we extract the value of every
gridded variable using xarray's .sel() with nearest-neighbour lookup.  This keeps
everything in xarray-land until we hand off a clean numpy array to PyTorch.
"""

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

from config import Config, DataConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  Load & co-locate
# ---------------------------------------------------------------------------

def load_observations(cfg: DataConfig) -> pd.DataFrame:
    """Load sparse tracer observations from CSV."""
    df = pd.read_csv(cfg.obs_path, parse_dates=[cfg.obs_time_col])
    log.info(f"Loaded {len(df):,} observations from {cfg.obs_path}")
    return df


def colocate_gridded_vars(
    obs: pd.DataFrame,
    ds: xr.Dataset,
    cfg: DataConfig,
) -> pd.DataFrame:
    """
    For every observation, extract co-located gridded variable values.

    Uses nearest-neighbour selection in (lon, lat, depth, time).  For large
    datasets consider chunking obs and using xarray vectorised selection instead.

    Returns obs DataFrame augmented with one column per gridded variable.
    """
    extracted: Dict[str, list] = {v: [] for v in cfg.grid_vars}

    for _, row in obs.iterrows():
        sel_kwargs = {
            cfg.lon_dim:   row[cfg.obs_lon_col],
            cfg.lat_dim:   row[cfg.obs_lat_col],
            cfg.depth_dim: row[cfg.obs_depth_col],
            cfg.time_dim:  row[cfg.obs_time_col],
        }
        for var in cfg.grid_vars:
            try:
                val = float(
                    ds[var].sel(method="nearest", **sel_kwargs).values
                )
            except Exception:
                val = np.nan
            extracted[var].append(val)

    for var in cfg.grid_vars:
        obs[var] = extracted[var]

    n_before = len(obs)
    obs = obs.dropna(subset=cfg.grid_vars + [cfg.obs_value_col])
    log.info(
        f"Co-location complete: {len(obs):,} / {n_before:,} observations retained "
        f"after dropping NaNs."
    )
    return obs.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2.  Feature engineering
# ---------------------------------------------------------------------------

def build_features(obs: pd.DataFrame, cfg: DataConfig) -> np.ndarray:
    """
    Build the feature matrix X from co-located gridded variables and,
    optionally, spatial / temporal coordinate features.

    Coordinate features help the model learn residual spatial structure not
    captured by the gridded variables alone.
    """
    feature_cols = list(cfg.grid_vars)

    if cfg.use_coords_as_features:
        # Day-of-year encoded as sin/cos to make it cyclic
        doy = obs[cfg.obs_time_col].dt.day_of_year.values
        obs = obs.copy()
        obs["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        obs["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

        # Encode lon/lat as sin/cos too (handles ±180 wrap for lon)
        obs["sin_lon"] = np.sin(np.deg2rad(obs[cfg.obs_lon_col]))
        obs["cos_lon"] = np.cos(np.deg2rad(obs[cfg.obs_lon_col]))
        obs["sin_lat"] = np.sin(np.deg2rad(obs[cfg.obs_lat_col]))
        obs["cos_lat"] = np.cos(np.deg2rad(obs[cfg.obs_lat_col]))

        # Log-depth (depths span orders of magnitude)
        obs["log_depth"] = np.log1p(obs[cfg.obs_depth_col].clip(lower=0))

        feature_cols += [
            "sin_doy", "cos_doy",
            "sin_lon", "cos_lon",
            "sin_lat", "cos_lat",
            "log_depth",
        ]

    X = obs[feature_cols].values.astype(np.float32)
    return X, feature_cols


# ---------------------------------------------------------------------------
# 3.  Normalisation  (fit on train, apply to val + grid)
# ---------------------------------------------------------------------------

class Normaliser:
    """Thin wrapper around sklearn StandardScaler saved alongside the model."""

    def __init__(self):
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Normaliser":
        self.X_scaler.fit(X)
        self.y_scaler.fit(y.reshape(-1, 1))
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return self.X_scaler.transform(X).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return self.y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(y.reshape(-1, 1)).ravel()

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Normaliser":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# 4.  PyTorch Dataset
# ---------------------------------------------------------------------------

class TracerDataset(Dataset):
    """Simple dataset wrapping normalised (X, y) arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(-1)   # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# 5.  Full preprocessing pipeline
# ---------------------------------------------------------------------------

def prepare_dataloaders(
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, Normaliser, int]:
    """
    End-to-end:  CSV + NetCDF → co-located → features → normalise → DataLoaders.

    Returns
    -------
    train_loader, val_loader, normaliser, n_features
    """
    dc = cfg.data
    tc = cfg.train

    # Load raw data
    obs = load_observations(dc)
    ds  = xr.open_dataset(dc.grid_path, chunks="auto")   # lazy Dask-backed

    # Co-locate
    obs = colocate_gridded_vars(obs, ds, dc)

    # Ensure datetime type
    obs[dc.obs_time_col] = pd.to_datetime(obs[dc.obs_time_col], errors='coerce')

    # Temporal train / val split
    split_date = pd.Timestamp(f"{dc.val_year_from}-01-01")
    train_obs  = obs[obs[dc.obs_time_col] <  split_date]
    val_obs    = obs[obs[dc.obs_time_col] >= split_date]
    log.info(f"Train: {len(train_obs):,}  Val: {len(val_obs):,}")

    # Features
    X_train, feature_names = build_features(train_obs, dc)
    X_val,   _             = build_features(val_obs,   dc)
    y_train = train_obs[dc.obs_value_col].values.astype(np.float32)
    y_val   = val_obs  [dc.obs_value_col].values.astype(np.float32)

    # Normalise (fit on train only)
    norm = Normaliser().fit(X_train, y_train)
    X_train_n = norm.transform_X(X_train)
    X_val_n   = norm.transform_X(X_val)
    y_train_n = norm.transform_y(y_train)
    y_val_n   = norm.transform_y(y_val)

    log.info(f"Feature names ({len(feature_names)}): {feature_names}")

    train_loader = DataLoader(
        TracerDataset(X_train_n, y_train_n),
        batch_size=tc.batch_size, shuffle=True,  num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        TracerDataset(X_val_n,   y_val_n),
        batch_size=tc.batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    return train_loader, val_loader, norm, X_train_n.shape[1]


# ---------------------------------------------------------------------------
# 6.  Grid feature builder (for inference)
# ---------------------------------------------------------------------------

def build_grid_features(
    ds: xr.Dataset,
    cfg: DataConfig,
    time_slice: Optional[str] = None,
) -> Tuple[np.ndarray, xr.Dataset]:
    """
    Flatten the full ocean model grid into a feature matrix ready for inference.

    Parameters
    ----------
    ds         : xarray Dataset (lon × lat × depth × time)
    cfg        : DataConfig
    time_slice : optional ISO string e.g. "2020-06" to select a single time step

    Returns
    -------
    X_grid     : (N_cells, n_features)  float32 array
    template   : xr.Dataset with the coordinate metadata for reshaping output
    """
    if time_slice is not None:
        ds = ds.sel({cfg.time_dim: time_slice}, method="nearest")

    # Stack all spatial dims into a single 'cell' dimension
    stacked = ds[cfg.grid_vars].stack(cell=(cfg.lon_dim, cfg.lat_dim, cfg.depth_dim))

    X_parts = [stacked[v].values.T for v in cfg.grid_vars]   # (N_cells, n_vars)
    X = np.column_stack(X_parts).astype(np.float32)

    if cfg.use_coords_as_features:
        lons   = stacked[cfg.lon_dim].values.astype(np.float32)
        lats   = stacked[cfg.lat_dim].values.astype(np.float32)
        depths = stacked[cfg.depth_dim].values.astype(np.float32)

        # Use a representative time if available
        if cfg.time_dim in ds.coords:
            t = pd.Timestamp(ds[cfg.time_dim].values)
            doy = t.day_of_year
        else:
            doy = 180   # fallback mid-year

        sin_doy = np.full(len(lons), np.sin(2 * np.pi * doy / 365.25), dtype=np.float32)
        cos_doy = np.full(len(lons), np.cos(2 * np.pi * doy / 365.25), dtype=np.float32)

        X = np.column_stack([
            X,
            np.sin(np.deg2rad(lons)), np.cos(np.deg2rad(lons)),
            np.sin(np.deg2rad(lats)), np.cos(np.deg2rad(lats)),
            np.log1p(np.clip(depths, 0, None)),
            sin_doy, cos_doy,
        ])

    return X, stacked
