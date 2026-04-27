"""
config.py — Central configuration for tracer inference pipeline.
Adjust paths, variable names, and hyperparameters here.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    # --- File paths ---
    obs_path: str       = "../data/bas_data/pre_alpha_AISIT_0.1.4.csv"   # sparse tracer observations
    grid_path: str      = "/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_thetao-so_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-01-01.nc"     # gridded ocean model (xarray-readable)

    # --- Observation columns ---
    obs_lon_col: str    = "Longitude_[degE]"
    obs_lat_col: str    = "Latitude_[degN]"
    obs_depth_col: str  = "combined_depth_[meters_below_surface]"
    obs_time_col: str   = "Sample_Date_[dd/mm/yyyy]"
    obs_value_col: str  = "DELO18_[permillle]"                  # the conservative tracer to predict

    # --- Gridded variable names (as they appear in the NetCDF) ---
    grid_vars: List[str] = field(default_factory=lambda: [
        "thetao",   # potential temperature
        "so",       # salinity
        # "uo",       # zonal velocity  (optional)
        # "vo",       # meridional velocity (optional)
    ])

    # --- Grid dimension names ---
    lon_dim: str        = "longitude"
    lat_dim: str        = "latitude"
    depth_dim: str      = "depth"
    time_dim: str       = "time"

    # --- Spatial features to include alongside gridded vars ---
    use_coords_as_features: bool = True   # append (lon, lat, depth, doy) to features

    # --- Train / validation temporal split ---
    val_year_from: int  = 2010            # observations from this year onward → validation


@dataclass
class ModelConfig:
    hidden_dims: List[int]  = field(default_factory=lambda: [128, 256, 256, 128])
    dropout: float          = 0.1
    activation: str         = "gelu"      # "relu" | "gelu" | "silu"


@dataclass
class TrainConfig:
    batch_size: int     = 512
    lr: float           = 3e-4
    weight_decay: float = 1e-5
    epochs: int         = 200
    patience: int       = 20             # early stopping patience
    scheduler: str      = "cosine"       # "cosine" | "plateau"
    device: str         = "cuda"         # "cuda" | "mps" | "cpu"
    seed: int           = 42
    checkpoint_path: str = "checkpoints/best_model.pt"


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
