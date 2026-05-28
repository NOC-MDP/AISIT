import xarray as xr
import glob
import numpy as np
import os 
import icechunk as ic

import configparser

def preprocess(ds):
    # Extract the scalar time value and expand it as a new dimension
    # Replace "time" with whatever your time variable is actually called
    time_val = ds["time"].values  # scalar or 1-element array
    ds = ds.expand_dims({"time": np.atleast_1d(time_val)})
    return ds

# Grab all files - adjust pattern to match your naming convention
files = sorted(glob.glob("/gws/ssde/j25a/nemo/vol4/thopri/PINN/infer_output/tracer_predicted_*.nc"))  # sorting ensures time order
# files = sorted(glob.glob("/gws/ssde/j25a/nemo/vol4/thopri/PINN_BARIUM/infer_output/tracer_predicted_*.nc"))  # sorting ensures time order
ds = xr.open_mfdataset(
    files,
    combine="nested",
    concat_dim="time",
    preprocess=preprocess,
    parallel=True,
    engine="h5netcdf"
)

print(ds)  # check it looks right before writing

# Rechunk before writing
ds_chunked = ds.chunk({
    "time": 12,      # e.g. 1 year of months per chunk
    "depth": 40,
    "latitude": 32,
    "longitude": 90
})


# Read credentials from ~/.s3cfg
config = configparser.ConfigParser()
config.read(os.path.expanduser("~/.s3cfg"))

key    = config["default"]["access_key"]
secret = config["default"]["secret_key"]
host   = config["default"]["host_base"] 

storage = ic.s3_storage(
bucket="conservative-tracers",
prefix="d18O",
# prefix="Ba",
access_key_id=key,
secret_access_key=secret,
endpoint_url = f"http://{host}",
force_path_style=True, 
)

repo_config2 = ic.RepositoryConfig(
storage = ic.StorageSettings(
    unsafe_use_conditional_update=False,
    unsafe_use_conditional_create=False,
)
)

repo = ic.Repository.create(storage=storage, config=repo_config2)

session = repo.writable_session("main")

store = session.store

ds_chunked.to_zarr(
    store,
    mode="w"
)

# session.commit("Initial Ba_AOR dataset")
session.commit("Initial d18O_AOR dataset")