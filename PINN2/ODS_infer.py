# /// script
# dependencies = [
#     "marimo>=0.23.14",
#     "numpy==2.5.0",
#     "torch==2.12.1",
#     "xarray==2026.4.0",
# ]
# [tool.marimo.venv]
# path = "/home/users/thopri/micromamba/envs/AISIT2"      
# writable = false 
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import torch
    import xarray as xr
    from OceanDataStore import OceanDataCatalog
    from nemo_cookbook import NEMODataTree
    import torch
    import gsw
    from sklearn.preprocessing import StandardScaler
    from pinn_tracer import TracerPINN
    import os
    import sys

    return (
        NEMODataTree,
        OceanDataCatalog,
        StandardScaler,
        TracerPINN,
        gsw,
        np,
        os,
        sys,
        torch,
        xr,
    )


@app.cell
def _(np, torch):
    Config = {
        "start_year": 1976,
        "end_year": 1976,
        "bbox":  (-180, 180, 60, 90),
        "model": "/home/users/thopri/AISIT/PINN2/output_v1.0.0/pinn_tracer.pt",
        "seed": 42
    }
    np.random.seed(Config['seed'])
    torch.manual_seed(Config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config['seed'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    return Config, device


@app.cell
def _(Config, device, load_checkpoint):
    # ── Load checkpoint ───────────────────────────────────────────────────────
    print("── Loading checkpoint ──────────────────────────────────────")
    model, scaler, feat_names = load_checkpoint(Config["model"], device)
    return feat_names, model, scaler


@app.cell
def _(Config, NEMODataTree, OceanDataCatalog):
    catalog = OceanDataCatalog(catalog_name="noc-stac")

    ds_gridT = catalog.open_dataset(id='noc-npd-era5/npd-eorca12-era5v1/r1i1c1f1/T1m_4d',
                         start_datetime=f'{Config['start_year']}-01',
                         end_datetime=f'{Config['end_year']}-12',
                                   )


    ds_gridU = catalog.open_dataset(id='noc-npd-era5/npd-eorca12-era5v1/r1i1c1f1/U1m_4d',
                         start_datetime=f'{Config['start_year']}-01',
                         end_datetime=f'{Config['end_year']}-12',
                                   )

    ds_gridV = catalog.open_dataset(id='noc-npd-era5/npd-eorca12-era5v1/r1i1c1f1/V1m_4d',
                         start_datetime=f'{Config['start_year']}-01',
                         end_datetime=f'{Config['end_year']}-12',
                                   )

    ds_domain = catalog.open_dataset(id='noc-npd-era5/npd-eorca12-era5v1/r1i1c1f1/domain/domain_cfg')


    datasets = {"parent": {"domain": ds_domain, "gridT": ds_gridT,"gridU":ds_gridU,"gridV":ds_gridV}}

    nemo = NEMODataTree.from_datasets(datasets=datasets, read_mask=True)

    nemo = nemo.clip_domain(dom=".", bbox=Config['bbox'])

    nemo["gridT/uo"] = nemo["gridU/uo"].interp_to(to="T")

    nemo["gridT/vo"] = nemo["gridV/vo"].interp_to(to="T")

    print(nemo)
    return (nemo,)


@app.cell
def _(model, nemo, np, predict_tracer_ufunc, scaler, xr):
    # 1. Target the gridT node
    # 1. Target the gridT node
    target_node = nemo["gridT"]

    # 2. Extract and rechunk the core 4D arrays to keep vertical columns whole
    SA = target_node["so_abs"].chunk({"k": -1})
    CT = target_node["thetao_con"].chunk({"k": -1})

    # Grab the pre-aligned velocity components from your cook-book setup
    u_aligned = nemo["gridT/uo"].chunk({"k": -1})
    v_aligned = nemo["gridT/vo"].chunk({"k": -1})

    # 3. Grab coordinates
    lat_2d = target_node["gphit"]  # (j, i)
    lon_2d = target_node["glamt"]  # (j, i)

    # Broadcast coordinates so they cleanly match the 3D spatial layout shape
    _, _, z_3d = xr.broadcast(lat_2d, lon_2d, target_node["deptht"])
    z_3d = z_3d.chunk({"k": -1})

    # 4. Execute apply_ufunc (Pass time_counter explicitly)
    tracer_pred, uncert_pred = xr.apply_ufunc(
        predict_tracer_ufunc,
        model,
        scaler,
        SA,
        CT,
        u_aligned,
        v_aligned,
        lat_2d,
        lon_2d,
        z_3d,
        target_node["time_counter"], 
        input_core_dims=[
            [],  # 1. model (no dimensions)
            [],  # 2. scaler (no dimensions)
            ["time_counter", "k", "j", "i"],  # 3. SA
            ["time_counter", "k", "j", "i"],  # 4. CT
            ["time_counter", "k", "j", "i"],  # 5. u_aligned
            ["time_counter", "k", "j", "i"],  # 6. v_aligned
            ["j", "i"],  # 7. lat_2d
            ["j", "i"],  # 8. lon_2d
            ["k", "j", "i"],  # 9. z_3d
            ["time_counter"],  # 10. time_counter array
        ],
        output_core_dims=[
            ["time_counter", "k", "j", "i"],
            ["time_counter", "k", "j", "i"],
        ],
        vectorize=False,
        dask="parallelized",
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "meta": (np.array([], dtype=np.float32), np.array([], dtype=np.float32)),
            "output_chunks": {
                "time_counter": 1,
                "k": -1,
                "j": -1,
                "i": -1,
            },
        },
    )

    # 5. Save the lazy definitions back to the tree metadata
    target_node["predicted_tracer"] = tracer_pred
    target_node["prediction_uncertainty"] = uncert_pred

    # 6. Trigger the PyTorch Pipeline loop for a single month slice to test
    print("Starting ML prediction pipeline execution...")
    sample_prediction = target_node["predicted_tracer"].isel(time_counter=0).compute()
    print("Computation successful!")
    print(sample_prediction)
    return


@app.cell
def _(device, feat_names, gsw, np, torch):
    def predict_tracer_ufunc(model, scaler,SA_3d, CT_3d, u_3d, v_3d, lat_3d, lon_3d, z_3d,time_coords, **kwargs):
        """This function receives raw 3D/4D numpy arrays from xarray,

        runs your ML pipeline, and returns the predictions reshaped back to the
        input shape.
        """
        # 1. Keep track of the original shape to reconstruct the grid later
        orig_shape = SA_3d.shape  # E.g., (1, 75, 693, 3665)
        n_times = orig_shape[0]  # Usually 1 month per chunk given output_chunks configuration
        spatial_pts = (
            orig_shape[1] * orig_shape[2] * orig_shape[3]
        )  # Number of 3D grid cells per month

        # 1. Extract years and months from the NumPy datetime64 array
        # time_coords arrives as a 1D numpy array of np.datetime64 type
        times = time_coords.astype("datetime64[M]")
        years = times.astype("datetime64[Y]").astype(int) + 1970
        months = (times.astype(int) % 12) + 1  # 1-12 range

        # 2. Flatten inputs to 1D vectors
        SA = SA_3d.ravel()
        CT = CT_3d.ravel()
        u_flat = u_3d.ravel()
        v_flat = v_3d.ravel()

        # Repeat spatial coordinates across the time dimension to match 4D flattened size
        lat_f = np.tile(lat_3d.ravel(), n_times)
        lon_f = np.tile(lon_3d.ravel(), n_times)
        z_flat = np.tile(z_3d.ravel(), n_times)

        # ── NEW DYNAMIC TIME LOGIC ──────────────────────────────────────────
        # Create matching 1D vectors for year/month by repeating each timestamp's info
        # across all its corresponding spatial points.
        chunk_years = np.repeat(years, spatial_pts)
        chunk_months = np.repeat(months, spatial_pts)

        year_norm = (chunk_years - 1970) / 53.0
        season_sin = np.sin(2 * np.pi * chunk_months / 12)
        season_cos = np.cos(2 * np.pi * chunk_months / 12)

        # ── YOUR EXACT MATH LOGIC ───────────────────────────────────────────
        p_flat = gsw.p_from_z(-z_flat, lat_f)
        sigma0 = gsw.sigma0(SA, CT)
        spice = gsw.spiciness0(SA, CT)
        sigma2 = gsw.sigma2(SA, CT)

        alpha = gsw.alpha(SA, CT, p_flat)
        beta = gsw.beta(SA, CT, p_flat)
        rho = gsw.rho(SA, CT, p_flat)
        n2_proxy = np.maximum(9.7963 / rho * rho * (alpha + beta) / 100.0, 1e-8)
        log_n2 = np.log10(n2_proxy)

        eps_v = 1e-10
        speed = np.sqrt(u_flat**2 + v_flat**2)
        flow_sin = np.where(speed > eps_v, v_flat / (speed + eps_v), 0.0)
        flow_cos = np.where(speed > eps_v, u_flat / (speed + eps_v), 0.0)
        log_speed = np.log1p(speed)

        feature_map = {
            "sigma0": sigma0,
            "spice": spice,
            "sigma2": sigma2,
            "log_depth": np.log1p(z_flat),
            "log_n2": log_n2,
            "year_norm": year_norm,  # Now a dynamic array matching data size!
            "season_sin": season_sin,  # Now a dynamic array matching data size!
            "season_cos": season_cos,  # Now a dynamic array matching data size!
            "lat_norm": np.sin(np.deg2rad(lat_f)),
            "lon_sin": np.sin(np.deg2rad(lon_f)),
            "lon_cos": np.cos(np.deg2rad(lon_f)),
            "log_speed": log_speed,
            "flow_sin": flow_sin,
            "flow_cos": flow_cos,
        }

        X_model = np.column_stack([feature_map[f] for f in feat_names])

        # ── Mask & Predict (Your exact loop) ─────────────────────────────────
        valid = np.isfinite(X_model).all(axis=1)

        X_valid = scaler.transform(X_model[valid])

        # ── MC-Dropout inference ──────────────────────────────────────────────────
        model.train()

        batch_size = 100_000  # tune this (start lower if needed)
        n_mc = 50  

        n = X_valid.shape[0]
        mean = np.zeros(n, dtype=np.float32)
        sq_mean = np.zeros(n, dtype=np.float32)

        for i in range(0, n, batch_size):
            X_batch_np = X_valid[i : i + batch_size]

            X_batch = torch.tensor(X_batch_np, dtype=torch.float32).to(device)

            mc_preds = []

            for _ in range(n_mc):
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    y = model(X_batch).cpu().numpy()
                mc_preds.append(y)

            mc_preds = np.stack(mc_preds)  # (n_mc, batch)

            mean[i : i + batch_size] = mc_preds.mean(axis=0)
            sq_mean[i : i + batch_size] = (mc_preds**2).mean(axis=0)

            del X_batch, mc_preds
            torch.cuda.empty_cache()

        tracer_mean = mean
        tracer_std = np.sqrt(sq_mean - mean**2)

        tracer_field = np.full(valid.size, np.nan)
        uncert_field = np.full(valid.size, np.nan)
        tracer_field[valid] = tracer_mean
        uncert_field[valid] = tracer_std

        # 3. Reshape back to the original incoming 3D/4D grid structure
        return tracer_field.reshape(orig_shape), uncert_field.reshape(orig_shape)

    return (predict_tracer_ufunc,)


@app.cell
def _(StandardScaler, TracerPINN, os, sys, torch):
    def load_checkpoint(path, device):
        """
        Load a .pt checkpoint saved by main.py and reconstruct the model + scaler.

        Checkpoint schema (written by main.py):
            model_state   : nn.Module state_dict
            feat_names    : list[str]
            hidden_dim    : int
            n_blocks      : int
            n_features    : int
            scaler_mean   : np.ndarray
            scaler_scale  : np.ndarray
        """
        if not os.path.isfile(path):
            sys.exit(f"[ERROR] Checkpoint not found: {path}")

        ckpt = torch.load(path, weights_only=False, map_location=device)

        required_keys = {
            "model_state",
            "feat_names",
            "hidden_dim",
            "n_blocks",
            "n_features",
            "scaler_mean",
            "scaler_scale",
        }
        missing = required_keys - set(ckpt.keys())
        if missing:
            sys.exit(
                f"[ERROR] Checkpoint is missing keys: {missing}\n"
                f"        Make sure you are loading a checkpoint saved by main.py."
            )

        # Reconstruct model
        model = TracerPINN(
            n_features=ckpt["n_features"],
            hidden_dim=ckpt["hidden_dim"],
            n_blocks=ckpt["n_blocks"],
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Reconstruct scaler (sklearn StandardScaler shell, no re-fitting needed)
        scaler = StandardScaler()
        scaler.mean_ = ckpt["scaler_mean"]
        scaler.scale_ = ckpt["scaler_scale"]
        scaler.n_features_in_ = ckpt["n_features"]

        feat_names = ckpt["feat_names"]

        print(f"  Checkpoint   : {path}")
        print(
            f"  Architecture : hidden_dim={ckpt['hidden_dim']}, "
            f"n_blocks={ckpt['n_blocks']}, n_features={ckpt['n_features']}"
        )
        print(f"  Features     : {feat_names}")
        return model, scaler, feat_names

    return (load_checkpoint,)


if __name__ == "__main__":
    app.run()
