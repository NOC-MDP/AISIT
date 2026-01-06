"""Functions for training a Machine Learning model"""

import yaml

import numpy as np
import xarray as xr

import torch
from torch import nn
from torch import optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

import joblib
import matplotlib.pyplot as plt

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# --------------------------
# Machine Learning Functions
# --------------------------


## Training Functions ##


# Neural network
class Oxygen18Net(nn.Module):
    """Class defining the neural network"""

    def __init__(self, input_dim=3, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))  # output δ18O
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Training plots
class MLModel:
    """Class to run the ML model, including training & inference"""

    def __init__(self, df_90, df_10, input_cols, target_col):
        """
        df_90 : pandas.DataFrame
            Input training data. Recommend using a subset of the full input data e.g. 90%
        df_10 : pandas.DataFrame
            Input inference data. Recommend a subset of the full input data e.g. 10% (remaining left from training)
        input_cols : list of str
            Input column names for training
        target_col : str
            Target column name
        n_models : int, default 10
            Number of models to run
        n_epochs : int, default 500
            Number of epochs to run in the training loop
        create_plots : bool, default True
            Choose whether to create plots of model weights vs epoch for each model
        save_trainingfigs : bool, default True
            Choose whether to save training plots
        save_trainingweights : bool, default True
            Choose wheter to save training weights
        save_scaler, save_fig, save_weights : bool, default True
            Options to save the scale, figures, & weights
        """
        self.df_90 = df_90
        self.df_10 = df_10
        self.input_cols = input_cols
        self.target_col = target_col
        self.n_models = 10
        self.n_epochs = 500
        self.create_plots = False
        self.save_scaler = False
        self.save_trainingfigs = False
        self.save_trainingweights = False

    def training_plots(self, train_losses, val_losses, model):
        """Plot learning curves for training"""
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Training Loss", linewidth=2)
        plt.plot(val_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(f"Training vs Validation Loss for Model {model}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if self.save_trainingfigs:
            plt.savefig(
                CONFIG["mse_plots"].format(m=model),
                bbox_inches="tight",
            )

        return None

    # Training
    def training(self):
        """
        Training function

        Returns
        -------
        scaler: sklearn.preprocessing._data.StandardScaler
            Scaler for normalisation of columns
        state_dicts: dict of torch.Tensor
            Dictionary of the weights, one for each model
        """
        # Generate inputs
        x = self.df_90[self.input_cols].values.astype(float)
        y = self.df_90[self.target_col].values.astype(float)

        # Normalise the inputs (makes ML better)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        # Option to save scaler
        if self.save_scaler:
            joblib.dump(scaler, CONFIG["scaler"])

        state_dicts = {}

        # Iterate over the models
        for i in range(self.n_models):
            train_losses, val_losses = [], []
            # Train/test split
            x_train, x_val, y_train, y_val = train_test_split(
                x_scaled, y, test_size=0.2, random_state=np.random.randint(100)
            )

            # Convert to PyTorch tensors
            x_train = torch.tensor(x_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            x_val = torch.tensor(x_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

            model = Oxygen18Net(input_dim=x_train.shape[1])

            # Set up training
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            # Training loop over epochs
            for epoch in range(self.n_epochs):
                # Training
                model.train()
                optimizer.zero_grad()
                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    y_val_pred = model(x_val)
                    val_loss = criterion(y_val_pred, y_val)

                # if epoch % 10 == 0:
                #     print(
                #         f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
                #     )

            # Output weights as a dictionary (one value for each model)
            state_dicts[i] = model.state_dict()

            # Option to create plots of MSE v epoch for each model
            if self.create_plots:
                # Create plot x-y axes
                train_losses.append(loss.item())
                val_losses.append(val_loss.item())
                # Call plotting function
                self.training_plots(train_losses, val_losses, model=i)

            # Option to save weights
            if self.save_trainingweights:
                torch.save(model.state_dict(), CONFIG["oxygen_weights"].format(m=i))
                print(f"Model weights saved to oxygen18_model_{i}.pth")

        return scaler, state_dicts

    ## Inference Functions ##

    def inference(self, state_dicts, scaler):
        """
        Inference of the training data

        Parameters
        ----------
        scaler : sklearn.preprocessing._data.StandardScaler
            Scaler used for normalisation. Pass using training()[0]
        state_dicts : dict of torch.Tensor
            Tensors of the training weights. Pass using training()[1]

        Returns
        -------
        df_out : pandas.DataFrame
            Copy of df_out with the predicted target_col added
        rmse, r2, abs_err : float
            Output of RMSE, R2 & abs. error, compared with target_col in df_10

        """
        oxygen_predictions = []
        # inference points of salinity and temperature and depth
        inference_points = self.df_10[self.input_cols].values.tolist()

        for i in range(self.n_models):
            # Create the model instance
            model = Oxygen18Net(input_dim=state_dicts[0]["net.0.weight"].size()[1])

            # Load saved weights
            model.load_state_dict(state_dicts[i])
            model.eval()  # important for inference
            # print("Model weights loaded successfully")

            # process inference points using saved scaler
            x_new = np.array(inference_points)
            x_new_scaled = scaler.transform(x_new)
            # Convert to tensor and run model
            x_new_tensor = torch.tensor(x_new_scaled, dtype=torch.float32)

            with torch.no_grad():
                delta18O_pred = model(x_new_tensor).numpy()
                # print(f"Predicted d18O for model {i+1}: {delta18O_pred}")
                oxygen_predictions.append(delta18O_pred)

        # Convert to NumPy array for easy axis operations
        arr = np.array(oxygen_predictions)
        # Average across all models for each index
        mean_values = np.mean(arr, axis=0)
        # Create new DataFrame with predicted values
        df_out = self.df_10.copy()
        df_out[f"ML_predicted_{self.target_col}"] = mean_values

        # Compute errors
        rmse = root_mean_squared_error(mean_values, self.df_10[self.target_col])
        r2 = r2_score(mean_values, self.df_10[self.target_col])
        err = mean_values.squeeze() - self.df_10[self.target_col]  # signed error
        abs_err = np.abs(err)

        print(f"ML Predicted Oxygen RMSE: {rmse}")
        print(f"ML Predicted Oxygen R2: {r2}")
        print(
            f"ML Predicted Oxygen Error Percentiles {np.percentile(abs_err, [5, 25, 50, 75, 95])}"
        )

        return df_out, rmse, r2, abs_err

    ## Apply Model Function ##
    def apply_model(
        ds, data_vars, scaler, state_dicts, x="longitude", y="latitude", xy_inp=True
    ):
        """
        Apply the model output to some data

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset containing the data variables. Must only have lon & lat as the coordinates: no time/depth
        data_vars : list of str
            List of data variable names to use as inputs to the model. Must be in the SAME ORDER as the training data
        scaler : sklearn.preprocessing._data.StandardScaler
            Scaler used for normalisation. Must be same as the training model. Pass using training()[0]
        state_dicts : dict of torch.Tensor
            Tensors of the training weights. Pass using training()[1]
        x, y : str, default 'longitude', 'latitude'
            Names of the longitude and latitude dimensions in ds
        xy_inp : bool, default True
            Choose whether to include latitude and longitude as variables used in the model input.
            If True, it's assumed that the input is [sin(lat), sin(lon), cos(lon)] (IN THAT ORDER) after the data_vars to account for cyclical coordinate system

        Returns
        -------
        da : xarray.DataArray
            DataArray of the predicted oxygen isotope tracer
        """

        # Broadcast lon/lat to same shape
        lon_vals = ds[x].values
        lat_vals = ds[y].values
        lons, lats = np.meshgrid(lon_vals, lat_vals)

        # Broadcast input variables to same shape
        var_flat = [ds[var].values.ravel() for var in data_vars]

        if xy_inp:
            # Create sine and cosine of longitude for correct wrapping
            lon_rad, lat_rad = np.deg2rad(lons), np.deg2rad(lats)
            lon_sin = np.sin(lon_rad).ravel()
            lon_cos = np.cos(lon_rad).ravel()
            lat_sin = np.sin(lat_rad).ravel()
            # lat_cos = np.cos(lat_rad).ravel()

            # Build input array in SAME ORDER as training
            X = np.column_stack([*var_flat, lat_sin, lon_sin, lon_cos])
        else:
            # Build input array in SAME ORDER as training
            X = np.column_stack([var_flat])
        # Create sine and cosine of longitude for correct wrapping
        lon_rad, lat_rad = np.deg2rad(lons), np.deg2rad(lats)
        lon_sin = np.sin(lon_rad).ravel()
        lon_cos = np.cos(lon_rad).ravel()
        lat_sin = np.sin(lat_rad).ravel()
        # lat_cos = np.cos(lat_rad).ravel()

        # Broadcast input variables to same shape
        var_flat = [ds[var].values.ravel() for var in data_vars]

        # Build input array in SAME ORDER as training
        X = np.column_stack([*var_flat, lon_sin, lon_cos, lat_sin])

        # Scale the input data
        X_scaled = scaler.transform(X)

        # Apply training model to get predictions
        preds = []
        for i, _ in enumerate(state_dicts):
            model = Oxygen18Net(input_dim=X_scaled.shape[1])
            model.load_state_dict(state_dicts[i])
            model.eval()

            with torch.no_grad():
                y = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy().squeeze()
                preds.append(y)

        mean_pred = np.mean(np.stack(preds), axis=0)

        # Build an oxygen isotope tracer DataArray
        da = xr.DataArray(
            mean_pred.reshape(ds[data_vars[0]].values.shape),
            coords=ds.coords,
            dims=ds.dims,
            name="oxygen_iso_predicted",
        )

        return da


# ------------------------
# Polynomial Fit Functions
# ------------------------


def polyplot(df_90, input_col, target_col, xfit, yfit):
    """Plot a polynomial fit"""
    plt.scatter(
        df_90[input_col],
        df_90[target_col],
        label="Data",
    )
    plt.plot(xfit, yfit, label=f"2nd-order fit", linewidth=2, color="red")
    plt.legend()
    plt.show()
    return None


def poly_fit(df_90, df_10, input_col, target_col, create_plot=True):
    """
    Perform a polynomial fit of the a subset of the input data and infer with the rest

    Parameters
    ----------
    df_90, df_10 : pandas.DataFrame
        Input data. df_90 is for the `training' i.e. generating fit parameters and df_10 is for the `inference'
    input_col, target_col : str
        Name of input and target columns. Note for polynomial fit, only one input column can be input
    create_plot : bool, default True
        Option to create a plot of the fit

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of df_out with the predicted target_col added
    rmse, r2, abs_err : float
        Output of RMSE, R2 & abs. error, compared with target_col in df_10
    yfit, xfit : numpy.ndarray
        Arrays for the y & x-values of the fit


    """
    # `Training`
    # Fit a 2nd-order polynomial: coeffs are [a, b, c] for ax² + bx + c
    degree = 2
    coeffs = np.polyfit(df_90[input_col], df_90[target_col], degree)

    # Create a polynomial function
    poly = np.poly1d(coeffs)

    # Evaluate fit
    xfit = np.linspace(
        df_90[input_col].min(),
        df_90[input_col].max(),
        2000,
    )
    yfit = poly(xfit)

    if create_plot:
        polyplot(df_90, input_col, target_col, xfit, yfit)

    # `Inference`
    # Compute predicted oxygen & create DataFrame
    predicted_oxygen = (
        coeffs[0] * df_10[input_col] ** 2 + coeffs[1] * df_10[input_col] + coeffs[2]
    )

    df_out = df_10.copy()
    df_out[f"poly_predicted_{target_col}"] = predicted_oxygen.values

    # Compute errors
    rmse = root_mean_squared_error(predicted_oxygen, df_10[target_col])
    r2 = r2_score(predicted_oxygen, df_10[target_col])
    err = predicted_oxygen.squeeze() - df_10[target_col]  # signed error
    abs_err = np.abs(err)
    print(
        "Poly Predicted Oxygen RMSE:",
        rmse,
    )
    print(
        "Poly Predicted Oxygen R2:",
        r2,
    )
    print(
        f"Poly Predicted Oxygen Error Percentiles {np.percentile(abs_err, [5, 25, 50, 75, 95])}"
    )
    return df_out, rmse, r2, abs_err, xfit, yfit
