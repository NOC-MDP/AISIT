import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
import copy
import random
import joblib
import matplotlib.pyplot as plt
import os
from attrs import define


# -------------------------------
# 0. Reproducibility if uncommented will make models produce same inference values
# -------------------------------
#seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)

@define
class DatasetSpec:
    model_dir: str
    input_data: str
    salinity_field: str
    temperature_field: str
    oxygen_iso_field: str
    depth_field: str
    header: int = 0

    def __attrs_post_init__(self):
        os.makedirs(self.model_dir, exist_ok=True)


ML_dataset = DatasetSpec(
    model_dir = "glodap_models",
    input_data = "window_data_from_GLODAPv2.2023.csv",
    header = 0,
    salinity_field = "SALNTY [PSS-78]",
    temperature_field = "TEMPERATURE [DEG C]",
    oxygen_iso_field = "O18/O16 [/MILLE]",
    depth_field = "DEPTH [M]",
)

n_models = 10

# -------------------------------
# 1. Load and process CSV data
# -------------------------------
df = pd.read_csv(f"input_data/{ML_dataset.input_data}", header=ML_dataset.header)
df_len = df.__len__()
# Replace '**' with NaN and drop missing values
df.replace("**", np.nan, inplace=True)
df = df.dropna(subset=[ML_dataset.salinity_field, ML_dataset.temperature_field, ML_dataset.oxygen_iso_field,ML_dataset.depth_field])
df_new_len = df.__len__()
print(f"removed {df_len-df_new_len} rows out of {df_len} rows")

df_10 = df.sample(frac=0.10)   # 10% sample
df_90 = df.drop(df_10.index) # get rest

# Extract inputs and target
X = df_90[[ML_dataset.salinity_field, ML_dataset.temperature_field, ML_dataset.depth_field]].values.astype(float)
y = df_90[ML_dataset.oxygen_iso_field].values.astype(float)

# -------------------------------
# 2. Normalize inputs
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, f"{ML_dataset.model_dir}/scaler.save")

# -------------------------------
# 3. Define the neural network
# -------------------------------
class Oxygen18Net(nn.Module):
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


for i in range(n_models):
    train_losses, val_losses = [], []
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=np.random.randint(100)
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    model = Oxygen18Net(input_dim=X_train.shape[1])

    # -------------------------------
    # 4. Training setup
    # -------------------------------
    criterion = nn.SmoothL1Loss() #nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 1000

    # -------------------------------
    # 5. Training loop
    # -------------------------------
    for epoch in range(n_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        # Store for plotting
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

    # ---- Plot learning curves ----
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training vs Validation Loss for Model {i}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Save the model weights to a file
    torch.save(model.state_dict(), f"{ML_dataset.model_dir}/oxygen18_model_{i}.pth")
    print(f"Model weights saved to oxygen18_model_{i}.pth")


# -------------------------------
# 6. Inference
# -------------------------------
oxygen_predictions = []
# inference points of salinity and temperature and depth
inference_points = df_10[[ML_dataset.salinity_field, ML_dataset.temperature_field, ML_dataset.depth_field]].values.tolist()

for i in range(n_models):
    # Create the model instance
    model = Oxygen18Net()

    # Load saved weights
    model.load_state_dict(torch.load(f"{ML_dataset.model_dir}/oxygen18_model_{i}.pth"))
    model.eval()  # important for inference
    # print("Model weights loaded successfully")

    # process inference points using saved scaler
    X_new = np.array(inference_points)
    scaler = joblib.load(f"{ML_dataset.model_dir}/scaler.save")
    X_new_scaled = scaler.transform(X_new)
    # Convert to tensor and run model
    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

    with torch.no_grad():
        delta18O_pred = model(X_new_tensor).numpy()
        # print(f"Predicted d18O for model {i+1}: {delta18O_pred}")
        oxygen_predictions.append(delta18O_pred)

# Convert to NumPy array for easy axis operations
arr = np.array(oxygen_predictions)
# Average across all sublists for each index
mean_values = np.mean(arr, axis=0)
print(f"ML Predicted Oxygen RMSE: {root_mean_squared_error(mean_values, df_10[ML_dataset.oxygen_iso_field])}")
print(f"ML Predicted Oxygen R2: {r2_score(mean_values, df_10[ML_dataset.oxygen_iso_field])}")


# for i in range(mean_values.__len__()):
#     print(f"Mean Oxygen prediction for point {i}: {mean_values[i][0]:.3f}")

# Test against simple poly fit
# Fit a 2nd-order polynomial
degree = 2
coeffs = np.polyfit(df_90[ML_dataset.salinity_field], df_90[ML_dataset.oxygen_iso_field], degree)

# coeffs are [a, b, c] for ax² + bx + c
# print("Coefficients:", coeffs)

# Create a polynomial function
poly = np.poly1d(coeffs)

# Evaluate fit
xfit = np.linspace(df_90[ML_dataset.salinity_field].min(), df_90[ML_dataset.salinity_field].max(), 2000)
yfit = poly(xfit)

# Plot
plt.scatter(df_90[ML_dataset.salinity_field], df_90[ML_dataset.oxygen_iso_field], label="Data")
plt.plot(xfit, yfit, label=f"{degree}-degree fit", linewidth=2,color="red")
plt.legend()
plt.show()


predicted_oxygen = coeffs[0] * df_10[ML_dataset.salinity_field]**2 + coeffs[1] * df_10[ML_dataset.salinity_field] + coeffs[2]
print("Poly Predicted Oxygen RMSE:", root_mean_squared_error(predicted_oxygen,df_10[ML_dataset.oxygen_iso_field]))
print("Poly Predicted Oxygen R2:", r2_score(predicted_oxygen, df_10[ML_dataset.oxygen_iso_field]))