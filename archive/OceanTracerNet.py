import torch
import torch.nn as nn
import math
from attrs import define
import os
import pandas as pd
import numpy as np

class FourierEncoding(nn.Module):
    """Fourier positional encoding for (x, y, z)."""
    def __init__(self, num_input_dims=3, num_frequencies=8):
        super().__init__()
        self.num_input_dims = num_input_dims
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, x):
        # x shape: [batch, 3]
        enc = [x]  # keep raw coords
        for f in self.freq_bands:
            enc.append(torch.sin(x * f * math.pi))
            enc.append(torch.cos(x * f * math.pi))
        return torch.cat(enc, dim=-1)


class OceanTracerNet(nn.Module):
    """
    MLP for regressing an ocean tracer from:
        (x, y, z) + (temperature, salinity)
    """
    def __init__(self, hidden_dim=128, num_frequencies=8):
        super().__init__()

        # Positional encoding handles 3 dims → (x, y, depth)
        self.encoder = FourierEncoding(num_input_dims=3,
                                       num_frequencies=num_frequencies)

        encoded_dim = 3 + 2 * 3 * num_frequencies  # (raw + sin + cos)
        physical_dim = 2  # temperature + salinity

        input_dim = encoded_dim + physical_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),

            nn.Linear(hidden_dim // 4, 1)  # tracer output
        )

    def forward(self, coords, phys):
        """
        coords: [batch, 3]  -> (x, y, z)
        phys:   [batch, 2]  -> (temp, sal)
        """
        enc = self.encoder(coords)
        x = torch.cat([enc, phys], dim=-1)
        return self.net(x)



import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

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
    model_dir ="../glodap_models",
    input_data = "window_data_from_GLODAPv2.2023.csv",
    header = 0,
    salinity_field = "SALNTY [PSS-78]",
    temperature_field = "TEMPERATURE [DEG C]",
    oxygen_iso_field = "O18/O16 [/MILLE]",
    depth_field = "DEPTH [M]",
)

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

# =======================================================================
# 1. Prepare Data
# =======================================================================

# coords: [N, 3]
# phys:   [N, 2]
# target: [N, 1]
coords = torch.tensor(
    df[["Longitude [degrees East]", "Latitude [degrees North]", ML_dataset.depth_field]].values,
    dtype=torch.float32
)

phys = torch.tensor(
    df[[ML_dataset.temperature_field, ML_dataset.salinity_field]].values,
    dtype=torch.float32
)

target = torch.tensor(
    df[ML_dataset.oxygen_iso_field].values.reshape(-1, 1),
    dtype=torch.float32
)

dataset = TensorDataset(coords, phys, target)
loader = DataLoader(dataset, batch_size=128, shuffle=True)


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================================================================
# 2. Initialize Model, Loss, Optimizer
# =======================================================================

model = OceanTracerNet(hidden_dim=128, num_frequencies=8).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

# Optional learning rate scheduler (highly recommended)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=10
)

# =======================================================================
# 3. Training Loop
# =======================================================================

n_epochs = 200
best_loss = float("inf")

for epoch in range(1, n_epochs + 1):
    model.train()
    running_loss = 0.0

    for batch_coords, batch_phys, batch_target in loader:
        batch_coords = batch_coords.to(device)
        batch_phys = batch_phys.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(batch_coords, batch_phys)

        # Compute loss
        loss = criterion(output, batch_target)

        # Backprop
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_coords.size(0)

    epoch_loss = running_loss / len(dataset)
    scheduler.step(epoch_loss)

    print(f"Epoch {epoch:03d} | Loss = {epoch_loss:.6f}")

    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), f"{ML_dataset.model_dir}/best_ocean_tracer_net.pt")

print("Training complete.")

model.load_state_dict(torch.load(f"{ML_dataset.model_dir}/best_ocean_tracer_net.pt", map_location=device))
model.eval()  # important for inference

# -------------------------------
# 2. Prepare new data
# -------------------------------

# Example: a few points you want to predict
new_data = pd.DataFrame({
    "Longitude": [140.168],
    "Latitude": [77.888],
    "Depth": [175],
    "Salinity": [34.114],
    "Temperature": [-1.247],
})

coords = torch.tensor(
    new_data[["Longitude", "Latitude", "Depth"]].values,
    dtype=torch.float32
).to(device)

phys = torch.tensor(
    new_data[["Salinity", "Temperature"]].values,
    dtype=torch.float32
).to(device)

# -------------------------------
# 3. Run inference
# -------------------------------

with torch.no_grad():  # disables gradients
    predictions = model(coords, phys)

# Convert back to numpy for further use
predictions = predictions.cpu().numpy()
print("Predicted tracer values:", predictions)