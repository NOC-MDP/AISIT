"""
Spatiotemporal Gap-Filling Model
=================================
Transforms sparse (lat, lon, time, value) observations into dense gridded fields.

Architecture:
  1. Fourier positional encoding  — map continuous coordinates to feature vectors
  2. Observation encoder (MLP)    — embed each sparse observation independently
  3. Cross-attention to grid      — grid positions query encoded observations
  4. U-Net refinement             — enforce spatial coherence and long-range structure

Usage:
  model = SpatiotemporalGapFiller(coord_dim=3, value_dim=1, grid_size=(64, 64))
  pred  = model(obs_coords, obs_values, query_time=t, obs_mask=mask)
  # pred: (B, out_channels, H, W)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 1. Fourier Positional Encoding
# ---------------------------------------------------------------------------

class FourierEncoding(nn.Module):
    """
    Random Fourier Feature encoding for continuous coordinates.

    Maps an input vector x ∈ R^d to [sin(B x), cos(B x)] ∈ R^{2m}, where B is
    a learnable (or fixed) frequency matrix. Works well for smooth spatial and
    temporal signals.

    Args:
        input_dim:    dimensionality of input coordinates
        num_features: number of frequency pairs (output dim = 2 * num_features)
        learnable:    if True, frequencies are learned; otherwise fixed at init
        scale:        std of initial random frequencies (tune per domain)
    """
    def __init__(self, input_dim: int, num_features: int = 64,
                 learnable: bool = False, scale: float = 1.0):
        super().__init__()
        freqs = torch.randn(input_dim, num_features) * scale
        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 2 * self.freqs.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dim) -> (..., 2 * num_features)
        proj = x @ self.freqs          # (..., num_features)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ---------------------------------------------------------------------------
# 2. Observation Encoder
# ---------------------------------------------------------------------------

class ObservationEncoder(nn.Module):
    """
    Encodes each sparse observation (coordinate + value) independently into a
    latent embedding via a small MLP. Observations are processed in parallel —
    there is no cross-observation interaction at this stage.

    Args:
        coord_dim:       spatial + temporal coordinate dimension (e.g. 3 for lat/lon/t)
        value_dim:       number of observed channels
        embed_dim:       output embedding dimension
        fourier_features: Fourier feature pairs per coordinate dimension
        scale:           frequency scale (increase for high-frequency signals)
    """
    def __init__(self, coord_dim: int, value_dim: int, embed_dim: int,
                 fourier_features: int = 64, scale: float = 1.0):
        super().__init__()
        self.pos_enc = FourierEncoding(coord_dim, fourier_features, scale=scale)
        in_dim = self.pos_enc.out_dim + value_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, coords: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, N, coord_dim)   — coordinates of each observation
        values: (B, N, value_dim)   — observed values

        Returns: (B, N, embed_dim)
        """
        pos_feats = self.pos_enc(coords)                      # (B, N, pos_dim)
        x = torch.cat([pos_feats, values], dim=-1)            # (B, N, pos_dim + val_dim)
        return self.norm(self.mlp(x))                         # (B, N, embed_dim)


# ---------------------------------------------------------------------------
# 3. Cross-Attention: grid positions → observations
# ---------------------------------------------------------------------------

class CrossAttentionGridLayer(nn.Module):
    """
    One cross-attention + FFN layer where grid point positions act as queries
    and encoded observations act as keys/values.

    This is the core operation that aggregates information from scattered
    observation points to each regular grid location.

    Args:
        coord_dim:       coordinate dimensionality (same as ObservationEncoder)
        embed_dim:       embedding/attention dimension
        num_heads:       number of attention heads
        fourier_features: Fourier feature pairs for grid position encoding
        dropout:         dropout rate on attention weights
    """
    def __init__(self, coord_dim: int, embed_dim: int, num_heads: int = 8,
                 fourier_features: int = 64, dropout: float = 0.1):
        super().__init__()
        self.pos_enc = FourierEncoding(coord_dim, fourier_features)

        # Project grid positional features to query space
        self.query_proj = nn.Linear(self.pos_enc.out_dim, embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        grid_coords: torch.Tensor,
        obs_embeddings: torch.Tensor,
        obs_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        grid_coords:    (B, H*W, coord_dim)  — flattened grid query positions
        obs_embeddings: (B, N, embed_dim)    — encoded observation tokens
        obs_mask:       (B, N) bool          — True = padding (no observation)

        Returns: (B, H*W, embed_dim)
        """
        # Build queries from grid positions
        queries = self.query_proj(self.pos_enc(grid_coords))  # (B, H*W, D)

        # Cross-attention (grid queries attend to observation keys/values)
        attn_out, _ = self.cross_attn(
            query=queries,
            key=obs_embeddings,
            value=obs_embeddings,
            key_padding_mask=obs_mask,   # (B, N) True → masked out
        )
        x = self.norm1(queries + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# 4. U-Net Refinement
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetRefinement(nn.Module):
    """
    Lightweight U-Net that refines the coarse grid features produced by
    cross-attention. Skip connections preserve fine-scale structure while
    the bottleneck captures long-range dependencies.

    Input/output: (B, embed_dim, H, W) → (B, out_channels, H, W)

    Note: H and W must be divisible by 8 for the 3-level pooling.
    """
    def __init__(self, embed_dim: int, out_channels: int = 1, base_ch: int = 64):
        super().__init__()
        C = base_ch

        self.input_proj = nn.Conv2d(embed_dim, C, 1)

        # Encoder
        self.enc1 = _DoubleConv(C,     C)
        self.enc2 = _DoubleConv(C,     C * 2)
        self.enc3 = _DoubleConv(C * 2, C * 4)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _DoubleConv(C * 4, C * 8)

        # Decoder
        self.up3  = nn.ConvTranspose2d(C * 8, C * 4, 2, stride=2)
        self.dec3 = _DoubleConv(C * 8, C * 4)
        self.up2  = nn.ConvTranspose2d(C * 4, C * 2, 2, stride=2)
        self.dec2 = _DoubleConv(C * 4, C * 2)
        self.up1  = nn.ConvTranspose2d(C * 2, C,     2, stride=2)
        self.dec1 = _DoubleConv(C * 2, C)

        self.head = nn.Conv2d(C, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.input_proj(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b  = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


# ---------------------------------------------------------------------------
# 5. Full Model
# ---------------------------------------------------------------------------

class SpatiotemporalGapFiller(nn.Module):
    """
    End-to-end gap-filling model.

    Takes sparse (lat, lon, time, value) observations and produces a dense
    gridded field at a specified query time.

    Args:
        coord_dim:        coordinate dimensionality (default 3: lat/lon/time)
        value_dim:        number of observed channels (e.g. 1 for temperature)
        out_channels:     number of output grid channels
        embed_dim:        latent dimension for attention (power of 2 recommended)
        grid_size:        (H, W) of output grid; H, W must be divisible by 8
        num_attn_layers:  number of stacked cross-attention layers
        num_heads:        attention heads (embed_dim must be divisible by num_heads)
        fourier_features: Fourier feature pairs per coordinate
        fourier_scale:    frequency scale (increase for high-frequency signals)
        unet_base_ch:     base channel width for U-Net
        dropout:          dropout on attention weights

    Forward signature:
        obs_coords:  (B, N, coord_dim)  — (lat, lon, t) of each sparse obs
        obs_values:  (B, N, value_dim)  — observed values
        query_time:  (B,) float | None  — output time; defaults to mean obs time
        obs_mask:    (B, N) bool | None — True = padded slot (no observation)

    Returns: (B, out_channels, H, W)
    """
    def __init__(
        self,
        coord_dim: int = 3,
        value_dim: int = 1,
        out_channels: int = 1,
        embed_dim: int = 128,
        grid_size: tuple[int, int] = (64, 64),
        num_attn_layers: int = 2,
        num_heads: int = 8,
        fourier_features: int = 64,
        fourier_scale: float = 1.0,
        unet_base_ch: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert all(s % 8 == 0 for s in grid_size), "H and W must be divisible by 8"

        # Canonical normalised grid coordinates in [-1, 1]^2
        H, W = grid_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij",
        )
        self.register_buffer("grid_xy", torch.stack([yy, xx], dim=-1).reshape(-1, 2))

        # Sub-modules
        self.obs_encoder = ObservationEncoder(
            coord_dim, value_dim, embed_dim,
            fourier_features=fourier_features, scale=fourier_scale,
        )
        self.attn_layers = nn.ModuleList([
            CrossAttentionGridLayer(
                coord_dim, embed_dim, num_heads,
                fourier_features=fourier_features, dropout=dropout,
            )
            for _ in range(num_attn_layers)
        ])
        self.unet = UNetRefinement(embed_dim, out_channels, unet_base_ch)

    def forward(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        query_time: torch.Tensor | None = None,
        obs_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = obs_coords.size(0)
        H, W = self.grid_size

        # Default query time: mean of observation times
        if query_time is None:
            weights = (~obs_mask).float() if obs_mask is not None else torch.ones_like(obs_coords[..., 2])
            query_time = (obs_coords[..., 2] * weights).sum(dim=1) / weights.sum(dim=1)

        # Build grid query coordinates: attach query time to each spatial grid point
        grid_xy = self.grid_xy.unsqueeze(0).expand(B, -1, -1)        # (B, H*W, 2)
        t_grid  = query_time[:, None, None].expand(B, H * W, 1)       # (B, H*W, 1)
        grid_coords = torch.cat([grid_xy, t_grid], dim=-1)             # (B, H*W, 3)

        # 1. Encode sparse observations
        obs_emb = self.obs_encoder(obs_coords, obs_values)             # (B, N, D)

        # 2. Stack of cross-attention layers: grid queries attend to obs
        x = None
        for layer in self.attn_layers:
            x = layer(grid_coords, obs_emb, obs_mask)                  # (B, H*W, D)
            obs_emb = obs_emb  # observations are fixed; only grid feats update

        # 3. Reshape to spatial tensor and apply U-Net
        grid_feats = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        return self.unet(grid_feats)                                   # (B, C, H, W)


# ---------------------------------------------------------------------------
# 6. Loss Function
# ---------------------------------------------------------------------------

def masked_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Compute reconstruction loss only at known grid locations.

    Args:
        pred:        (B, C, H, W) — model predictions
        target:      (B, C, H, W) — ground-truth values
        target_mask: (B, H, W) bool — True where ground truth is available
        loss_type:   'mse' | 'mae' | 'huber'

    Returns: scalar loss averaged over valid locations
    """
    assert target_mask.any(), "No valid target locations in batch"

    # Select predictions and targets at valid locations
    mask = target_mask.unsqueeze(1).expand_as(pred)   # (B, C, H, W)
    p = pred[mask]
    t = target[mask]

    if loss_type == "mse":
        return F.mse_loss(p, t)
    elif loss_type == "mae":
        return F.l1_loss(p, t)
    elif loss_type == "huber":
        return F.huber_loss(p, t)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")


# ---------------------------------------------------------------------------
# 7. Example Dataset
# ---------------------------------------------------------------------------

class SyntheticGapFillingDataset(Dataset):
    """
    Generates synthetic training data by sampling a smooth ground-truth field
    and masking it to simulate sparse observations.

    Each sample:
        obs_coords:  (N, 3)   — (lat, lon, t) normalised to [-1, 1]
        obs_values:  (N, 1)   — observed values at those locations
        obs_mask:    (N,)     — False everywhere (no padding in this toy dataset)
        query_time:  scalar   — time to reconstruct
        target:      (1, H, W)— full ground-truth field at query_time
        target_mask: (H, W)   — True everywhere (full ground truth available)
    """
    def __init__(
        self,
        num_samples: int = 1000,
        grid_size: tuple[int, int] = (64, 64),
        num_obs: int = 100,
        num_time_steps: int = 20,
    ):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.num_obs = num_obs
        H, W = grid_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij",
        )
        self.register_grid = torch.stack([yy, xx], dim=-1)  # (H, W, 2)
        self.time_steps = torch.linspace(-1, 1, num_time_steps)

    def _ground_truth(self, t: float) -> torch.Tensor:
        """Generate a smooth synthetic field at time t. Replace with your data."""
        H, W = self.grid_size
        yy, xx = self.register_grid[..., 0], self.register_grid[..., 1]
        # Superposition of sine waves (toy); use your real data loader here
        field = (
            torch.sin(3.0 * yy + t * 2.0)
            * torch.cos(2.5 * xx - t * 1.5)
            + 0.3 * torch.sin(7.0 * (yy + xx))
        )
        return field.unsqueeze(0)  # (1, H, W)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        # Pick a random query time
        t_idx = torch.randint(len(self.time_steps), ()).item()
        query_time = self.time_steps[t_idx].item()

        # Full ground truth field
        target = self._ground_truth(query_time)           # (1, H, W)

        # Sample random sparse observation locations
        H, W = self.grid_size
        obs_y = torch.rand(self.num_obs) * 2 - 1
        obs_x = torch.rand(self.num_obs) * 2 - 1
        obs_t = query_time + (torch.rand(self.num_obs) - 0.5) * 0.2  # near query time

        obs_coords = torch.stack([obs_y, obs_x, obs_t], dim=-1)      # (N, 3)

        # Bilinear interpolation of the ground truth at observation locations
        grid_sample_pts = torch.stack([obs_x, obs_y], dim=-1).view(1, 1, -1, 2)
        obs_values = F.grid_sample(
            target.unsqueeze(0),
            grid_sample_pts,
            align_corners=True,
            mode="bilinear",
        ).squeeze().unsqueeze(-1)                                      # (N, 1)

        # Add noise to simulate measurement error
        obs_values = obs_values + 0.05 * torch.randn_like(obs_values)

        return {
            "obs_coords":  obs_coords,                                 # (N, 3)
            "obs_values":  obs_values,                                 # (N, 1)
            "obs_mask":    torch.zeros(self.num_obs, dtype=torch.bool),# (N,) all False
            "query_time":  torch.tensor(query_time),
            "target":      target,                                     # (1, H, W)
            "target_mask": torch.ones(H, W, dtype=torch.bool),        # (H, W) all True
        }


# ---------------------------------------------------------------------------
# 8. Training Loop
# ---------------------------------------------------------------------------

def train(
    model: SpatiotemporalGapFiller,
    dataset: Dataset,
    num_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    loss_type: str = "mse",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Minimal training loop — adapt for your own data pipeline."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:
            obs_coords  = batch["obs_coords"].to(device)   # (B, N, 3)
            obs_values  = batch["obs_values"].to(device)   # (B, N, 1)
            obs_mask    = batch["obs_mask"].to(device)     # (B, N)
            query_time  = batch["query_time"].to(device)   # (B,)
            target      = batch["target"].to(device)       # (B, 1, H, W)
            target_mask = batch["target_mask"].to(device)  # (B, H, W)

            pred = model(obs_coords, obs_values, query_time, obs_mask)
            loss = masked_reconstruction_loss(pred, target, target_mask, loss_type)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1:3d}/{num_epochs}  loss={avg:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return model


# ---------------------------------------------------------------------------
# 9. Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    GRID  = (64, 64)
    N_OBS = 150
    B     = 4

    model   = SpatiotemporalGapFiller(
        coord_dim=3, value_dim=1, out_channels=1,
        embed_dim=128, grid_size=GRID,
        num_attn_layers=2, num_heads=8,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Dummy forward pass ---
    obs_coords = torch.randn(B, N_OBS, 3).clamp(-1, 1)
    obs_values = torch.randn(B, N_OBS, 1)
    obs_mask   = torch.zeros(B, N_OBS, dtype=torch.bool)
    query_time = torch.zeros(B)

    with torch.no_grad():
        pred = model(obs_coords, obs_values, query_time, obs_mask)

    print(f"Output shape: {pred.shape}")   # expect (4, 1, 64, 64)
    assert pred.shape == (B, 1, *GRID), "Shape mismatch!"

    # --- Synthetic training run ---
    dataset = SyntheticGapFillingDataset(num_samples=200, grid_size=GRID, num_obs=N_OBS)
    trained = train(model, dataset, num_epochs=5, batch_size=8)
    print("Training smoke test passed.")

    # ---------------------------------------------------------------------------
    # 10. Inference + visualisation
    # ---------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    trained.eval()

    # Pull one sample from the dataset for a clean ground-truth comparison
    sample      = dataset[0]
    obs_c       = sample["obs_coords"].unsqueeze(0)   # (1, N, 3)
    obs_v       = sample["obs_values"].unsqueeze(0)   # (1, N, 1)
    obs_m       = sample["obs_mask"].unsqueeze(0)     # (1, N)
    qt          = sample["query_time"].unsqueeze(0)   # (1,)
    ground_truth = sample["target"].squeeze().numpy() # (H, W)

    device = next(trained.parameters()).device
    obs_c  = obs_c.to(device)
    obs_v  = obs_v.to(device)
    obs_m  = obs_m.to(device)
    qt     = qt.to(device)

    with torch.no_grad():
        prediction = trained(obs_c, obs_v, qt, obs_m)

    pred_grid = prediction.squeeze().cpu().numpy()    # (H, W)

    # Scatter coords are in [-1, 1]; convert to pixel space for overlay
    H, W = GRID
    obs_xy = obs_c.squeeze().cpu().numpy()             # (N, 3): [lat, lon, t]
    # lat → row (y axis), lon → col (x axis); map [-1,1] → [0, W/H]
    scatter_col = (obs_xy[:, 1] + 1) / 2 * (W - 1)   # lon → x
    scatter_row = (obs_xy[:, 0] + 1) / 2 * (H - 1)   # lat → y
    obs_vals    = obs_v.squeeze().cpu().numpy()        # (N,)

    # Shared colour limits across all three panels
    vmin = min(ground_truth.min(), pred_grid.min())
    vmax = max(ground_truth.max(), pred_grid.max())

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Spatiotemporal gap-filling — inference visualisation", fontsize=13)
    gs  = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05],
                            wspace=0.35, left=0.05, right=0.93)

    ax_gt   = fig.add_subplot(gs[0])
    ax_pred = fig.add_subplot(gs[1])
    ax_obs  = fig.add_subplot(gs[2])
    ax_cb   = fig.add_subplot(gs[3])

    cmap = "RdYlBu_r"

    # --- Panel 1: ground truth ---
    im = ax_gt.imshow(ground_truth, origin="lower", cmap=cmap,
                      vmin=vmin, vmax=vmax, aspect="auto")
    ax_gt.set_title("Ground truth", fontsize=11)
    ax_gt.set_xlabel("lon (grid col)")
    ax_gt.set_ylabel("lat (grid row)")

    # --- Panel 2: model prediction ---
    ax_pred.imshow(pred_grid, origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="auto")
    ax_pred.set_title("Model prediction", fontsize=11)
    ax_pred.set_xlabel("lon (grid col)")
    ax_pred.set_yticks([])

    # --- Panel 3: prediction + sparse observations overlaid ---
    ax_obs.imshow(pred_grid, origin="lower", cmap=cmap,
                  vmin=vmin, vmax=vmax, aspect="auto")
    sc = ax_obs.scatter(
        scatter_col, scatter_row,
        c=obs_vals, cmap=cmap, vmin=vmin, vmax=vmax,
        s=18, edgecolors="white", linewidths=0.5, zorder=3,
    )
    ax_obs.set_title("Prediction + sparse obs", fontsize=11)
    ax_obs.set_xlabel("lon (grid col)")
    ax_obs.set_yticks([])

    # Shared colorbar
    fig.colorbar(im, cax=ax_cb, label="value")

    plt.savefig("gap_filling_inference.png", dpi=150, bbox_inches="tight")
    print("Saved gap_filling_inference.png")
    plt.show()