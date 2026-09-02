import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -------------------------------------------------------------------------
# 1. Configuration & Customization
# -------------------------------------------------------------------------
WATER_MASSES = ["ATL", "PAC", "NAM", "EUR", "SIM"]
FULL_NAMES = {
    "ATL": "Atlantic Water",
    "PAC": "Pacific Water",
    "NAM": "North American River Runoff",
    "EUR": "Eurasian River Runoff",
    "SIM": "Sea Ice Meltwater",
}
YEAR = "2016"
MONTH = "09"
DEPTH = 10
FILE_PATH = f"/gws/ssde/j25a/nemo/vol4/thopri/MASSBAL/watermass_fractions_depth_{DEPTH}_{YEAR}_{MONTH}.nc"
OUTPUT_PLOT = "arctic_water_mass_fractions_and_uncertainty.png"


# -------------------------------------------------------------------------
# 2. Plotting Function
# -------------------------------------------------------------------------
def plot_water_mass_fractions_and_std(ds_path, min_lat=65):
    """Plots a 5x2 panel figure comparing Arctic water mass mean fractions

    and Monte Carlo standard deviations using NorthPolarStereo projection.
    """
    ds = xr.open_dataset(ds_path)

    # Convert coordinates if longitudes run from 0 to 360 instead of -180 to 180
    if (ds["longitude"].max() > 180).item():
        ds = ds.assign_coords(
            longitude=(((ds["longitude"] + 180) % 360) - 180)
        ).sortby("longitude")

    # Set up 5x2 Subplot Grid (5 rows for water masses, 2 cols for Mean vs Std Dev)
    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(12, 22),
        subplot_kw={"projection": ccrs.NorthPolarStereo()},
    )

    # Define circular boundary for Polar Stereo Projection
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    # Colormaps and color limits
    cmap_mean = "Blues_r"
    vmin_mean, vmax_mean = 0, 100

    cmap_std = "magma"  # 'magma', 'inferno', or 'viridis' work well for uncertainties
    vmin_std, vmax_std = 0, 20  # Adjust max % std dev based on your data range

    pcm_mean_handle = None
    pcm_std_handle = None

    for i, wm in enumerate(WATER_MASSES):
        ax_mean = axes[i, 0]
        ax_std = axes[i, 1]

        # Apply spatial formatting to both subplots in the row
        for ax in (ax_mean, ax_std):
            ax.set_boundary(circle, transform=ax.transAxes)
            ax.set_extent([-180, 180, min_lat, 90], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, zorder=2, edgecolor="black", facecolor="lightgray")
            ax.add_feature(cfeature.COASTLINE, zorder=3, linewidth=0.7)
            ax.gridlines(
                draw_labels=False,
                linewidth=0.5,
                color="gray",
                alpha=0.6,
                linestyle="--",
            )

        # --- Column 1: Mean Fraction ---
        data_mean = ds[f"{wm}_fraction_mean"] * 100.0
        pcm_mean = ax_mean.pcolormesh(
            ds["longitude"],
            ds["latitude"],
            data_mean,
            transform=ccrs.PlateCarree(),
            cmap=cmap_mean,
            vmin=vmin_mean,
            vmax=vmax_mean,
            shading="auto",
        )
        if pcm_mean_handle is None:
            pcm_mean_handle = pcm_mean

        letter_mean = chr(97 + i * 2)  # a, c, e, g, i
        ax_mean.set_title(
            f"({letter_mean}) {FULL_NAMES[wm]} ({wm})\nMean Fraction",
            fontsize=11,
            fontweight="bold",
            pad=6,
        )

        # --- Column 2: Monte Carlo Std Dev ---
        data_std = ds[f"{wm}_fraction_std"] * 100.0  # Assuming standard dev variable name
        pcm_std = ax_std.pcolormesh(
            ds["longitude"],
            ds["latitude"],
            data_std,
            transform=ccrs.PlateCarree(),
            cmap=cmap_std,
            vmin=vmin_std,
            vmax=vmax_std,
            shading="auto",
        )
        if pcm_std_handle is None:
            pcm_std_handle = pcm_std

        letter_std = chr(97 + i * 2 + 1)  # b, d, f, h, j
        ax_std.set_title(
            f"({letter_std}) {FULL_NAMES[wm]} ({wm})\nMonte Carlo Std Dev (1σ)",
            fontsize=11,
            fontweight="bold",
            pad=6,
        )

    # Adjust layout to make room for colorbars at the bottom
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08, top=0.95)

    # --- Shared Colorbar 1: Mean Fraction ---
    cbar_ax1 = fig.add_axes([0.15, 0.035, 0.32, 0.015])  # [left, bottom, width, height]
    cbar1 = fig.colorbar(
        pcm_mean_handle,
        cax=cbar_ax1,
        orientation="horizontal",
        extend="both",
    )
    cbar1.set_label("Water Mass Fraction (%)", fontsize=11, fontweight="bold")
    cbar1.ax.tick_params(labelsize=9)

    # --- Shared Colorbar 2: Monte Carlo Std Dev ---
    cbar_ax2 = fig.add_axes([0.55, 0.035, 0.32, 0.015])  # [left, bottom, width, height]
    cbar2 = fig.colorbar(
        pcm_std_handle,
        cax=cbar_ax2,
        orientation="horizontal",
        extend="max",
    )
    cbar2.set_label("Uncertainty / Std Dev (%)", fontsize=11, fontweight="bold")
    cbar2.ax.tick_params(labelsize=9)

    plt.suptitle(
        f"Arctic Ocean Water Mass Fractions & Monte Carlo Uncertainties at {DEPTH} m",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to {OUTPUT_PLOT}")
    plt.show()


if __name__ == "__main__":
    plot_water_mass_fractions_and_std(FILE_PATH, min_lat=65)
