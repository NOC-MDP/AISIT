"""Plotting classes"""

import plotly.express as px
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


class ArcticPlotter:
    """Class to create an Arctic projection plot"""

    def __init__(self, lat, lon, figsize=(8, 8)):
        self.figsize = figsize
        self.proj = ccrs.NorthPolarStereo()
        self.lat = lat
        self.lon = lon
        self.title = "Arctic Sampling Locations from GLODAP database"
        self.cbar_title = "year"

    def scatter_plot(self, ax, cbar_data, vmin=None, vmax=None, **kwargs):
        """Create a scatter plot with default kwargs"""
        default_kwargs = {
            "s": 40,
            "cmap": "viridis",
            "edgecolor": "black",
            "linewidth": 0.4,
        }
        kwargs = {**default_kwargs, **kwargs}

        sc = ax.scatter(
            self.lon,
            self.lat,
            c=cbar_data,
            s=kwargs["s"],
            cmap=kwargs["cmap"],
            transform=ccrs.PlateCarree(),
            edgecolor=kwargs["edgecolor"],
            linewidth=kwargs["linewidth"],
            vmin=vmin,
            vmax=vmax,
        )

        return sc

    def plot(self, cbar_data, vmin=None, vmax=None):
        """Create the plot"""
        # --- Create figure and axis ---
        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes(projection=self.proj)

        # --- Base map features ---
        ax.add_feature(cfeature.LAND, color="lightgray")
        ax.add_feature(cfeature.OCEAN, color="lightblue")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.gridlines(
            draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
        )

        # --- Arctic extent ---
        ax.set_extent([-180, 180, 55, 90], ccrs.PlateCarree())

        # --- Plot the sample points ---
        sc = self.scatter_plot(ax, cbar_data, vmin, vmax)

        # --- Colorbar ---
        cb = plt.colorbar(
            sc,
            ax=ax,
            orientation="vertical",
            shrink=0.6,
            pad=0.05,
        )
        cb.set_label(self.cbar_title)

        # --- Title ---
        plt.title(self.title, fontsize=12)

        return fig, ax


class OxygenIsotopePlots:
    """Class to generate plots of d18O vs other variables"""

    def __init__(self, x, y, ref):
        self.proj = ccrs.NorthPolarStereo()
        self.x = x
        self.y = y
        self.ref = ref
        self.title = "δ¹⁸O vs. Salinity from BAS Prealpha database"
        self.x_title = "salinity (PSU)"
        self.y_title = "δ¹⁸O (‰)"
        self.cbar_title = "year"

    def plot(self, df):
        """Scatter plot: d18O vs other variables"""
        fig_scatter = px.scatter(
            df,
            x=self.x,
            y=self.y,
            color=df["datetime"].dt.year,
            hover_data=[self.x, self.y, self.ref],
            title=self.title,
            # trendline="ols"  # optional: adds regression line
        )
        fig_scatter.update_layout(
            xaxis_title=self.x_title,
            yaxis_title=self.y_title,
            coloraxis_colorbar_title_text=self.cbar_title,
        )
        return fig_scatter
