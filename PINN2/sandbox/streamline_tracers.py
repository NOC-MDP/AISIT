"""
streamlines_tracer.py
---------------------
Convert a matplotlib streamplot result into a 2-D tracer mass-fraction field.

River mouths emit tracer at concentration C0; the tracer decays exponentially
along each streamline with e-folding length `decay_length`.

Algorithm
---------
1. Extract all polyline paths from the StreamplotSet (.lines.get_paths()).
2. For every path, find the vertex closest to any river mouth.
3. Walk *downstream* from that vertex, accumulating arc length s.
4. Assign  C(s) = C0 * exp(-s / decay_length)  to each vertex.
5. Scatter all labelled (x, y, C) points and interpolate onto the output grid.

Usage
-----
Run as a script for a self-contained demo, or import
`streamlines_to_mass_fraction` for use in your own workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def streamlines_to_mass_fraction(
    streamplot_result,
    river_mouths,
    grid_x,
    grid_y,
    *,
    decay_length: float = 1.0,
    initial_concentration: float = 1.0,
    mouth_snap_radius: float | None = None,
    max_interp_dist: float | None = None, 
    interpolation_method: str = "linear",
    combine: str = "max",
):
    """
    Build a tracer mass-fraction field from a matplotlib streamplot result.

    Parameters
    ----------
    streamplot_result : matplotlib.streamplot.StreamplotSet
        The object returned by ``ax.streamplot(...)``.
    river_mouths : array-like, shape (N, 2)
        (x, y) coordinates of each river mouth.
    grid_x, grid_y : 2-D ndarrays
        Output grid produced by ``np.meshgrid``.  Must share a coordinate
        system with the streamplot.
    decay_length : float
        Exponential e-folding length in the same units as the coordinate
        system.  Larger values = slower decay = tracer reaches further.
    initial_concentration : float
        Tracer concentration emitted at the river mouth (default 1.0 = 100%).
    mouth_snap_radius : float or None
        Maximum distance from a path vertex to a river mouth for that vertex
        to count as "at the mouth".  Defaults to 5% of the domain diagonal --
        a robust automatic choice that works across grid scales.
        Increase it if river mouths sit between streamlines and no paths match.
    interpolation_method : {'linear', 'nearest', 'cubic'}
        Passed to ``scipy.interpolate.griddata``.
    combine : {'max', 'sum'}
        How to handle overlapping contributions from multiple river mouths or
        multiple streamline paths that cover the same grid cell.
        'max'  -- keep the highest concentration (default; avoids super-saturation).
        'sum'  -- add contributions (useful when mouths are far apart).

    Returns
    -------
    concentration : 2-D ndarray, same shape as ``grid_x``
        Tracer mass-fraction field on the output grid, values in
        [0, initial_concentration].  Grid cells that no streamline reaches
        are set to 0.
    """
    river_mouths = np.asarray(river_mouths, dtype=float)
    if river_mouths.ndim == 1:
        river_mouths = river_mouths[np.newaxis, :]  # single mouth

    paths = streamplot_result.lines.get_paths()
    if not paths:
        raise ValueError(
            "No paths found in streamplot_result.lines. "
            "Check that the streamplot was drawn successfully."
        )

    # Auto snap radius: 5% of domain diagonal
    if mouth_snap_radius is None:
        all_verts = np.vstack([p.vertices for p in paths])
        domain_diag = np.linalg.norm(
            all_verts.max(axis=0) - all_verts.min(axis=0)
        )
        mouth_snap_radius = 0.05 * domain_diag

    all_xy = []
    all_c  = []

    for path in paths:
        verts = path.vertices          # shape (N, 2)
        n = len(verts)
        if n < 2:
            continue

        # Cumulative arc length along this path
        diffs   = np.diff(verts, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        arc     = np.concatenate([[0.0], np.cumsum(seg_len)])

        # Distance from every vertex to every river mouth  -> shape (N, M)
        dists = np.linalg.norm(
            verts[:, np.newaxis, :] - river_mouths[np.newaxis, :, :], axis=2
        )
        min_dist_per_vertex = dists.min(axis=1)
        best_vertex_idx     = int(min_dist_per_vertex.argmin())
        min_dist            = min_dist_per_vertex[best_vertex_idx]

        if min_dist > mouth_snap_radius:
            continue   # streamline doesn't pass near any river mouth

        mouth_arc    = arc[best_vertex_idx]
        s_from_mouth = arc - mouth_arc   # negative = upstream of mouth

        c = np.where(
            s_from_mouth >= 0,
            initial_concentration * np.exp(-s_from_mouth / decay_length),
            0.0,
        )

        all_xy.append(verts)
        all_c.append(c)

    if not all_xy:
        all_verts = np.vstack([p.vertices for p in paths])
        dists_all = np.linalg.norm(
            all_verts[:, np.newaxis, :] - river_mouths[np.newaxis, :, :], axis=2
        )
        nearest_miss = dists_all.min()
        print(
            f"Warning: no streamline paths matched any river mouth.\n"
            f"  Nearest path vertex is {nearest_miss:.4f} units from a mouth;\n"
            f"  mouth_snap_radius is {mouth_snap_radius:.4f}.\n"
            f"  Try increasing mouth_snap_radius to > {nearest_miss:.3f}."
        )
        return np.zeros_like(grid_x)

    all_xy_arr = np.vstack(all_xy)
    all_c_arr  = np.concatenate(all_c)

    concentration = griddata(
        all_xy_arr, all_c_arr,
        (grid_x, grid_y),
        method=interpolation_method,
        fill_value=0.0,
    )
    concentration = np.where(np.isnan(concentration), 0.0, concentration)
    if max_interp_dist is not None:
        tree        = cKDTree(all_xy_arr)
        grid_pts    = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        nn_dist, _  = tree.query(grid_pts, k=1, workers=-1)
        nn_dist     = nn_dist.reshape(grid_x.shape)
        concentration[nn_dist > max_interp_dist] = 0.0
    concentration = np.clip(concentration, 0.0, initial_concentration)
    return concentration


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _make_demo_velocity_field(nx=80, ny=60):
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 6,  ny)
    X, Y = np.meshgrid(x, y)

    # Coastal jet near x~1, flowing southward
    V_jet  = -0.7 * np.exp(-((X - 1.0) ** 2) / 0.5)
    # Offshore gyre
    U_gyre =  0.5 * np.sin(np.pi * Y / 6) * np.cos(np.pi * X / 10)
    V_gyre = -0.5 * np.cos(np.pi * Y / 6) * np.sin(np.pi * X / 10)
    # Mean eastward shelf transport
    U_mean =  0.35 * np.exp(-((Y - 3.0) ** 2) / 4.0)
    # Mild turbulence
    rng = np.random.default_rng(42)
    U_noise = 0.04 * rng.standard_normal(X.shape)
    V_noise = 0.04 * rng.standard_normal(Y.shape)

    U = U_mean + U_gyre + U_noise
    V = V_jet  + V_gyre + V_noise
    return x, y, X, Y, U, V


def run_demo():
    x, y, X, Y, U, V = _make_demo_velocity_field()

    # River mouths on the western coast (x ~ 0)
    river_mouths = np.array([
        [0.0, 1.0],
        [0.0, 3.0],
        [0.0, 5.0],
    ])

    decay_configs = [
        (0.8, "Short decay  (λ = 0.8)"),
        (2.5, "Medium decay  (λ = 2.5)"),
        (7.0, "Long decay  (λ = 7.0)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle(
        "River-mouth tracer: streamlines → mass-fraction field",
        fontsize=13, y=1.01,
    )

    for ax, (decay_length, label) in zip(axes, decay_configs):
        sp = ax.streamplot(
            x, y, U, V,
            density=1.6,
            linewidth=0.7,
            color="0.60",
            arrowsize=0.8,
            zorder=2,
        )

        concentration = streamlines_to_mass_fraction(
            sp,
            river_mouths,
            X, Y,
            decay_length=decay_length,
            initial_concentration=1.0,
            combine="max",
        )

        masked = np.ma.masked_where(concentration < 0.01, concentration)

        pcm = ax.pcolormesh(
            X, Y, masked,
            cmap="YlOrRd",
            norm=Normalize(vmin=0, vmax=1),
            shading="gouraud",
            zorder=0,
            alpha=0.88,
        )

        ax.scatter(
            river_mouths[:, 0], river_mouths[:, 1],
            s=70, c="royalblue", zorder=5,
            edgecolors="white", linewidths=0.6,
            label="River mouth",
        )

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_aspect("equal")
        cb = fig.colorbar(pcm, ax=ax, label="Mass fraction  C / C0", pad=0.02)
        cb.ax.tick_params(labelsize=9)

    axes[0].legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    out = "tracer_field_demo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    run_demo()