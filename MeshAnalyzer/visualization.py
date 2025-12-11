"""Visualization functions for mesh analysis."""
from typing import Optional, Union, Tuple
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt

from .datatypes import MeshFrame

NATURE_COLORS = {
    'blue': '#2E86AB',
    'red': '#E63946',
    'gray': '#6C757D',
    'light_gray': '#ADB5BD',
    'dark_gray': '#343A40',
    'green': '#028A0F',
    'orange': '#F77F00'
}


@contextmanager
def publication_style():
    """
    Context manager for publication-quality matplotlib style.

    Applies a clean, professional style suitable for scientific publications
    without polluting global matplotlib state. Style is automatically restored
    when exiting the context.

    Example:
        >>> with publication_style():
        ...     fig, ax = plt.subplots()
        ...     ax.plot(data)

    Style features:
        - Arial font
        - Minimal borders (no top/right spines)
        - 300 DPI for high-resolution output
        - Thin lines for clean appearance
    """
    with plt.rc_context({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.5,
        'figure.dpi': 300
    }):
        yield



def plot_curvature_distribution(curvature: np.ndarray,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot curvature distribution with linear and log scale.

    Uses publication-quality style automatically via context manager.

    Args:
        curvature: Array of curvature values
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    with publication_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.hist(curvature, bins=100, alpha=0.7)
        ax1.axvline(0, color=NATURE_COLORS['red'], linestyle='--', label='Zero')
        ax1.set_xlabel('Curvature (1/pixels)')
        ax1.set_ylabel('Count')
        ax1.set_title('Curvature Distribution')
        ax1.legend()

        non_zero_curv = curvature[curvature != 0]
        ax2.hist(non_zero_curv, bins=100, alpha=0.7)
        ax2.set_yscale('log')
        ax2.set_xlabel('Curvature (1/pixels)')
        ax2.set_ylabel('Count (log scale)')
        ax2.set_title('Curvature Distribution (Log Scale)')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def basic_spatial_plot(mesh: Union['vedo.Mesh', MeshFrame],
                       curvature: Optional[np.ndarray] = None,
                       save_path: Optional[str] = None,
                       title: str = "Spatial Curvature Distribution",
                       face_centers: Optional[np.ndarray] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a spatial visualization showing where high/low curvatures are located.

    Args:
        mesh: vedo.Mesh object OR MeshFrame object (preferred)
        curvature: Array of curvature values (one per face).
                  If mesh is MeshFrame, this is ignored (uses mesh.curvature)
        save_path: Optional path to save figure
        title: Plot title
        face_centers: Optional pre-computed face centers (Nx3 array).
                     If mesh is MeshFrame, uses cached face_centers property.
                     Otherwise computed from mesh geometry.

    Returns:
        matplotlib Figure object

    Example:
        # New API (no redundant computation)
        frame = load_mesh_frame(surface_path, curvature_path)
        fig = basic_spatial_plot(frame)

        # Legacy API (still supported)
        mesh = vedo.Mesh([vertices, faces])
        fig = basic_spatial_plot(mesh, curvature_array)
    """
    # Handle MeshFrame input (new API)
    if isinstance(mesh, MeshFrame):
        face_centers = mesh.face_centers  # Use cached property
        curvature = mesh.curvature
        vedo_mesh = mesh.mesh
    # Legacy API
    else:
        vedo_mesh = mesh
        if curvature is None:
            raise ValueError("curvature must be provided when using legacy API")
        if face_centers is None:
            vertices = vedo_mesh.vertices
            faces = vedo_mesh.cells
            face_centers = vertices[faces].mean(axis=1)

    with publication_style():
        fig, ax = plt.subplots(figsize=(10, 8))

        vmax = np.percentile(np.abs(curvature), 95)
        vmin = -vmax

        scatter = ax.scatter(face_centers[:, 0], face_centers[:, 1],
                            c=curvature, s=0.5, cmap='RdBu',
                            vmin=vmin, vmax=vmax)

        ax.set_xlabel('X position (pixels)')
        ax.set_ylabel('Y position (pixels)')
        ax.set_title(title)
        ax.set_aspect('equal')

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Mean Curvature (1/pixels)')

        stats_text = f'Mean: {np.mean(curvature):.3f}\nStd: {np.std(curvature):.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax