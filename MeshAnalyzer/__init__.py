"""Mesh analysis package for u-shape3D data processing."""
import logging

from .analyzer import MeshAnalyzer
from .visualization import plot_curvature_distribution, basic_spatial_plot, publication_style
from .io import load_surface_data, load_curvature_data, load_mesh_frame
from .utils import convert_pixels_to_um
from .datatypes import (
    AnalysisResults,
    MeshStatistics,
    CurvatureStatistics,
    QualityMetrics,
    MeshFrame,
    ProcessingMetadata,
    DEFAULT_MESH_PARAMETERS,
    DEFAULT_PIXEL_SIZE_XY_UM,
    DEFAULT_PIXEL_SIZE_Z_UM
)
from .timeseries import (
    TimeSeriesManager,
    TimeFrameInfo
)


def load_cell(cell_dir, channel: int = 1, **kwargs):
    """
    Load mesh data from cell directory (recommended entry point).

    Automatically extracts metadata and discovers all frames.
    Returns TimeSeriesManager with dict-like interface.

    Parameters:
        cell_dir: Path to cell directory
        channel: Channel number (default: 1)
        **kwargs: Additional arguments passed to TimeSeriesManager

    Returns:
        TimeSeriesManager with discovered frames

    Example:
        >>> manager = load_cell('/path/to/cell_directory')
        >>> frame = manager[1]  # Access frame by time index
        >>> for time_idx, frame in manager:
        >>>     print(f"Frame {time_idx}: {frame.n_vertices} vertices")
    """
    from pathlib import Path
    manager = TimeSeriesManager.from_cell_directory(Path(cell_dir), channel, **kwargs)
    manager.discover_frames()
    return manager


__version__ = "1.2.0"
__all__ = [
    # Recommended entry point
    'load_cell',
    # Legacy API (maintained for backwards compatibility)
    'MeshAnalyzer',
    # New simplified API (recommended)
    'MeshFrame',
    'load_mesh_frame',
    # Data structures
    'AnalysisResults',
    'MeshStatistics',
    'CurvatureStatistics',
    'QualityMetrics',
    'ProcessingMetadata',
    # Time-series
    'TimeSeriesManager',
    'TimeFrameInfo',
    # Visualization
    'plot_curvature_distribution',
    'basic_spatial_plot',
    'publication_style',
    # Utilities
    'convert_pixels_to_um',
    'setup_logging',
    # Constants
    'DEFAULT_MESH_PARAMETERS',
    'DEFAULT_PIXEL_SIZE_XY_UM',
    'DEFAULT_PIXEL_SIZE_Z_UM'
]


def setup_logging(level: str = 'INFO', format: str = None) -> None:
    """
    Configure logging for MeshAnalyzer.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        format: Custom log format string (optional)

    Example:
        >>> import MeshAnalyzer
        >>> MeshAnalyzer.setup_logging('DEBUG')
        >>> analyzer = MeshAnalyzer.MeshAnalyzer('surface.mat', 'curvature.mat')
        >>> analyzer.load_data()
    """
    if format is None:
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

