"""Mesh analysis package for u-shape3D data processing."""
import logging

from .analyzer import MeshAnalyzer
from .visualization import plot_curvature_distribution, basic_spatial_plot
from .io import load_surface_data, load_curvature_data
from .utils import convert_pixels_to_um
from .datatypes import (
    AnalysisResults,
    MeshStatistics,
    CurvatureStatistics,
    QualityMetrics
)
from .timeseries import (
    TimeSeriesManager,
    TimeFrameInfo,
    TimeSeriesData
)

__version__ = "1.1.0"
__all__ = [
    'MeshAnalyzer',
    'AnalysisResults',
    'MeshStatistics',
    'CurvatureStatistics',
    'QualityMetrics',
    'TimeSeriesManager',
    'TimeFrameInfo',
    'TimeSeriesData',
    'setup_logging'
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

