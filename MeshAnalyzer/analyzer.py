import logging
import re
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import vedo

from .io import load_surface_data, load_curvature_data, validate_file_paths, load_auxiliary_data
from .utils import calculate_mesh_quality_metrics
from .datatypes import AnalysisResults, MeshStatistics, CurvatureStatistics, ProcessingMetadata, AuxiliaryMeshData
from ._validation import validate_pixel_sizes

logger = logging.getLogger(__name__)

class MeshAnalyzer:
    """
    Analyze u-shape3D mesh data including surface geometry and curvature.

    Example:
        >>> analyzer = MeshAnalyzer('surface.mat', 'curvature.mat')
        >>> analyzer.load_data()
        >>> stats = analyzer.calculate_statistics()
    """
    VERSION = "1.0.0"
    SUPPORTED_FORMATS = ['.mat', '.h5']
    DEFAULT_PIXEL_SIZE_XY = 0.103 #um
    DEFAULT_PIXEL_SIZE_Z = 0.2167 #um

    def __init__(self, surface_path:str, curvature_path: str,
                 pixel_size_xy: float = None, pixel_size_z: float = None,
                 metadata: Optional[ProcessingMetadata] = None,
                 load_auxiliary: bool = False):
        """
        Initialize the MeshAnalyzer.

        Parameters:
            surface_path: Path to surface .mat file
            curvature_path: Path to curvature .mat file
            pixel_size_xy: XY pixel size in micrometers (default: 0.1)
            pixel_size_z: Z pixel size in micrometers (default: 0.2)
            metadata: Optional ProcessingMetadata (provenance)
            load_auxiliary: If True, load auxiliary data (normals, Gaussian curv, etc.)
        """
        import warnings
        warnings.warn(
            "MeshAnalyzer is deprecated. Use load_cell() or "
            "TimeSeriesManager.from_cell_directory() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        self.surface_path = Path(surface_path)
        self.curvature_path = Path(curvature_path)
        self._validate_inputs()

        self.pixel_size_xy = pixel_size_xy or self.DEFAULT_PIXEL_SIZE_XY
        self.pixel_size_z = pixel_size_z or self.DEFAULT_PIXEL_SIZE_Z

        validate_pixel_sizes(self.pixel_size_xy, self.pixel_size_z)

        self.metadata = metadata
        self.load_auxiliary = load_auxiliary
        self.auxiliary_data: Optional[AuxiliaryMeshData] = None

        self.vertices: Optional[np.ndarray] = None
        self.faces: Optional[np.ndarray] = None
        self.mesh: Optional[vedo.Mesh] = None
        self.curvature: Optional[np.ndarray] = None

        self._processed = False
        self._mesh_was_corrected = False
        self.results = AnalysisResults()

    def _validate_inputs(self):
        """Validate input files exist and have correct format."""
        validate_file_paths(self.surface_path, self.curvature_path, self.SUPPORTED_FORMATS)

    @staticmethod
    def _extract_channel_time(filename: str) -> Tuple[int, int]:
        """Extract channel and time index from filename."""
        match = re.search(r'surface_(\d+)_(\d+)\.mat', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 1, 1

    def load_data(self, verbose: bool = False) -> None:
        """Load mesh and curvature data from files."""
        try:
            logger.info(f"Loading surface from {self.surface_path}")
            self.vertices, self.faces, self.mesh = load_surface_data(self.surface_path)

            if self.mesh.volume() < 0:
                logger.warning(
                    f"Mesh has negative volume ({self.mesh.volume():.2f}), indicating "
                    "inverted face normals. Automatically correcting orientation. "
                    "Original mesh preserved via clone()."
                )
                self.mesh = self.mesh.clone().reverse()
                self._mesh_was_corrected = True
            else:
                self._mesh_was_corrected = False

            logger.info(f"Loading curvature from {self.curvature_path}")
            self.curvature = load_curvature_data(self.curvature_path, len(self.faces))

            if self.load_auxiliary:
                mesh_dir = self.surface_path.parent
                channel, time_index = self._extract_channel_time(self.surface_path.name)
                self.auxiliary_data = load_auxiliary_data(mesh_dir, channel, time_index)

            self._processed = True
            logger.info(f"Loaded {len(self.vertices)} vertices, {len(self.faces)} faces")

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def calculate_statistics(self, force_recalculate: bool = False) -> AnalysisResults:
        """Calculate comprehensive mesh and curvature statistics."""
        if not self._processed:
            raise RuntimeError("Must load data first. Call load_data()")

        if self.results.is_complete() and not force_recalculate:
            return self.results

        self.results.mesh_stats = MeshStatistics(
            n_vertices=len(self.vertices),
            n_faces=len(self.faces),
            n_edges=len(self.mesh.edges),
            volume_pixels3=float(self.mesh.volume()),
            # Mesh is already isotropic (Z resampled in MATLAB)
            volume_um3=float(self.mesh.volume() * self.pixel_size_xy**3),
            surface_area_pixels2=float(self.mesh.area()),
            surface_area_um2=float(self.mesh.area() * self.pixel_size_xy**2),
            is_watertight=self.mesh.is_closed(),
            euler_number=self.mesh.euler_characteristic()
        )

        self.results.curvature_stats = CurvatureStatistics.from_array(self.curvature)
        self.results.quality_metrics = calculate_mesh_quality_metrics(self.mesh)

        return self.results
    
    def calculate_statistics_dict(self, force_recalculate: bool = False) -> Dict:
        """Legacy method that returns statistics as dictionary."""
        results = self.calculate_statistics(force_recalculate)
        return {
            'mesh': results.mesh_stats.to_dict() if results.mesh_stats else {},
            'curvature': results.curvature_stats.__dict__ if results.curvature_stats else {},
            'quality': results.quality_metrics.__dict__ if results.quality_metrics else {}
        }

    @property
    def is_loaded(self) -> bool:
        """Check if data has been loaded."""
        return self._processed

    @property
    def was_mesh_corrected(self) -> bool:
        """
        Check if mesh orientation was automatically corrected during loading.

        Returns:
            True if mesh had inverted normals and was corrected, False otherwise
        """
        return self._mesh_was_corrected

    @property
    def face_centers(self) -> np.ndarray:
        """
        Get face centers (cached, vectorized calculation).

        Face centers are computed as the average of the three vertex
        coordinates for each triangular face. This property uses vectorized
        numpy operations for efficient computation and caches the result.

        Returns:
            Nx3 array of face center coordinates (one per face)

        Raises:
            RuntimeError: If data hasn't been loaded yet

        Note:
            Result is cached after first computation for performance.
        """
        if not self._processed:
            raise RuntimeError(
                "Must load data first. Call load_data() before accessing face_centers."
            )

        if not hasattr(self, '_face_centers_cache'):
            # Vectorized calculation: 10-100x faster than list comprehension
            # Old: [np.mean(self.vertices[face], axis=0) for face in self.faces]
            # New: self.vertices[self.faces].mean(axis=1)
            self._face_centers_cache = self.vertices[self.faces].mean(axis=1)

        return self._face_centers_cache

    @property
    def physical_dimensions(self) -> Dict[str, float]:
        """Get physical dimensions in micrometers."""
        if not self._processed:
            return {}

        bounds = self.mesh.bounds()
        return {
            'x_um': (bounds[1] - bounds[0]) * self.pixel_size_xy,
            'y_um': (bounds[3] - bounds[2]) * self.pixel_size_xy,
            'z_um': (bounds[5] - bounds[4]) * self.pixel_size_z
        }

    @property
    def processing_metadata(self) -> Optional['ProcessingMetadata']:
        """Access processing metadata for this mesh."""
        return self.metadata

    @property
    def mesh_parameters(self) -> Optional['MeshParameters']:
        """Access mesh generation parameters."""
        return self.metadata.mesh_parameters if self.metadata else None

    @property
    def face_normals(self) -> Optional[np.ndarray]:
        """Convenience property for face normals."""
        return self.auxiliary_data.face_normals if self.auxiliary_data else None

    @property
    def gaussian_curvature(self) -> Optional[np.ndarray]:
        """Convenience property for Gaussian curvature."""
        return self.auxiliary_data.gaussian_curvature if self.auxiliary_data else None

    def __str__(self) -> str:
        if not self._processed:
            return f"MeshAnalyzer(not loaded)"
        return f"MeshAnalyzer({len(self.vertices)} vertices, {len(self.faces)} faces)"

    def __repr__(self) -> str:
        return f"MeshAnalyzer(surface='{self.surface_path.name}', curvature='{self.curvature_path.name}')"

    def __enter__(self):
        self.load_data()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass