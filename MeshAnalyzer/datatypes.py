"""
Data structures for mesh analysis results.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List
from datetime import datetime
import numpy as np


# Default physical size constants (micrometers)
DEFAULT_PIXEL_SIZE_XY_UM = 0.103
DEFAULT_PIXEL_SIZE_Z_UM = 0.2167

# Default physical size constants (nanometers)
DEFAULT_PIXEL_SIZE_XY_NM = 103.0
DEFAULT_PIXEL_SIZE_Z_NM = 216.7


@dataclass(frozen=True)
class MeshStatistics:
    """Statistics for mesh geometry."""
    n_vertices: int
    n_faces: int
    n_edges: int
    volume_pixels3: float
    volume_um3: float
    surface_area_pixels2: float
    surface_area_um2: float
    is_watertight: bool
    euler_number: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for backwards compatibility."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class CurvatureStatistics:
    """Statistics for curvature distribution."""
    mean: float
    std: float
    sem: float
    median: float
    min: float
    max: float
    percentiles: Dict[int, float] = field(default_factory=dict)
    
    @classmethod
    def from_array(cls, curvature: np.ndarray) -> 'CurvatureStatistics':
        """Create from curvature array."""
        percentile_values = [1, 5, 25, 50, 75, 95, 99]
        return cls(
            mean=float(np.mean(curvature)),
            std=float(np.std(curvature)),
            sem=float(np.std(curvature) / np.sqrt(len(curvature))),
            median=float(np.median(curvature)),
            min=float(np.min(curvature)),
            max=float(np.max(curvature)),
            percentiles={p: float(np.percentile(curvature, p)) for p in percentile_values}
        )


@dataclass(frozen=True)
class QualityMetrics:
    """Mesh quality metrics."""
    mean_edge_length: float
    std_edge_length: float
    min_edge_length: float
    max_edge_length: float
    mean_face_area: float
    std_face_area: float
    aspect_ratio_mean: float
    aspect_ratio_std: float
    
    def get_warnings(self) -> list[str]:
        """Check for quality issues."""
        warnings = []
        if self.aspect_ratio_mean > 3.0:
            warnings.append(f"High aspect ratio: {self.aspect_ratio_mean:.2f}")
        if self.std_edge_length / self.mean_edge_length > 0.5:
            warnings.append("High edge length variation")
        return warnings


@dataclass
class AnalysisResults:
    """Container for all analysis results."""
    mesh_stats: Optional[MeshStatistics] = None
    curvature_stats: Optional[CurvatureStatistics] = None
    quality_metrics: Optional[QualityMetrics] = None
    
    def is_complete(self) -> bool:
        """Check if all analyses have been run."""
        return all([self.mesh_stats, self.curvature_stats, self.quality_metrics])
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        if not self.mesh_stats:
            return "No analysis performed yet."
        
        lines = [
            "=== Analysis Summary ===",
            f"Vertices: {self.mesh_stats.n_vertices:,}",
            f"Faces: {self.mesh_stats.n_faces:,}",
            f"Volume: {self.mesh_stats.volume_um3:.2f} μm³",
            f"Surface Area: {self.mesh_stats.surface_area_um2:.2f} μm²",
        ]
        
        if self.curvature_stats:
            lines.extend([
                f"\nCurvature: {self.curvature_stats.mean:.4f} ± {self.curvature_stats.std:.4f}",
                f"Range: [{self.curvature_stats.min:.4f}, {self.curvature_stats.max:.4f}]"
            ])
        
        if self.quality_metrics:
            warnings = self.quality_metrics.get_warnings()
            if warnings:
                lines.append("\nWarnings:")
                lines.extend(f"  - {w}" for w in warnings)

        return "\n".join(lines)


@dataclass(frozen=True)
class MeshParameters:
    """Parameters used for mesh generation."""
    mesh_mode: str
    inside_gamma: float
    inside_blur: Union[float, List[float]]
    filter_scales: List[float]
    filter_num_std_surface: float
    inside_dilate_radius: float
    inside_erode_radius: float
    smooth_mesh_mode: str
    smooth_mesh_iterations: int
    use_undeconvolved: bool
    image_gamma: float
    scale_otsu: float
    smooth_image_size: float
    curvature_median_filter_radius: int
    curvature_smooth_on_mesh_iterations: int
    register_images: bool
    save_raw_images: bool
    registration_mode: str


# Default mesh parameters (single source of truth)
DEFAULT_MESH_PARAMETERS = MeshParameters(
    mesh_mode='otsu',
    inside_gamma=0.6,
    inside_blur=2,
    filter_scales=[1.5, 2, 4],
    filter_num_std_surface=2.0,
    inside_dilate_radius=5,
    inside_erode_radius=6.5,
    smooth_mesh_mode='curvature',
    smooth_mesh_iterations=6,
    use_undeconvolved=False,
    image_gamma=1.0,
    scale_otsu=1.0,
    smooth_image_size=0.0,
    curvature_median_filter_radius=2,
    curvature_smooth_on_mesh_iterations=20,
    register_images=False,
    save_raw_images=False,
    registration_mode='translation'
)


@dataclass(frozen=True)
class ProcessingMetadata:
    """Provenance information for mesh generation."""
    pixel_size_xy_nm: float
    pixel_size_z_nm: float
    time_interval_sec: float
    source_image_path: str
    source_image_name: str
    processing_date: Optional[datetime]
    matlab_version: Optional[str]
    mesh_parameters: MeshParameters


@dataclass(frozen=True)
class AuxiliaryMeshData:
    """Optional auxiliary mesh data."""
    face_normals: Optional[np.ndarray] = None
    gaussian_curvature: Optional[np.ndarray] = None
    mean_curvature_raw: Optional[np.ndarray] = None
    neighbors: Optional[np.ndarray] = None
    image_surface: Optional[np.ndarray] = None


@dataclass
class MeshFrame:
    """Complete mesh data for a single timepoint.

    Unified data container that replaces MeshAnalyzer for data storage.
    Includes lazy-computed properties to avoid redundant calculations.
    """
    vertices: np.ndarray
    faces: np.ndarray
    curvature: np.ndarray
    mesh: 'vedo.Mesh'  # Forward reference for type hint
    time_index: int = 0
    metadata: Optional[ProcessingMetadata] = None
    auxiliary: Optional[AuxiliaryMeshData] = None
    _face_centers_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @property
    def face_centers(self) -> np.ndarray:
        """Compute face centers on demand, cache result."""
        if self._face_centers_cache is None:
            object.__setattr__(self, '_face_centers_cache',
                             self.vertices[self.faces].mean(axis=1))
        return self._face_centers_cache

    @property
    def n_faces(self) -> int:
        """Number of faces in mesh."""
        return len(self.faces)

    @property
    def n_vertices(self) -> int:
        """Number of vertices in mesh."""
        return len(self.vertices)

    @property
    def pixel_size_xy_um(self) -> Optional[float]:
        """XY pixel size in micrometers (from metadata)."""
        if self.metadata:
            return self.metadata.pixel_size_xy_nm / 1000.0
        return None

    @property
    def pixel_size_z_um(self) -> Optional[float]:
        """Z pixel size in micrometers (from metadata)."""
        if self.metadata:
            return self.metadata.pixel_size_z_nm / 1000.0
        return None