"""Time-series analysis extension for MeshAnalyzer."""
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Iterator, Tuple, List

import numpy as np

from .analyzer import MeshAnalyzer
from .datatypes import AnalysisResults

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeFrameInfo:
    """Metadata for a single time frame."""
    time_index: int
    surface_path: Path
    curvature_path: Path
    timestamp_sec: Optional[float] = None

    def __post_init__(self):
        if not self.surface_path.exists():
            raise FileNotFoundError(f"Surface file not found: {self.surface_path}")
        if not self.curvature_path.exists():
            raise FileNotFoundError(f"Curvature file not found: {self.curvature_path}")

    def __lt__(self, other):
        return self.time_index < other.time_index


@dataclass
class TimeSeriesData:
    """Loaded data for a single time frame."""
    time_index: int
    analyzer: MeshAnalyzer
    face_centers: np.ndarray
    curvature: np.ndarray

    @property
    def n_faces(self) -> int:
        return len(self.analyzer.faces)

    @property
    def n_vertices(self) -> int:
        return len(self.analyzer.vertices)


class TimeSeriesManager:
    """Manages time-lapse mesh data with lazy loading and caching (dict-like interface)."""

    def __init__(self, data_dir: Path, pixel_size_xy: float, pixel_size_z: float,
                 cache_mode: str = 'lazy', max_cached_frames: int = 10, verbose: bool = True):
        """Initialize TimeSeriesManager with caching options."""
        self.data_dir = Path(data_dir)
        self.pixel_size_xy = pixel_size_xy
        self.pixel_size_z = pixel_size_z
        self.verbose = verbose

        if cache_mode not in ('none', 'lazy', 'all'):
            raise ValueError(f"cache_mode must be 'none', 'lazy', or 'all', got: {cache_mode}")
        self.cache_mode = cache_mode
        self.max_cached_frames = max_cached_frames

        self._frame_info: OrderedDict[int, TimeFrameInfo] = OrderedDict()
        self._loaded_frames: OrderedDict[int, TimeSeriesData] = OrderedDict()
        self._access_order: List[int] = []
        self._global_stats_cache: Optional[Dict] = None

    def discover_frames(self, pattern: str = 'surface_*.mat') -> int:
        """Discover all time frames in data directory."""
        logger.info(f"Discovering frames in: {self.data_dir}")

        surface_files = sorted(self.data_dir.glob(pattern))
        if not surface_files:
            raise FileNotFoundError(f"No surface files found in {self.data_dir}")

        discovered_frames = []

        for surface_path in surface_files:
            try:
                parts = surface_path.stem.split('_')
                if len(parts) < 3:
                    raise ValueError("Invalid filename format")

                time_index = int(parts[2])
                curv_path = surface_path.parent / surface_path.name.replace('surface', 'meanCurvature')

                if not curv_path.exists():
                    continue

                frame_info = TimeFrameInfo(
                    time_index=time_index,
                    surface_path=surface_path,
                    curvature_path=curv_path
                )
                discovered_frames.append((time_index, frame_info))

            except (IndexError, ValueError):
                continue

        self._frame_info = OrderedDict(sorted(discovered_frames))

        if not self._frame_info:
            raise ValueError("No valid time frames found!")

        time_indices = list(self._frame_info.keys())
        logger.info(f"Found {len(self._frame_info)} frames: T{min(time_indices):02d} - T{max(time_indices):02d}")

        if self.cache_mode == 'all':
            self._preload_all_frames()

        return len(self._frame_info)

    def _preload_all_frames(self) -> None:
        """Preload all frames into cache."""
        logger.info("Preloading all frames...")

        for time_idx in self._frame_info.keys():
            data = self._load_frame_internal(time_idx)
            self._loaded_frames[time_idx] = data

        logger.info(f"Preloaded {len(self._loaded_frames)} frames")

    def _load_frame_internal(self, time_index: int) -> TimeSeriesData:
        """Load a single frame into memory."""
        frame_info = self._frame_info[time_index]

        analyzer = MeshAnalyzer(
            str(frame_info.surface_path),
            str(frame_info.curvature_path),
            pixel_size_xy=self.pixel_size_xy,
            pixel_size_z=self.pixel_size_z
        )
        analyzer.load_data(verbose=False)

        vertices = analyzer.vertices
        faces = analyzer.faces
        face_centers = np.array([np.mean(vertices[face], axis=0) for face in faces])

        data = TimeSeriesData(
            time_index=time_index,
            analyzer=analyzer,
            face_centers=face_centers,
            curvature=analyzer.curvature.copy()
        )

        return data

    def _evict_lru_frame(self) -> None:
        """Evict least recently used frame from cache."""
        if not self._access_order:
            return

        for time_idx in self._access_order:
            if time_idx in self._loaded_frames:
                del self._loaded_frames[time_idx]
                self._access_order.remove(time_idx)
                break

    def load_frame(self, time_index: int) -> TimeSeriesData:
        """Load a specific time frame with caching."""
        if time_index not in self._frame_info:
            available = list(self._frame_info.keys())
            raise KeyError(f"Time index {time_index} not found. Available: {available}")

        if time_index in self._access_order:
            self._access_order.remove(time_index)
        self._access_order.append(time_index)

        if time_index in self._loaded_frames:
            return self._loaded_frames[time_index]

        data = self._load_frame_internal(time_index)

        if self.cache_mode == 'none':
            return data
        elif self.cache_mode == 'lazy':
            if len(self._loaded_frames) >= self.max_cached_frames:
                self._evict_lru_frame()
            self._loaded_frames[time_index] = data
        elif self.cache_mode == 'all':
            self._loaded_frames[time_index] = data

        return data

    def validate_frames(self) -> Dict[str, any]:
        """Validate consistency across all frames (topology, data quality, temporal continuity)."""
        if not self._frame_info:
            raise RuntimeError("No frames discovered. Call discover_frames() first.")

        logger.info("Validating frames...")

        results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'topology_consistent': True,
            'temporal_gaps': []
        }

        first_idx = next(iter(self._frame_info.keys()))
        first_data = self.load_frame(first_idx)
        ref_n_vertices = first_data.n_vertices
        ref_n_faces = first_data.n_faces

        for time_idx in self._frame_info.keys():
            data = self.load_frame(time_idx)

            if data.n_vertices != ref_n_vertices or data.n_faces != ref_n_faces:
                msg = f"T{time_idx:02d}: Topology mismatch ({data.n_vertices} vertices, {data.n_faces} faces)"
                results['warnings'].append(msg)
                results['topology_consistent'] = False

            if np.any(np.isnan(data.curvature)):
                msg = f"T{time_idx:02d}: NaN values in curvature"
                results['errors'].append(msg)
                results['is_valid'] = False

        time_indices = sorted(self._frame_info.keys())
        for i in range(len(time_indices) - 1):
            gap = time_indices[i + 1] - time_indices[i]
            if gap > 1:
                results['temporal_gaps'].append((time_indices[i], time_indices[i + 1]))

        if results['temporal_gaps']:
            results['warnings'].append(f"Temporal gaps: {results['temporal_gaps']}")

        status = "Valid" if results['is_valid'] else f"{len(results['errors'])} errors"
        logger.info(f"Validation complete: {status}")

        if results['warnings']:
            for warning in results['warnings']:
                logger.warning(warning)
        if results['errors']:
            for error in results['errors']:
                logger.error(error)

        return results

    def get_normalized_curvature(self, method: str = 'symmetric',
                                 percentile_range: Optional[Tuple[float, float]] = None) -> Dict[int, np.ndarray]:
        """Get globally normalized curvature across all frames."""
        if not self._frame_info:
            raise RuntimeError("No frames discovered. Call discover_frames() first.")

        logger.info(f"Normalizing curvature globally (method={method})...")

        all_curvatures = []
        for time_idx in self._frame_info.keys():
            data = self.load_frame(time_idx)
            all_curvatures.append(data.curvature)

        all_curv_concat = np.concatenate(all_curvatures)

        if percentile_range:
            low, high = percentile_range
            vmin = np.percentile(all_curv_concat, low)
            vmax = np.percentile(all_curv_concat, high)
        else:
            vmin = np.min(all_curv_concat)
            vmax = np.max(all_curv_concat)

        if method == 'symmetric':
            vmax_sym = max(abs(vmin), abs(vmax))
            normalized = {}
            for time_idx, curv in zip(self._frame_info.keys(), all_curvatures):
                normalized[time_idx] = curv / vmax_sym
        elif method == 'full':
            normalized = {}
            for time_idx, curv in zip(self._frame_info.keys(), all_curvatures):
                normalized[time_idx] = (curv - vmin) / (vmax - vmin)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'symmetric' or 'full'")

        return normalized

    def __getitem__(self, time_index: int) -> TimeSeriesData:
        return self.load_frame(time_index)

    def __iter__(self) -> Iterator[Tuple[int, TimeSeriesData]]:
        for time_idx in self._frame_info.keys():
            yield time_idx, self.load_frame(time_idx)

    def __len__(self) -> int:
        return len(self._frame_info)

    def __contains__(self, time_index: int) -> bool:
        return time_index in self._frame_info

    def keys(self) -> Iterator[int]:
        return iter(self._frame_info.keys())

    def items(self) -> Iterator[Tuple[int, TimeSeriesData]]:
        return self.__iter__()

    def __str__(self) -> str:
        if not self._frame_info:
            return "TimeSeriesManager(no frames)"
        time_indices = list(self._frame_info.keys())
        return f"TimeSeriesManager({len(self)} frames: T{min(time_indices):02d}-T{max(time_indices):02d}, cache_mode='{self.cache_mode}')"

    def __repr__(self) -> str:
        return f"TimeSeriesManager(data_dir='{self.data_dir}', n_frames={len(self)}, cache_mode='{self.cache_mode}', cached={len(self._loaded_frames)})"

    def clear_cache(self) -> None:
        """Clear all cached frames."""
        self._loaded_frames.clear()
        self._access_order.clear()
        self._global_stats_cache = None

    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        return {
            'cache_mode': self.cache_mode,
            'max_cached_frames': self.max_cached_frames,
            'currently_cached': len(self._loaded_frames),
            'cached_indices': list(self._loaded_frames.keys()),
            'total_frames': len(self._frame_info)
        }
