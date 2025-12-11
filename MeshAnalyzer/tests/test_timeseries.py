"""
Comprehensive tests for TimeSeriesManager functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import scipy.io as sio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from MeshAnalyzer import TimeSeriesManager, TimeFrameInfo, TimeSeriesData
from MeshAnalyzer.analyzer import MeshAnalyzer




@pytest.fixture
def mock_data_dir(tmp_path):
    """
    Create mock time-series data directory with synthetic .mat files.

    Creates 5 time frames with matching surface and curvature files.
    """
    data_dir = tmp_path / "timeseries_data"
    data_dir.mkdir()

    # Generate simple cube mesh
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top
    ], dtype=np.float32)

    faces = np.array([
        [1, 2, 3], [1, 3, 4],  # Bottom
        [5, 6, 7], [5, 7, 8],  # Top
        [1, 2, 6], [1, 6, 5],  # Front
        [3, 4, 8], [3, 8, 7],  # Back
        [1, 4, 8], [1, 8, 5],  # Left
        [2, 3, 7], [2, 7, 6]   # Right
    ], dtype=np.int32)

    n_faces = len(faces)

    # Create 5 time frames
    for t in [1, 2, 3, 4, 5]:
        # Surface file
        surface_file = data_dir / f"surface_1_{t}.mat"
        surface_data = {
            'surface': {
                'vertices': vertices + 0.1 * t,  # Slight displacement over time
                'faces': faces
            }
        }
        sio.savemat(str(surface_file), surface_data)

        # Curvature file - vary curvature over time
        curv_file = data_dir / f"meanCurvature_1_{t}.mat"
        curvature = np.random.randn(n_faces) * 0.5 + 0.1 * t  # Increasing trend
        curv_data = {'meanCurvature': curvature}
        sio.savemat(str(curv_file), curv_data)

    return data_dir


@pytest.fixture
def real_data_dir():
    """
    Return path to real data directory if it exists.
    Skip test if data not available.
    """
    data_path = Path("/Volumes/T7/Analysis_Neutros/03/Morphology/Analysis/Mesh/ch1")

    if not data_path.exists():
        pytest.skip(f"Real data not available at {data_path}")

    return data_path




class TestTimeFrameInfo:
    """Test TimeFrameInfo dataclass."""

    def test_creation_valid(self, mock_data_dir):
        """Test creating TimeFrameInfo with valid paths."""
        surface_path = mock_data_dir / "surface_1_1.mat"
        curv_path = mock_data_dir / "meanCurvature_1_1.mat"

        info = TimeFrameInfo(
            time_index=1,
            surface_path=surface_path,
            curvature_path=curv_path
        )

        assert info.time_index == 1
        assert info.surface_path == surface_path
        assert info.curvature_path == curv_path
        assert info.timestamp_sec is None

    def test_creation_with_timestamp(self, mock_data_dir):
        """Test creating TimeFrameInfo with timestamp."""
        surface_path = mock_data_dir / "surface_1_1.mat"
        curv_path = mock_data_dir / "meanCurvature_1_1.mat"

        info = TimeFrameInfo(
            time_index=1,
            surface_path=surface_path,
            curvature_path=curv_path,
            timestamp_sec=10.5
        )

        assert info.timestamp_sec == 10.5

    def test_invalid_surface_path(self, mock_data_dir):
        """Test that missing surface file raises error."""
        surface_path = mock_data_dir / "nonexistent.mat"
        curv_path = mock_data_dir / "meanCurvature_1_1.mat"

        with pytest.raises(FileNotFoundError, match="Surface file not found"):
            TimeFrameInfo(
                time_index=1,
                surface_path=surface_path,
                curvature_path=curv_path
            )

    def test_invalid_curvature_path(self, mock_data_dir):
        """Test that missing curvature file raises error."""
        surface_path = mock_data_dir / "surface_1_1.mat"
        curv_path = mock_data_dir / "nonexistent.mat"

        with pytest.raises(FileNotFoundError, match="Curvature file not found"):
            TimeFrameInfo(
                time_index=1,
                surface_path=surface_path,
                curvature_path=curv_path
            )

    def test_sorting(self, mock_data_dir):
        """Test that TimeFrameInfo can be sorted by time_index."""
        surface_path = mock_data_dir / "surface_1_1.mat"
        curv_path = mock_data_dir / "meanCurvature_1_1.mat"

        info1 = TimeFrameInfo(1, surface_path, curv_path)
        info2 = TimeFrameInfo(2, surface_path, curv_path)
        info3 = TimeFrameInfo(3, surface_path, curv_path)

        unsorted = [info3, info1, info2]
        sorted_list = sorted(unsorted)

        assert sorted_list[0].time_index == 1
        assert sorted_list[1].time_index == 2
        assert sorted_list[2].time_index == 3


class TestTimeSeriesData:
    """Test TimeSeriesData dataclass."""

    def test_creation(self, mock_data_dir):
        """Test creating TimeSeriesData."""
        surface_path = mock_data_dir / "surface_1_1.mat"
        curv_path = mock_data_dir / "meanCurvature_1_1.mat"

        analyzer = MeshAnalyzer(str(surface_path), str(curv_path))
        analyzer.load_data(verbose=False)

        face_centers = np.array([np.mean(analyzer.vertices[face], axis=0)
                                for face in analyzer.faces])

        data = TimeSeriesData(
            time_index=1,
            analyzer=analyzer,
            face_centers=face_centers,
            curvature=analyzer.curvature.copy()
        )

        assert data.time_index == 1
        assert data.n_faces == len(analyzer.faces)
        assert data.n_vertices == len(analyzer.vertices)
        assert data.face_centers.shape[0] == data.n_faces


class TestTimeSeriesManager:
    """Test TimeSeriesManager class."""

    def test_initialization(self, mock_data_dir):
        """Test basic initialization."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            verbose=False
        )

        assert manager.data_dir == mock_data_dir
        assert manager.pixel_size_xy == 0.1
        assert manager.pixel_size_z == 0.2
        assert manager.cache_mode == 'lazy'
        assert len(manager) == 0  # No frames discovered yet

    def test_invalid_cache_mode(self, mock_data_dir):
        """Test that invalid cache_mode raises error."""
        with pytest.raises(ValueError, match="cache_mode must be"):
            TimeSeriesManager(
                data_dir=mock_data_dir,
                pixel_size_xy=0.1,
                pixel_size_z=0.2,
                cache_mode='invalid'
            )

    def test_discover_frames(self, mock_data_dir):
        """Test frame discovery."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            verbose=False
        )

        n_frames = manager.discover_frames()

        assert n_frames == 5
        assert len(manager) == 5
        assert list(manager.keys()) == [1, 2, 3, 4, 5]

    def test_discover_frames_no_data(self, tmp_path):
        """Test discovery with no data files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        manager = TimeSeriesManager(
            data_dir=empty_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            verbose=False
        )

        with pytest.raises(FileNotFoundError, match="No surface files found"):
            manager.discover_frames()

    def test_load_frame(self, mock_data_dir):
        """Test loading individual frame."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            verbose=False
        )
        manager.discover_frames()

        data = manager.load_frame(1)

        assert isinstance(data, TimeSeriesData)
        assert data.time_index == 1
        assert data.n_faces > 0
        assert data.n_vertices > 0
        assert data.curvature.shape[0] == data.n_faces
        assert data.face_centers.shape == (data.n_faces, 3)

    def test_load_invalid_frame(self, mock_data_dir):
        """Test loading non-existent frame."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            verbose=False
        )
        manager.discover_frames()

        with pytest.raises(KeyError, match="Time index 99 not found"):
            manager.load_frame(99)

    def test_dict_like_access(self, mock_data_dir):
        """Test dictionary-style access."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            verbose=False
        )
        manager.discover_frames()

        # __getitem__
        data = manager[1]
        assert data.time_index == 1

        # __contains__
        assert 1 in manager
        assert 99 not in manager

        # __len__
        assert len(manager) == 5

    def test_iteration(self, mock_data_dir):
        """Test iteration over frames."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            verbose=False
        )
        manager.discover_frames()

        indices = []
        for time_idx, data in manager:
            indices.append(time_idx)
            assert isinstance(data, TimeSeriesData)
            assert data.time_index == time_idx

        assert indices == [1, 2, 3, 4, 5]

    def test_cache_mode_none(self, mock_data_dir):
        """Test cache_mode='none' (no caching)."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            cache_mode='none',
            verbose=False
        )
        manager.discover_frames()

        # Load frame
        data1 = manager.load_frame(1)

        # Cache should be empty
        cache_stats = manager.get_cache_stats()
        assert cache_stats['currently_cached'] == 0

    def test_cache_mode_lazy(self, mock_data_dir):
        """Test cache_mode='lazy' (lazy loading with LRU)."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            cache_mode='lazy',
            max_cached_frames=2,
            verbose=False
        )
        manager.discover_frames()

        # Load frames
        manager.load_frame(1)
        manager.load_frame(2)

        cache_stats = manager.get_cache_stats()
        assert cache_stats['currently_cached'] == 2

        # Load third frame - should evict frame 1 (LRU)
        manager.load_frame(3)

        cache_stats = manager.get_cache_stats()
        assert cache_stats['currently_cached'] == 2
        assert 1 not in cache_stats['cached_indices']

    def test_cache_mode_all(self, mock_data_dir):
        """Test cache_mode='all' (preload all frames)."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            cache_mode='all',
            verbose=False
        )

        # Discovery should preload all
        manager.discover_frames()

        cache_stats = manager.get_cache_stats()
        assert cache_stats['currently_cached'] == 5
        assert cache_stats['cached_indices'] == [1, 2, 3, 4, 5]

    def test_validate_frames(self, mock_data_dir):
        """Test frame validation."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            verbose=False
        )
        manager.discover_frames()

        results = manager.validate_frames()

        assert results['is_valid'] is True
        assert results['topology_consistent'] is True
        assert len(results['errors']) == 0

    def test_get_normalized_curvature_symmetric(self, mock_data_dir):
        """Test symmetric normalization."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            verbose=False
        )
        manager.discover_frames()

        normalized = manager.get_normalized_curvature(method='symmetric')

        # Check all frames normalized
        assert len(normalized) == 5
        assert all(idx in normalized for idx in [1, 2, 3, 4, 5])

        # Check normalization range (should be in [-1, +1])
        all_values = np.concatenate(list(normalized.values()))
        assert all_values.min() >= -1.0
        assert all_values.max() <= 1.0

    def test_get_normalized_curvature_full(self, mock_data_dir):
        """Test min-max normalization."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            verbose=False
        )
        manager.discover_frames()

        normalized = manager.get_normalized_curvature(method='full')

        # Check normalization range (should be in [0, 1])
        all_values = np.concatenate(list(normalized.values()))
        assert all_values.min() >= 0.0
        assert all_values.max() <= 1.0

    def test_get_normalized_curvature_percentiles(self, mock_data_dir):
        """Test normalization with percentile clipping."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            verbose=False
        )
        manager.discover_frames()

        normalized = manager.get_normalized_curvature(
            method='symmetric',
            percentile_range=(5, 95)
        )

        assert len(normalized) == 5

    def test_clear_cache(self, mock_data_dir):
        """Test cache clearing."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            cache_mode='lazy',
            verbose=False
        )
        manager.discover_frames()

        # Load some frames
        manager.load_frame(1)
        manager.load_frame(2)

        assert manager.get_cache_stats()['currently_cached'] == 2

        # Clear cache
        manager.clear_cache()

        assert manager.get_cache_stats()['currently_cached'] == 0

    def test_string_representation(self, mock_data_dir):
        """Test __str__ and __repr__."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            verbose=False
        )
        manager.discover_frames()

        str_repr = str(manager)
        assert "TimeSeriesManager" in str_repr
        assert "5 frames" in str_repr
        assert "T01-T05" in str_repr

        repr_str = repr(manager)
        assert "TimeSeriesManager" in repr_str
        assert "n_frames=5" in repr_str




class TestIntegrationWithRealData:
    """Integration tests with real neutrophil data (if available)."""

    def test_real_data_discovery(self, real_data_dir):
        """Test discovery with real data."""
        manager = TimeSeriesManager(
            data_dir=real_data_dir,
            pixel_size_xy=0.1030,
            pixel_size_z=0.2167,
            verbose=True
        )

        n_frames = manager.discover_frames()

        assert n_frames > 0
        print(f"\nDiscovered {n_frames} real frames")

    def test_real_data_validation(self, real_data_dir):
        """Test validation with real data."""
        manager = TimeSeriesManager(
            data_dir=real_data_dir,
            pixel_size_xy=0.1030,
            pixel_size_z=0.2167,
            cache_mode='lazy',
            max_cached_frames=10,
            verbose=True
        )

        manager.discover_frames()
        results = manager.validate_frames()

        print("\nValidation results:")
        print(f"  Valid: {results['is_valid']}")
        print(f"  Topology consistent: {results['topology_consistent']}")
        print(f"  Errors: {len(results['errors'])}")
        print(f"  Warnings: {len(results['warnings'])}")

        # For biological data, topology changes are expected (cell changing shape)
        # We just want to ensure no NaN values or missing files
        # So we check that no critical errors exist (like missing data or NaN values)
        nan_errors = [e for e in results['errors'] if 'NaN' in e or 'missing' in e.lower()]
        assert len(nan_errors) == 0, f"Critical errors found: {nan_errors}"

    def test_real_data_normalization(self, real_data_dir):
        """Test normalization with real data."""
        manager = TimeSeriesManager(
            data_dir=real_data_dir,
            pixel_size_xy=0.1030,
            pixel_size_z=0.2167,
            cache_mode='lazy',
            max_cached_frames=5,
            verbose=True
        )

        manager.discover_frames()

        # Test symmetric normalization
        normalized = manager.get_normalized_curvature(
            method='symmetric',
            percentile_range=(5, 95)
        )

        assert len(normalized) > 0

        # Check normalization quality
        all_values = np.concatenate(list(normalized.values()))
        print(f"\nNormalized curvature statistics:")
        print(f"  Range: [{all_values.min():.4f}, {all_values.max():.4f}]")
        print(f"  Mean: {all_values.mean():.4f}")
        print(f"  Std: {all_values.std():.4f}")




class TestPerformance:
    """Performance and memory tests."""

    def test_memory_management(self, mock_data_dir):
        """Test that LRU eviction prevents unbounded memory growth."""
        manager = TimeSeriesManager(
            data_dir=mock_data_dir,
            pixel_size_xy=0.1,
            pixel_size_z=0.2,
            cache_mode='lazy',
            max_cached_frames=2,
            verbose=False
        )
        manager.discover_frames()

        # Load all 5 frames sequentially
        for i in range(1, 6):
            manager.load_frame(i)

        # Should only have last 2 frames cached
        cache_stats = manager.get_cache_stats()
        assert cache_stats['currently_cached'] <= 2




if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
