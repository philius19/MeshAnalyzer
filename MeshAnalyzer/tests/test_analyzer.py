"""
Comprehensive tests for MeshAnalyzer core functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import scipy.io as sio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from MeshAnalyzer import MeshAnalyzer
from MeshAnalyzer.datatypes import AnalysisResults, MeshStatistics, CurvatureStatistics




@pytest.fixture
def simple_cube_mesh(tmp_path):
    """
    Create simple cube mesh for testing.

    Creates a unit cube with 8 vertices and 12 triangular faces.
    """
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()

    # Cube vertices
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top
    ], dtype=np.float32)

    # Cube faces (1-indexed for MATLAB)
    faces = np.array([
        [1, 2, 3], [1, 3, 4],  # Bottom
        [5, 6, 7], [5, 7, 8],  # Top
        [1, 2, 6], [1, 6, 5],  # Front
        [3, 4, 8], [3, 8, 7],  # Back
        [1, 4, 8], [1, 8, 5],  # Left
        [2, 3, 7], [2, 7, 6]   # Right
    ], dtype=np.int32)

    n_faces = len(faces)

    # Surface file
    surface_file = data_dir / "surface.mat"
    surface_data = {
        'surface': {
            'vertices': vertices,
            'faces': faces
        }
    }
    sio.savemat(str(surface_file), surface_data)

    # Curvature file (uniform curvature)
    curv_file = data_dir / "curvature.mat"
    curvature = np.ones(n_faces, dtype=np.float32) * 0.5
    curv_data = {'meanCurvature': curvature}
    sio.savemat(str(curv_file), curv_data)

    return surface_file, curv_file, vertices, faces, curvature


@pytest.fixture
def inverted_mesh(tmp_path):
    """Create mesh with inverted normals (negative volume)."""
    data_dir = tmp_path / "inverted_data"
    data_dir.mkdir()

    # Same cube but with flipped face winding
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float32)

    # Inverted faces (swap indices to invert normals)
    faces = np.array([
        [1, 3, 2], [1, 4, 3],  # Inverted
        [5, 7, 6], [5, 8, 7],
        [1, 6, 2], [1, 5, 6],
        [3, 8, 4], [3, 7, 8],
        [1, 8, 4], [1, 5, 8],
        [2, 7, 3], [2, 6, 7]
    ], dtype=np.int32)

    surface_file = data_dir / "surface_inverted.mat"
    curv_file = data_dir / "curvature_inverted.mat"

    sio.savemat(str(surface_file), {'surface': {'vertices': vertices, 'faces': faces}})
    sio.savemat(str(curv_file), {'meanCurvature': np.ones(len(faces))})

    return surface_file, curv_file


@pytest.fixture
def large_mesh(tmp_path):
    """Create larger mesh for performance testing."""
    data_dir = tmp_path / "large_data"
    data_dir.mkdir()

    # Create a larger mesh (sphere-like)
    n_points = 1000
    phi = np.linspace(0, np.pi, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)

    vertices = np.column_stack([
        np.sin(phi).flatten(),
        np.cos(phi).flatten() * np.sin(theta).flatten(),
        np.cos(phi).flatten() * np.cos(theta).flatten()
    ]).astype(np.float32)

    # Generate faces (simplified triangulation)
    n_vert = len(vertices)
    faces_list = []
    for i in range(0, n_vert-51, 50):
        for j in range(49):
            faces_list.append([i+j+1, i+j+2, i+j+51])
            faces_list.append([i+j+2, i+j+52, i+j+51])

    faces = np.array(faces_list, dtype=np.int32)

    surface_file = data_dir / "surface_large.mat"
    curv_file = data_dir / "curvature_large.mat"

    sio.savemat(str(surface_file), {'surface': {'vertices': vertices, 'faces': faces}})
    sio.savemat(str(curv_file), {'meanCurvature': np.random.randn(len(faces))})

    return surface_file, curv_file




class TestMeshAnalyzerInitialization:
    """Test MeshAnalyzer initialization and validation."""

    def test_initialization_valid(self, simple_cube_mesh):
        """Test initialization with valid paths."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(
            str(surface_file),
            str(curv_file),
            pixel_size_xy=0.1,
            pixel_size_z=0.2
        )

        assert analyzer.surface_path == surface_file
        assert analyzer.curvature_path == curv_file
        assert analyzer.pixel_size_xy == 0.1
        assert analyzer.pixel_size_z == 0.2
        assert not analyzer.is_loaded

    def test_initialization_default_pixel_sizes(self, simple_cube_mesh):
        """Test initialization with default pixel sizes."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))

        assert analyzer.pixel_size_xy == MeshAnalyzer.DEFAULT_PIXEL_SIZE_XY
        assert analyzer.pixel_size_z == MeshAnalyzer.DEFAULT_PIXEL_SIZE_Z

    def test_initialization_missing_surface_file(self, tmp_path):
        """Test initialization with missing surface file."""
        surface_file = tmp_path / "nonexistent_surface.mat"
        curv_file = tmp_path / "curvature.mat"
        curv_file.touch()

        with pytest.raises(FileNotFoundError, match="Surface file not found"):
            analyzer = MeshAnalyzer(str(surface_file), str(curv_file))

    def test_initialization_missing_curvature_file(self, tmp_path):
        """Test initialization with missing curvature file."""
        surface_file = tmp_path / "surface.mat"
        surface_file.touch()
        curv_file = tmp_path / "nonexistent_curv.mat"

        with pytest.raises(FileNotFoundError, match="Curvature file not found"):
            analyzer = MeshAnalyzer(str(surface_file), str(curv_file))

    def test_initialization_wrong_file_format(self, tmp_path):
        """Test initialization with wrong file format."""
        surface_file = tmp_path / "surface.txt"
        surface_file.touch()
        curv_file = tmp_path / "curvature.txt"
        curv_file.touch()

        with pytest.raises(ValueError, match="not supported"):
            analyzer = MeshAnalyzer(str(surface_file), str(curv_file))


class TestMeshAnalyzerLoading:
    """Test data loading functionality."""

    def test_load_data_success(self, simple_cube_mesh):
        """Test successful data loading."""
        surface_file, curv_file, vertices, faces, curvature = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)

        assert analyzer.is_loaded
        assert analyzer.vertices is not None
        assert analyzer.faces is not None
        assert analyzer.mesh is not None
        assert analyzer.curvature is not None

        assert len(analyzer.vertices) == len(vertices)
        assert len(analyzer.faces) == len(faces)

    def test_load_data_vertex_count(self, simple_cube_mesh):
        """Test that vertex count matches expected."""
        surface_file, curv_file, vertices, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)

        assert len(analyzer.vertices) == 8  # Cube has 8 vertices

    def test_load_data_face_count(self, simple_cube_mesh):
        """Test that face count matches expected."""
        surface_file, curv_file, _, faces, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)

        assert len(analyzer.faces) == 12  # Cube has 12 triangular faces

    def test_load_data_curvature_length_match(self, simple_cube_mesh):
        """Test that curvature length matches face count."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)

        assert len(analyzer.curvature) == len(analyzer.faces)

    def test_load_data_inverted_mesh_correction(self, inverted_mesh):
        """Test that inverted mesh is automatically corrected."""
        surface_file, curv_file = inverted_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)

        # After loading, volume should be positive
        assert analyzer.mesh.volume() > 0

    def test_load_data_face_indices_zero_based(self, simple_cube_mesh):
        """Test that MATLAB 1-indexed faces are converted to 0-indexed."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)

        # Faces should be 0-indexed (min = 0)
        assert analyzer.faces.min() >= 0
        # Faces should not exceed vertex count
        assert analyzer.faces.max() < len(analyzer.vertices)


class TestMeshAnalyzerStatistics:
    """Test statistics calculation."""

    def test_calculate_statistics_before_loading(self, simple_cube_mesh):
        """Test that calculate_statistics fails before loading data."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))

        with pytest.raises(RuntimeError, match="Must load data first"):
            analyzer.calculate_statistics()

    def test_calculate_statistics_success(self, simple_cube_mesh):
        """Test successful statistics calculation."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)
        results = analyzer.calculate_statistics()

        assert isinstance(results, AnalysisResults)
        assert results.is_complete()
        assert results.mesh_stats is not None
        assert results.curvature_stats is not None
        assert results.quality_metrics is not None

    def test_calculate_statistics_mesh_stats(self, simple_cube_mesh):
        """Test mesh statistics values."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)
        results = analyzer.calculate_statistics()

        assert results.mesh_stats.n_vertices == 8
        assert results.mesh_stats.n_faces == 12
        assert results.mesh_stats.volume_pixels3 > 0
        assert results.mesh_stats.surface_area_pixels2 > 0

    def test_calculate_statistics_curvature_stats(self, simple_cube_mesh):
        """Test curvature statistics values."""
        surface_file, curv_file, _, _, curvature = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)
        results = analyzer.calculate_statistics()

        # Curvature is uniform 0.5
        assert abs(results.curvature_stats.mean - 0.5) < 0.01
        assert results.curvature_stats.std < 0.1  # Should be very small

    def test_calculate_statistics_caching(self, simple_cube_mesh):
        """Test that statistics are cached."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)

        results1 = analyzer.calculate_statistics()
        results2 = analyzer.calculate_statistics()

        # Should return same object (cached)
        assert results1 is results2

    def test_calculate_statistics_force_recalculate(self, simple_cube_mesh):
        """Test force recalculation."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)

        results1 = analyzer.calculate_statistics()
        results2 = analyzer.calculate_statistics(force_recalculate=True)

        # Should have same values but different objects
        assert results1.mesh_stats.n_vertices == results2.mesh_stats.n_vertices

    def test_calculate_statistics_dict(self, simple_cube_mesh):
        """Test dictionary output format."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)
        stats_dict = analyzer.calculate_statistics_dict()

        assert isinstance(stats_dict, dict)
        assert 'mesh' in stats_dict
        assert 'curvature' in stats_dict
        assert 'quality' in stats_dict

        assert 'n_vertices' in stats_dict['mesh']
        assert 'mean' in stats_dict['curvature']


class TestMeshAnalyzerProperties:
    """Test analyzer properties and methods."""

    def test_is_loaded_property(self, simple_cube_mesh):
        """Test is_loaded property."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        assert not analyzer.is_loaded

        analyzer.load_data(verbose=False)
        assert analyzer.is_loaded

    def test_physical_dimensions_before_loading(self, simple_cube_mesh):
        """Test physical_dimensions before loading data."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        dims = analyzer.physical_dimensions

        assert dims == {}

    def test_physical_dimensions_after_loading(self, simple_cube_mesh):
        """Test physical_dimensions after loading data."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file),
                               pixel_size_xy=0.1, pixel_size_z=0.2)
        analyzer.load_data(verbose=False)
        dims = analyzer.physical_dimensions

        assert 'x_um' in dims
        assert 'y_um' in dims
        assert 'z_um' in dims
        assert dims['x_um'] > 0

    def test_str_representation(self, simple_cube_mesh):
        """Test string representation."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))

        str_not_loaded = str(analyzer)
        assert "not loaded" in str_not_loaded.lower()

        analyzer.load_data(verbose=False)
        str_loaded = str(analyzer)
        assert "8 vertices" in str_loaded
        assert "12 faces" in str_loaded

    def test_repr_representation(self, simple_cube_mesh):
        """Test repr representation."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        repr_str = repr(analyzer)

        assert "MeshAnalyzer" in repr_str
        assert "surface.mat" in repr_str
        assert "curvature.mat" in repr_str


class TestMeshAnalyzerContextManager:
    """Test context manager functionality."""

    def test_context_manager_auto_load(self, simple_cube_mesh):
        """Test that context manager auto-loads data."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        with MeshAnalyzer(str(surface_file), str(curv_file)) as analyzer:
            assert analyzer.is_loaded
            assert analyzer.vertices is not None

    def test_context_manager_cleanup(self, simple_cube_mesh):
        """Test that context manager cleans up properly."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        with MeshAnalyzer(str(surface_file), str(curv_file)) as analyzer:
            pass

        # Should still be loaded after exit
        assert analyzer.is_loaded


class TestMeshAnalyzerEdgeCases:
    """Test edge cases and error handling."""

    def test_single_triangle_mesh(self, tmp_path):
        """Test with minimal valid mesh (tetrahedron - smallest closed mesh)."""
        data_dir = tmp_path / "tetrahedron"
        data_dir.mkdir()

        # Tetrahedron: 4 vertices, 4 faces
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.289, 0.816]
        ], dtype=np.float32)
        faces = np.array([
            [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]
        ], dtype=np.int32)

        surface_file = data_dir / "surface.mat"
        curv_file = data_dir / "curvature.mat"

        sio.savemat(str(surface_file), {'surface': {'vertices': vertices, 'faces': faces}})
        sio.savemat(str(curv_file), {'meanCurvature': np.ones(len(faces)) * 0.5})

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)

        assert len(analyzer.vertices) == 4
        assert len(analyzer.faces) == 4

        results = analyzer.calculate_statistics()
        assert results.mesh_stats.n_faces == 4




class TestMeshAnalyzerIntegration:
    """Integration tests with complete workflows."""

    def test_complete_workflow(self, simple_cube_mesh):
        """Test complete analysis workflow."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        # Initialize
        analyzer = MeshAnalyzer(
            str(surface_file),
            str(curv_file),
            pixel_size_xy=0.1,
            pixel_size_z=0.2
        )

        # Load
        analyzer.load_data(verbose=False)
        assert analyzer.is_loaded

        # Calculate statistics
        results = analyzer.calculate_statistics()
        assert results.is_complete()

        # Get summary
        summary = results.summary()
        assert "Vertices" in summary
        assert "Faces" in summary

        # Get dictionary format
        stats_dict = analyzer.calculate_statistics_dict()
        assert len(stats_dict) == 3

    def test_workflow_with_context_manager(self, simple_cube_mesh):
        """Test workflow using context manager."""
        surface_file, curv_file, _, _, _ = simple_cube_mesh

        with MeshAnalyzer(str(surface_file), str(curv_file)) as analyzer:
            results = analyzer.calculate_statistics()
            assert results.is_complete()

            summary = results.summary()
            assert len(summary) > 0




class TestMeshAnalyzerPerformance:
    """Performance tests with larger meshes."""

    def test_large_mesh_loading(self, large_mesh):
        """Test loading performance with larger mesh."""
        surface_file, curv_file = large_mesh

        import time
        start = time.time()

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)

        elapsed = time.time() - start

        # Should load in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        assert analyzer.is_loaded

    def test_large_mesh_statistics(self, large_mesh):
        """Test statistics calculation with larger mesh."""
        surface_file, curv_file = large_mesh

        analyzer = MeshAnalyzer(str(surface_file), str(curv_file))
        analyzer.load_data(verbose=False)

        import time
        start = time.time()

        results = analyzer.calculate_statistics()

        elapsed = time.time() - start

        # Should calculate in reasonable time (< 10 seconds)
        assert elapsed < 10.0
        assert results.is_complete()




if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
