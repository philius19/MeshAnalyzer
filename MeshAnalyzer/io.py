"""File input/output operations for mesh analysis."""
from pathlib import Path
from typing import Tuple, List
import json

import numpy as np
import scipy.io as sio
import vedo
from mat73 import loadmat as loadmat_v73


def loadmat(filepath: str) -> dict:
    """Load MATLAB file, automatically handling both v7.3 (HDF5) and older formats."""
    try:
        return loadmat_v73(filepath)
    except (TypeError, OSError):
        return sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)

def validate_file_paths(surface_path: Path, curvature_path: Path,
                       supported_formats: List[str]) -> None:
    """Validate input files exist and have correct format."""
    for path, name in [(surface_path, "Surface"), (curvature_path, "Curvature")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

        if path.suffix not in supported_formats:
            raise ValueError(f"{name} file format {path.suffix} not supported. "
                           f"Supported formats: {supported_formats}")


def load_surface_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray, vedo.Mesh]:
    """Load surface mesh data from .mat file (both v7.3 and older formats)."""
    surface_data = loadmat(str(filepath))
    surface = surface_data['surface']

    if isinstance(surface, dict):
        vertices = np.array(surface['vertices'], dtype=np.float32)
        faces = np.array(surface['faces'], dtype=np.int32) - 1
    else:
        vertices = np.array(surface.vertices, dtype=np.float32)
        faces = np.array(surface.faces, dtype=np.int32) - 1

    mesh = vedo.Mesh([vertices, faces])
    return vertices, faces, mesh


def load_curvature_data(filepath: Path, expected_length: int) -> np.ndarray:
    """Load mean curvature data and validate length."""
    curv_data = loadmat(str(filepath))
    curvature = np.array(curv_data['meanCurvature']).flatten()

    if len(curvature) != expected_length:
        raise ValueError(f"Curvature length ({len(curvature)}) doesn't match "
                        f"expected count ({expected_length})")

    return curvature


def load_curvature_data_raw(filepath: Path) -> np.ndarray:
    """Load raw (unsmoothed) mean curvature data."""
    curv_raw_data = loadmat(str(filepath))
    return np.array(curv_raw_data['meanCurvatureUnsmoothed']).flatten()


def load_gauss_data(filepath: Path) -> np.ndarray:
    """Load Gaussian curvature data."""
    gauss_data = loadmat(str(filepath))
    return np.array(gauss_data['gaussCurvatureUnsmoothed']).flatten()


def save_mesh_to_ply(mesh, filepath: Path) -> None:
    """Export mesh to PLY format."""
    mesh.export(str(filepath))


def save_results_to_json(results: dict, filepath: Path) -> None:
    """Save analysis results to JSON."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)