"""
Internal data validation functions (not part of public API).

This module provides comprehensive validation for mesh data to ensure
data integrity and prevent silent failures.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_vertices(vertices: np.ndarray) -> None:
    """
    Validate vertex array for correct shape and finite values.

    Parameters:
        vertices: Nx3 array of vertex coordinates

    Raises:
        ValueError: If vertices have wrong shape or contain NaN/Inf values
    """
    if vertices.ndim != 2:
        raise ValueError(
            f"Vertices must be 2-dimensional array, got {vertices.ndim}D"
        )

    if vertices.shape[1] != 3:
        raise ValueError(
            f"Vertices must be Nx3 (3D coordinates), got shape {vertices.shape}"
        )

    if vertices.shape[0] == 0:
        raise ValueError("Vertices array is empty")

    # Check for NaN/Inf values
    if not np.isfinite(vertices).all():
        n_bad = (~np.isfinite(vertices)).sum()
        raise ValueError(
            f"Vertices contain {n_bad} NaN/Inf values. "
            f"Check input data quality."
        )


def validate_faces(faces: np.ndarray, n_vertices: int) -> None:
    """
    Validate face indices for correct shape, range, and consistency.

    Parameters:
        faces: Mx3 array of face indices (0-indexed)
        n_vertices: Number of vertices in mesh

    Raises:
        ValueError: If faces have wrong shape, out-of-bound indices, or negative values
    """
    if faces.ndim != 2:
        raise ValueError(
            f"Faces must be 2-dimensional array, got {faces.ndim}D"
        )

    if faces.shape[1] != 3:
        raise ValueError(
            f"Faces must be Mx3 (triangular faces), got shape {faces.shape}"
        )

    if faces.shape[0] == 0:
        raise ValueError("Faces array is empty")

    # Check for negative indices (indicates MATLAB conversion error)
    if faces.min() < 0:
        raise ValueError(
            f"Faces contain negative index: {faces.min()}. "
            f"This indicates a problem with MATLAB 1-to-0 index conversion."
        )

    # Check for out-of-bounds indices
    if faces.max() >= n_vertices:
        raise ValueError(
            f"Face index {faces.max()} exceeds vertex count {n_vertices}. "
            f"Valid range: [0, {n_vertices-1}]. "
            f"Check that surface and face data are from the same mesh."
        )

    # Check for very large indices that might indicate int32 overflow
    if faces.max() > np.iinfo(np.int32).max:
        logger.warning(
            f"Face indices exceed int32 range (max={faces.max()}). "
            f"Mesh has {n_vertices} vertices. "
            f"Using int64 representation."
        )


def validate_curvature(curvature: np.ndarray, n_faces: int) -> None:
    """
    Validate curvature data for correct length and finite values.

    Parameters:
        curvature: Array of curvature values (one per face)
        n_faces: Expected number of faces

    Raises:
        ValueError: If curvature has wrong length or contains NaN/Inf values
    """
    if curvature.ndim != 1:
        raise ValueError(
            f"Curvature must be 1-dimensional array, got {curvature.ndim}D with shape {curvature.shape}"
        )

    if len(curvature) != n_faces:
        raise ValueError(
            f"Curvature length ({len(curvature)}) doesn't match "
            f"face count ({n_faces}). "
            f"Ensure curvature and surface files are from the same analysis."
        )

    # Check for NaN/Inf values
    if not np.isfinite(curvature).all():
        n_nan = np.isnan(curvature).sum()
        n_inf = np.isinf(curvature).sum()
        raise ValueError(
            f"Curvature contains {n_nan} NaN and {n_inf} Inf values. "
            f"Check curvature calculation quality."
        )


def validate_pixel_sizes(pixel_size_xy: float, pixel_size_z: float) -> None:
    """
    Validate pixel size parameters.

    Parameters:
        pixel_size_xy: XY pixel size in micrometers
        pixel_size_z: Z pixel size in micrometers

    Raises:
        ValueError: If pixel sizes are not positive finite numbers
    """
    if not np.isfinite(pixel_size_xy) or pixel_size_xy <= 0:
        raise ValueError(
            f"pixel_size_xy must be positive, got {pixel_size_xy}"
        )

    if not np.isfinite(pixel_size_z) or pixel_size_z <= 0:
        raise ValueError(
            f"pixel_size_z must be positive, got {pixel_size_z}"
        )

    # Warn about unusual pixel sizes (likely user error)
    if pixel_size_xy < 0.01 or pixel_size_xy > 10:
        logger.warning(
            f"Unusual pixel_size_xy: {pixel_size_xy} μm. "
        )

    if pixel_size_z < 0.01 or pixel_size_z > 10:
        logger.warning(
            f"Unusual pixel_size_z: {pixel_size_z} μm. "
        )


def validate_face_indices_safe_for_int32(faces: np.ndarray) -> bool:
    """
    Check if face indices can be safely represented as int32.

    Parameters:
        faces: Face index array

    Returns:
        True if safe for int32, False otherwise
    """
    return faces.max() <= np.iinfo(np.int32).max
