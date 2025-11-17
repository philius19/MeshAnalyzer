"""Utility functions for mesh analysis."""
import numpy as np
import vedo

from .datatypes import QualityMetrics


def convert_pixels_to_um(value: float, pixel_size: float) -> float:
    """Convert pixel units to micrometers."""
    return value * pixel_size


def calculate_mesh_quality_metrics(mesh: vedo.Mesh) -> QualityMetrics:
    """Calculate mesh quality metrics including edge lengths, face areas, and aspect ratios."""
    edges = mesh.edges
    edge_lengths = []
    for edge in edges:
        p1, p2 = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
        length = np.linalg.norm(p2 - p1)
        edge_lengths.append(length)

    edge_lengths = np.array(edge_lengths)

    face_areas = []
    for face in mesh.cells:
        vertices = mesh.vertices[face]
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        face_areas.append(area)

    face_areas = np.array(face_areas)

    aspect_ratios = []
    for face in mesh.cells:
        vertices = mesh.vertices[face]
        edges = [
            np.linalg.norm(vertices[1] - vertices[0]),
            np.linalg.norm(vertices[2] - vertices[1]),
            np.linalg.norm(vertices[0] - vertices[2])
        ]
        aspect_ratio = max(edges) / min(edges)
        aspect_ratios.append(aspect_ratio)

    aspect_ratios = np.array(aspect_ratios)

    return QualityMetrics(
        mean_edge_length=float(np.mean(edge_lengths)),
        std_edge_length=float(np.std(edge_lengths)),
        min_edge_length=float(np.min(edge_lengths)),
        max_edge_length=float(np.max(edge_lengths)),
        mean_face_area=float(np.mean(face_areas)),
        std_face_area=float(np.std(face_areas)),
        aspect_ratio_mean=float(np.mean(aspect_ratios)),
        aspect_ratio_std=float(np.std(aspect_ratios))
    )



def calculate_surface_roughness(curvature: np.ndarray) -> float:
    """Calculate surface roughness from curvature."""
    return float(np.std(np.abs(curvature)))


def find_high_curvature_regions(curvature: np.ndarray,
                               threshold: float = 2.0) -> np.ndarray:
    """Find indices of high curvature regions above threshold."""
    return np.abs(curvature) > threshold