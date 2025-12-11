"""File input/output operations for mesh analysis."""
from pathlib import Path
from typing import Tuple, List, Optional
import json

import numpy as np
import scipy.io as sio
import vedo
from mat73 import loadmat as loadmat_v73

from .datatypes import (
    MeshParameters,
    ProcessingMetadata,
    AuxiliaryMeshData,
    MeshFrame,
    DEFAULT_MESH_PARAMETERS,
    DEFAULT_PIXEL_SIZE_XY_NM,
    DEFAULT_PIXEL_SIZE_Z_NM,
    DEFAULT_PIXEL_SIZE_XY_UM,
    DEFAULT_PIXEL_SIZE_Z_UM
)
from ._validation import (
    validate_vertices,
    validate_faces,
    validate_curvature,
    validate_face_indices_safe_for_int32
)

import logging
logger = logging.getLogger(__name__)
import re


def loadmat(filepath: str) -> dict:
    """Load MATLAB file, automatically handling both v7.3 (HDF5) and older formats."""
    try:
        return loadmat_v73(filepath)
    except (TypeError, OSError):
        return sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)


def extract_nested_field(matlab_obj, path: str, default=None):
    """
    Extract field from nested MATLAB structure.

    Handles cell arrays, dict vs object attribute access, and missing fields.

    Parameters:
        matlab_obj: MATLAB structure (dict or object)
        path: Dot-separated field path (e.g., 'processes_.0.funParams_.insideGamma')
        default: Value to return if field not found

    Returns:
        Field value or default if not found
    """
    parts = path.split('.')
    current = matlab_obj

    for part in parts:
        if part.endswith('_'):
            field_name = part[:-1]
            if hasattr(current, field_name):
                current = getattr(current, field_name)
            elif isinstance(current, dict) and field_name in current:
                current = current[field_name]
            else:
                return default
        elif part.isdigit():
            idx = int(part)
            if isinstance(current, (list, tuple)) and len(current) > idx:
                current = current[idx]
            else:
                return default
        else:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

    return current


def safe_scalar(arr):
    """Extract scalar from MATLAB array (handles both scipy.io and mat73 formats)."""
    if arr is None:
        return None
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 0:
        raise ValueError(f"Expected scalar, got shape {arr.shape}")
    return float(arr)


def load_moviedata_metadata(moviedata_path: Path, channel: int = 1) -> ProcessingMetadata:
    """
    Extract processing metadata from MovieData .mat file or metadata_export.mat.

    PRIORITY ORDER:
    1. Try metadata_export.mat (Python-friendly struct from MATLAB)
    2. Fall back to MovieData object (may fail with MatlabOpaque)
    3. Fall back to default parameters with warning

    Parameters:
        moviedata_path: Path to MovieData .mat file
        channel: Channel number

    Returns:
        ProcessingMetadata with all parameters and provenance
    """
    # === PRIORITY 1: Try metadata_export.mat ===
    cell_dir = moviedata_path.parent
    metadata_export_path = cell_dir / 'metadata_export.mat'

    if metadata_export_path.exists():
        try:
            export_data = loadmat(str(metadata_export_path))
            metadata = export_data['metadata']

            # Extract Tier 1 (Essential Acquisition)
            pixel_size_xy = safe_scalar(metadata['pixel_size_xy_nm'])
            pixel_size_z = safe_scalar(metadata['pixel_size_z_nm'])
            time_interval = safe_scalar(metadata['time_interval_sec'])

            # Extract Tier 2 (Processing Parameters)
            # Handle both dict (mat73) and structured array (scipy.io)
            has_mesh_params = ('mesh_parameters' in metadata if isinstance(metadata, dict)
                              else 'mesh_parameters' in metadata.dtype.names)

            if has_mesh_params:
                # Handle both dict (mat73) and structured array (scipy.io)
                mp = (metadata['mesh_parameters'] if isinstance(metadata, dict)
                     else metadata['mesh_parameters'][0, 0])

                # Handle inside_blur (could be scalar or array)
                if isinstance(metadata, dict):
                    inside_blur_raw = mp['inside_blur']
                    if isinstance(inside_blur_raw, (list, np.ndarray)) and len(np.asarray(inside_blur_raw).flatten()) > 1:
                        inside_blur = np.asarray(inside_blur_raw).flatten().tolist()
                    else:
                        inside_blur = float(np.asarray(inside_blur_raw))
                else:
                    inside_blur_raw = mp['inside_blur'][0, 0]
                    if inside_blur_raw.size > 1:
                        inside_blur = inside_blur_raw.flatten().tolist()
                    else:
                        inside_blur = float(inside_blur_raw)

                # Extract string fields (handle both formats)
                def get_str(field):
                    val = mp[field]
                    if isinstance(val, str):
                        return val
                    elif isinstance(val, list) and len(val) > 0:
                        return val[0]
                    else:
                        return str(val[0]) if isinstance(val, np.ndarray) else str(val)

                # Extract array fields (handle both formats)
                def get_array(field):
                    val = mp[field]
                    if isinstance(val, list):
                        # Unwrap nested lists and convert numpy arrays
                        result = val[0] if isinstance(val[0], (list, np.ndarray)) else val
                        if isinstance(result, np.ndarray):
                            return result.flatten().tolist()
                        return result
                    else:
                        return val[0, 0].flatten().tolist() if hasattr(val, 'flatten') else val

                mesh_params = MeshParameters(
                    mesh_mode=get_str('mesh_mode'),
                    inside_gamma=safe_scalar(mp['inside_gamma']),
                    inside_blur=inside_blur,
                    filter_scales=get_array('filter_scales'),
                    filter_num_std_surface=safe_scalar(mp['filter_num_std_surface']),
                    inside_dilate_radius=safe_scalar(mp['inside_dilate_radius']),
                    inside_erode_radius=safe_scalar(mp['inside_erode_radius']),
                    smooth_mesh_mode=get_str('smooth_mesh_mode'),
                    smooth_mesh_iterations=int(safe_scalar(mp['smooth_mesh_iterations'])),
                    use_undeconvolved=bool(safe_scalar(mp['use_undeconvolved'])),
                    image_gamma=safe_scalar(mp['image_gamma']),
                    scale_otsu=safe_scalar(mp['scale_otsu']),
                    smooth_image_size=safe_scalar(mp['smooth_image_size']),
                    curvature_median_filter_radius=int(safe_scalar(mp['curvature_median_filter_radius'])),
                    curvature_smooth_on_mesh_iterations=int(safe_scalar(mp['curvature_smooth_on_mesh_iterations'])),
                    register_images=bool(safe_scalar(mp['register_images'])),
                    save_raw_images=bool(safe_scalar(mp['save_raw_images'])),
                    registration_mode=get_str('registration_mode')
                )
            else:
                mesh_params = DEFAULT_MESH_PARAMETERS

            # Extract Tier 4 (Provenance)
            source_path = str(metadata['source_image_path'][0] if not isinstance(metadata, dict)
                             else metadata['source_image_path'])
            processing_date = str(metadata['processing_date'][0] if not isinstance(metadata, dict)
                                 else metadata['processing_date'])
            matlab_version = str(metadata['matlab_version'][0] if not isinstance(metadata, dict)
                                else metadata['matlab_version'])

            logger.info(f"✓ Loaded metadata from metadata_export.mat")

            return ProcessingMetadata(
                pixel_size_xy_nm=pixel_size_xy,
                pixel_size_z_nm=pixel_size_z,
                time_interval_sec=time_interval,
                source_image_path=source_path,
                source_image_name=Path(source_path).name if source_path else '',
                processing_date=processing_date if processing_date else None,
                matlab_version=matlab_version if matlab_version else None,
                mesh_parameters=mesh_params
            )
        except Exception as e:
            logger.warning(f"Failed to load metadata_export.mat: {e}. Trying MovieData object...")

    # === PRIORITY 2: Try MovieData object (existing implementation) ===
    md_data = loadmat(str(moviedata_path))

    MD = md_data.get('MD', None)
    if MD is None:
        logger.warning(
            f"MovieData file {moviedata_path} does not contain 'MD' key. "
            f"Using default parameters. This may happen if MovieData wasn't saved properly."
        )
        return ProcessingMetadata(
            pixel_size_xy_nm=DEFAULT_PIXEL_SIZE_XY_NM,
            pixel_size_z_nm=DEFAULT_PIXEL_SIZE_Z_NM,
            time_interval_sec=1.0,
            source_image_path='',
            source_image_name='',
            processing_date=None,
            matlab_version=None,
            mesh_parameters=DEFAULT_MESH_PARAMETERS
        )

    pixel_size_xy = extract_nested_field(MD, 'pixelSize_', default=DEFAULT_PIXEL_SIZE_XY_NM)
    pixel_size_z = extract_nested_field(MD, 'pixelSizeZ_', default=DEFAULT_PIXEL_SIZE_Z_NM)
    time_interval = extract_nested_field(MD, 'timeInterval_', default=1.0)

    mesh_process = None
    processes = extract_nested_field(MD, 'processes_', default=[])
    for proc in processes:
        if proc is not None:
            proc_name = extract_nested_field(proc, 'name_', default='')
            if 'Mesh3DProcess' in proc_name:
                mesh_process = proc
                break

    if mesh_process:
        params = extract_nested_field(mesh_process, 'funParams_', default={})
        mesh_params = MeshParameters(
            mesh_mode=extract_nested_field(params, 'meshMode', DEFAULT_MESH_PARAMETERS.mesh_mode),
            inside_gamma=extract_nested_field(params, 'insideGamma', DEFAULT_MESH_PARAMETERS.inside_gamma),
            inside_blur=extract_nested_field(params, 'insideBlur', DEFAULT_MESH_PARAMETERS.inside_blur),
            filter_scales=extract_nested_field(params, 'filterScales', DEFAULT_MESH_PARAMETERS.filter_scales),
            filter_num_std_surface=extract_nested_field(params, 'filterNumStdSurface', DEFAULT_MESH_PARAMETERS.filter_num_std_surface),
            inside_dilate_radius=extract_nested_field(params, 'insideDilateRadius', DEFAULT_MESH_PARAMETERS.inside_dilate_radius),
            inside_erode_radius=extract_nested_field(params, 'insideErodeRadius', DEFAULT_MESH_PARAMETERS.inside_erode_radius),
            smooth_mesh_mode=extract_nested_field(params, 'smoothMeshMode', DEFAULT_MESH_PARAMETERS.smooth_mesh_mode),
            smooth_mesh_iterations=extract_nested_field(params, 'smoothMeshIterations', DEFAULT_MESH_PARAMETERS.smooth_mesh_iterations),
            use_undeconvolved=extract_nested_field(params, 'useUndeconvolved', DEFAULT_MESH_PARAMETERS.use_undeconvolved),
            image_gamma=extract_nested_field(params, 'imageGamma', DEFAULT_MESH_PARAMETERS.image_gamma),
            scale_otsu=extract_nested_field(params, 'scaleOtsu', DEFAULT_MESH_PARAMETERS.scale_otsu),
            smooth_image_size=extract_nested_field(params, 'smoothImageSize', DEFAULT_MESH_PARAMETERS.smooth_image_size),
            curvature_median_filter_radius=extract_nested_field(params, 'curvatureMedianFilterRadius', DEFAULT_MESH_PARAMETERS.curvature_median_filter_radius),
            curvature_smooth_on_mesh_iterations=extract_nested_field(params, 'curvatureSmoothOnMeshIterations', DEFAULT_MESH_PARAMETERS.curvature_smooth_on_mesh_iterations),
            register_images=extract_nested_field(params, 'registerImages', DEFAULT_MESH_PARAMETERS.register_images),
            save_raw_images=extract_nested_field(params, 'saveRawImages', DEFAULT_MESH_PARAMETERS.save_raw_images),
            registration_mode=extract_nested_field(params, 'registrationMode', DEFAULT_MESH_PARAMETERS.registration_mode)
        )
    else:
        logger.warning(f"No Mesh3DProcess found in MovieData. Using default parameters.")
        mesh_params = DEFAULT_MESH_PARAMETERS

    channels = extract_nested_field(MD, 'channels_', default=[])
    source_path = ''
    if len(channels) >= channel:
        channel_obj = channels[channel - 1]
        source_path = extract_nested_field(channel_obj, 'channelPath_', default='')

    return ProcessingMetadata(
        pixel_size_xy_nm=pixel_size_xy,
        pixel_size_z_nm=pixel_size_z,
        time_interval_sec=time_interval,
        source_image_path=source_path,
        source_image_name=Path(source_path).name if source_path else '',
        processing_date=None,
        matlab_version=None,
        mesh_parameters=mesh_params
    )


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
    """
    Load surface mesh data from .mat file with comprehensive validation.

    Handles both MATLAB v7.3 (HDF5) and older formats. Converts MATLAB's
    1-indexed faces to Python's 0-indexed. Validates data integrity.

    Parameters:
        filepath: Path to surface .mat file

    Returns:
        Tuple of (vertices, faces, vedo_mesh)
            - vertices: Nx3 float32 array of vertex coordinates
            - faces: Mx3 int64 array of face indices (0-indexed)
            - vedo_mesh: vedo.Mesh object

    Raises:
        ValueError: If data is invalid or corrupted
    """
    surface_data = loadmat(str(filepath))
    surface = surface_data['surface']

    # Extract vertices and faces from either dict or object format
    if isinstance(surface, dict):
        vertices_raw = np.array(surface['vertices'], dtype=np.float32)
        faces_matlab = np.array(surface['faces'], dtype=np.int64)
    else:
        vertices_raw = np.array(surface.vertices, dtype=np.float32)
        faces_matlab = np.array(surface.faces, dtype=np.int64)

    # Validate vertices before processing
    validate_vertices(vertices_raw)

    # Convert MATLAB 1-indexed to Python 0-indexed
    # Do this BEFORE any potential int32 cast to avoid overflow
    faces_python = faces_matlab - 1

    # Validate faces with vertex count
    validate_faces(faces_python, len(vertices_raw))

    # Check if safe for int32 (for memory efficiency with vedo)
    # Most meshes fit in int32, but we keep int64 for safety
    if not validate_face_indices_safe_for_int32(faces_python):
        # Keep as int64 for very large meshes
        faces = faces_python
    else:
        # Safe to cast to int32 for memory efficiency
        faces = faces_python.astype(np.int32)

    vertices = vertices_raw
    mesh = vedo.Mesh([vertices, faces])
    return vertices, faces, mesh


def load_curvature_data(filepath: Path, expected_length: int) -> np.ndarray:
    """
    Load mean curvature data with comprehensive validation.

    Parameters:
        filepath: Path to curvature .mat file
        expected_length: Expected number of faces (for validation)

    Returns:
        1D array of curvature values (one per face)

    Raises:
        ValueError: If curvature data is invalid or length mismatch
    """
    curv_data = loadmat(str(filepath))
    curvature = np.array(curv_data['meanCurvature']).flatten()

    # Comprehensive validation
    validate_curvature(curvature, expected_length)

    return curvature


def load_curvature_data_raw(filepath: Path) -> np.ndarray:
    """Load raw (unsmoothed) mean curvature data."""
    curv_raw_data = loadmat(str(filepath))
    return np.array(curv_raw_data['meanCurvatureUnsmoothed']).flatten()


def load_gauss_data(filepath: Path) -> np.ndarray:
    """Load Gaussian curvature data."""
    gauss_data = loadmat(str(filepath))
    return np.array(gauss_data['gaussCurvatureUnsmoothed']).flatten()


def load_face_normals(filepath: Path) -> np.ndarray:
    """Load face normal vectors."""
    data = loadmat(str(filepath))
    return np.array(data['faceNormals'], dtype=np.float32)


def load_neighbors(filepath: Path) -> np.ndarray:
    """Load face neighbor adjacency matrix."""
    data = loadmat(str(filepath))
    return np.array(data['neighbors'])


def load_image_surface(filepath: Path) -> np.ndarray:
    """Load binary segmentation mask."""
    data = loadmat(str(filepath))
    return np.array(data['imageSurface'])


def load_auxiliary_data(mesh_dir: Path, channel: int, time_index: int) -> AuxiliaryMeshData:
    """
    Load all auxiliary data for a single timepoint.

    Parameters:
        mesh_dir: Path to Morphology/Analysis/Mesh/ch{channel}/
        channel: Channel number
        time_index: Time index

    Returns:
        AuxiliaryMeshData with all optional fields
    """
    base_pattern = f"{channel}_{time_index}"

    def safe_load(filename, loader_func):
        filepath = mesh_dir / filename
        if filepath.exists():
            try:
                return loader_func(filepath)
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")
                return None
        return None

    return AuxiliaryMeshData(
        face_normals=safe_load(f'faceNormals_{base_pattern}.mat', load_face_normals),
        gaussian_curvature=safe_load(f'gaussCurvatureUnsmoothed_{base_pattern}.mat', load_gauss_data),
        mean_curvature_raw=safe_load(f'meanCurvatureUnsmoothed_{base_pattern}.mat', load_curvature_data_raw),
        neighbors=safe_load(f'neighbors_{base_pattern}.mat', load_neighbors),
        image_surface=safe_load(f'imageSurface_{base_pattern}.mat', load_image_surface)
    )


def load_mesh_frame(
    surface_path: Path,
    curvature_path: Path,
    pixel_size_xy: float = DEFAULT_PIXEL_SIZE_XY_UM,
    pixel_size_z: float = DEFAULT_PIXEL_SIZE_Z_UM,
    metadata: Optional[ProcessingMetadata] = None,
    load_auxiliary: bool = False
) -> MeshFrame:
    """
    Load complete mesh frame data from MATLAB files.

    Unified loader that replaces manual file loading and MeshAnalyzer initialization.
    Automatically handles mesh validation and normal correction.

    Parameters:
        surface_path: Path to surface_*.mat file
        curvature_path: Path to meanCurvature_*.mat file
        pixel_size_xy: XY pixel size in micrometers (default from constants)
        pixel_size_z: Z pixel size in micrometers (default from constants)
        metadata: Optional ProcessingMetadata (auto-loaded if available)
        load_auxiliary: Load auxiliary data (normals, Gaussian curvature, etc.)

    Returns:
        MeshFrame with all data loaded and validated

    Example:
        frame = load_mesh_frame(
            Path('surface_1_1.mat'),
            Path('meanCurvature_1_1.mat'),
            pixel_size_xy=0.103,
            pixel_size_z=0.217
        )
    """
    # Load geometry
    surface_path = Path(surface_path)
    curvature_path = Path(curvature_path)
    vertices, faces, mesh = load_surface_data(surface_path)
    curvature = load_curvature_data(curvature_path, len(faces))

    # Auto-correct inverted normals
    if mesh.volume() < 0:
        logger.debug("Negative mesh volume detected, reversing normals")
        mesh = mesh.clone().reverse()

    # Extract time index from filename (e.g., surface_1_42.mat -> time_index=42)
    match = re.search(r'_(\d+)_(\d+)\.mat$', surface_path.name)
    time_index = int(match.group(2)) if match else 0

    # Load auxiliary data if requested
    auxiliary = None
    if load_auxiliary:
        match_ch_time = re.search(r'_(\d+)_(\d+)\.mat$', surface_path.name)
        if match_ch_time:
            channel = int(match_ch_time.group(1))
            time_idx = int(match_ch_time.group(2))
            auxiliary = load_auxiliary_data(surface_path.parent, channel, time_idx)

    # Create metadata if not provided (use defaults)
    if metadata is None:
        metadata = ProcessingMetadata(
            pixel_size_xy_nm=pixel_size_xy * 1000.0,
            pixel_size_z_nm=pixel_size_z * 1000.0,
            time_interval_sec=1.0,
            source_image_path='',
            source_image_name='',
            processing_date=None,
            matlab_version=None,
            mesh_parameters=DEFAULT_MESH_PARAMETERS
        )

    return MeshFrame(
        vertices=vertices,
        faces=faces,
        curvature=curvature,
        mesh=mesh,
        time_index=time_index,
        metadata=metadata,
        auxiliary=auxiliary
    )


def save_mesh_to_ply(mesh, filepath: Path) -> None:
    """Export mesh to PLY format."""
    mesh.export(str(filepath))


def save_results_to_json(results: dict, filepath: Path) -> None:
    """Save analysis results to JSON."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)