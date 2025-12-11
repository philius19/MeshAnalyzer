"""Test metadata export and loading integration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'Pipeline' / 'Scripts'))

from MeshAnalyzer import TimeSeriesManager


def test_metadata_loading():
    """Test that all 18 parameters are loaded successfully."""
    cell_dir = Path('/Volumes/T9/LLSM/2025-11-12_10-03-22/06_Surface_Rendering/01')

    if not cell_dir.exists():
        print(f"⚠️  Test data not found: {cell_dir}")
        return False

    print(f"Loading data from: {cell_dir}")
    manager = TimeSeriesManager.from_cell_directory(cell_dir, channel=1)

    print(f"\n✓ TimeSeriesManager loaded successfully")
    print(f"  Frames: {len(manager)}")
    print(f"  Metadata available: {manager.metadata is not None}")

    if manager.metadata is None:
        print("✗ No metadata found!")
        return False

    print(f"\n=== Pixel Sizes (Bug #1 test) ===")
    print(f"  XY: {manager.metadata.pixel_size_xy_nm:.2f} nm")
    print(f"  Z:  {manager.metadata.pixel_size_z_nm:.2f} nm")

    print(f"\n=== Mesh Parameters (18 total) ===")
    params = manager.metadata.mesh_parameters

    param_list = [
        ('mesh_mode', params.mesh_mode),
        ('inside_gamma', params.inside_gamma),
        ('inside_blur', params.inside_blur),
        ('filter_scales', params.filter_scales),
        ('filter_num_std_surface', params.filter_num_std_surface),
        ('inside_dilate_radius', params.inside_dilate_radius),
        ('inside_erode_radius', params.inside_erode_radius),
        ('smooth_mesh_mode', params.smooth_mesh_mode),
        ('smooth_mesh_iterations', params.smooth_mesh_iterations),
        ('use_undeconvolved', params.use_undeconvolved),
        ('image_gamma', params.image_gamma),
        ('scale_otsu', params.scale_otsu),
        ('smooth_image_size', params.smooth_image_size),
        ('curvature_median_filter_radius', params.curvature_median_filter_radius),
        ('curvature_smooth_on_mesh_iterations', params.curvature_smooth_on_mesh_iterations),
        ('register_images', params.register_images),
        ('save_raw_images', params.save_raw_images),
        ('registration_mode', params.registration_mode),
    ]

    all_present = True
    for i, (name, value) in enumerate(param_list, 1):
        if value is None:
            print(f"  {i:2d}. ✗ {name}: MISSING")
            all_present = False
        else:
            print(f"  {i:2d}. ✓ {name}: {value}")

    print(f"\n=== Property Access Test ===")
    if len(manager) > 0:
        first_frame = manager[list(manager.time_indices)[0]]
        print(f"  processing_metadata: {first_frame.processing_metadata is not None}")
        print(f"  mesh_parameters: {first_frame.mesh_parameters is not None}")

        if first_frame.mesh_parameters:
            print(f"  mesh_mode via property: {first_frame.mesh_parameters.mesh_mode}")
            print(f"  inside_gamma via property: {first_frame.mesh_parameters.inside_gamma}")
    else:
        print(f"  No frames loaded - skipping property test")

    print(f"\n{'='*50}")
    if all_present:
        print("✓ SUCCESS: All 18 parameters loaded successfully")
        return True
    else:
        print("✗ FAILURE: Some parameters missing")
        return False


if __name__ == '__main__':
    success = test_metadata_loading()
    sys.exit(0 if success else 1)
