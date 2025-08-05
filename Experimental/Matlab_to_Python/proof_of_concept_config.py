#!/usr/bin/env python3
"""
Proof of Concept: Configuration-based MATLAB-Python Pipeline
This shows how we can manage the pipeline without tight integration
"""

import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import subprocess
import time

@dataclass
class SegmentationParams:
    """Parameters for three-level segmentation"""
    mesh_mode: str = 'threeLevelSurface'
    inside_gamma: float = 0.7
    inside_blur: List[float] = None
    filter_scales: List[float] = None
    smooth_mesh_mode: str = 'curvature'
    smooth_mesh_iterations: int = 10
    
    def __post_init__(self):
        if self.inside_blur is None:
            self.inside_blur = [5, 5, 2]
        if self.filter_scales is None:
            self.filter_scales = [1.5, 2, 4]

@dataclass 
class PipelineConfig:
    """Complete pipeline configuration"""
    image_directory: str
    save_directory: str
    pixel_size_xy: float  # nm
    pixel_size_z: float   # nm
    segmentation_params: SegmentationParams
    
    # Process control flags
    compute_mip: bool = True
    compute_mesh: bool = True
    run_python_analysis: bool = True
    
    def to_matlab_script(self) -> str:
        """Generate executable MATLAB script"""
        return f"""
% Auto-generated pipeline script
% Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

function run_pipeline()
    %% Set directories
    imageDirectory = '{self.image_directory}';
    saveDirectory = '{self.save_directory}';
    
    %% Set metadata
    pixelSizeXY = {self.pixel_size_xy};
    pixelSizeZ = {self.pixel_size_z}; 
    timeInterval = 1;
    
    %% Configure processes
    p.control.computeMIP = {int(self.compute_mip)};
    p.control.mesh = {int(self.compute_mesh)};
    p.control.surfaceSegment = 0;  % We don't use this
    
    %% Set channels
    cellSegChannel = 1;
    collagenChannel = 1;
    p = setChannels(p, cellSegChannel, collagenChannel);
    
    %% Configure mesh parameters
    p.mesh.meshMode = '{self.segmentation_params.mesh_mode}';
    p.mesh.insideGamma = {self.segmentation_params.inside_gamma};
    p.mesh.insideBlur = {self.segmentation_params.inside_blur};
    p.mesh.filterScales = {self.segmentation_params.filter_scales};
    p.mesh.smoothMeshMode = '{self.segmentation_params.smooth_mesh_mode}';
    p.mesh.smoothMeshIterations = {self.segmentation_params.smooth_mesh_iterations};
    
    %% Run analysis
    try
        MD = makeMovieDataOneChannel(imageDirectory, saveDirectory, ...
                                    pixelSizeXY, pixelSizeZ, timeInterval);
        morphology3D(MD, p);
        
        % Save success flag
        save(fullfile(saveDirectory, 'pipeline_complete.mat'), 'saveDirectory');
        disp('Pipeline completed successfully');
    catch ME
        % Save error information
        save(fullfile(saveDirectory, 'pipeline_error.mat'), 'ME');
        rethrow(ME);
    end
end

% Execute
run_pipeline();
"""

class SimplePipelineOrchestrator:
    """Lightweight orchestrator using file-based communication"""
    
    def __init__(self, matlab_executable: str = 'matlab'):
        self.matlab_executable = matlab_executable
        
    def run_pipeline(self, config: PipelineConfig):
        """Execute complete pipeline"""
        
        # Step 1: Validate configuration
        self._validate_config(config)
        
        # Step 2: Create MATLAB script
        script_path = Path(config.save_directory) / 'auto_pipeline.m'
        script_path.write_text(config.to_matlab_script())
        print(f"Generated MATLAB script: {script_path}")
        
        # Step 3: Run MATLAB
        if config.compute_mesh:
            print("Starting MATLAB processing...")
            self._run_matlab(script_path, config.save_directory)
        
        # Step 4: Run Python analysis
        if config.run_python_analysis:
            print("Starting Python analysis...")
            self._run_analysis(config)
            
        print("Pipeline complete!")
    
    def _validate_config(self, config: PipelineConfig):
        """Validate configuration parameters"""
        # Check paths exist
        if not Path(config.image_directory).exists():
            raise ValueError(f"Image directory not found: {config.image_directory}")
            
        # Create output directory
        Path(config.save_directory).mkdir(parents=True, exist_ok=True)
        
        # Validate parameters
        if config.pixel_size_xy <= 0 or config.pixel_size_z <= 0:
            raise ValueError("Pixel sizes must be positive")
            
        if not 0 < config.segmentation_params.inside_gamma <= 2:
            raise ValueError("inside_gamma should be between 0 and 2")
    
    def _run_matlab(self, script_path: Path, save_dir: str):
        """Execute MATLAB script"""
        matlab_cmd = [
            self.matlab_executable,
            '-batch',
            f"cd('{script_path.parent}'); {script_path.stem}"
        ]
        
        # Run and wait for completion
        result = subprocess.run(matlab_cmd, capture_output=True, text=True)
        
        # Check for success
        if (Path(save_dir) / 'pipeline_complete.mat').exists():
            print("MATLAB processing completed successfully")
        else:
            print("MATLAB processing failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            raise RuntimeError("MATLAB pipeline failed")
    
    def _run_analysis(self, config: PipelineConfig):
        """Run Python MeshAnalyzer"""
        from MeshAnalyzer import MeshAnalyzer
        
        # Standard u-shape3D output paths
        mesh_dir = Path(config.save_directory) / "Morphology" / "Analysis" / "Mesh" / "ch1"
        surface_path = mesh_dir / "surface_1_1.mat"
        curvature_path = mesh_dir / "meanCurvature_1_1.mat"
        
        if not surface_path.exists():
            print(f"Warning: Mesh output not found at {surface_path}")
            return
            
        # Run analysis
        analyzer = MeshAnalyzer(
            str(surface_path),
            str(curvature_path),
            pixel_size_xy=config.pixel_size_xy / 1000,  # Convert nm to μm
            pixel_size_z=config.pixel_size_z / 1000
        )
        
        analyzer.load_data()
        results = analyzer.calculate_statistics()
        
        # Save results
        results_path = Path(config.save_directory) / "python_analysis_results.txt"
        results_path.write_text(results.summary())
        print(f"Analysis results saved to: {results_path}")


# Example usage
if __name__ == "__main__":
    # Create configuration for immune cell
    config = PipelineConfig(
        image_directory="/path/to/images",
        save_directory="/path/to/results",
        pixel_size_xy=166.1,  # nm
        pixel_size_z=500,     # nm  
        segmentation_params=SegmentationParams(
            inside_gamma=0.6,  # Adjusted for immune cells
            filter_scales=[2, 3, 5]  # Larger scales
        )
    )
    
    # Save configuration for reproducibility
    config_path = Path(config.save_directory) / "pipeline_config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Run pipeline
    orchestrator = SimplePipelineOrchestrator()
    # orchestrator.run_pipeline(config)  # Uncomment to actually run
    
    print("Configuration-based pipeline ready!")
    print(f"Config saved to: {config_path}")
    print("\nThis approach provides:")
    print("- Parameter versioning (git-trackable)")
    print("- Batch processing capability") 
    print("- No complex integration required")
    print("- Easy debugging (can run MATLAB script manually)")