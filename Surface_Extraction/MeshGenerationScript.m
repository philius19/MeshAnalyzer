function MeshGenerationScript_BioFormats()

% Mesh Generation Script - BioFormats Integration
%
% This script uses u-shape3D's object-oriented architecture with support for
% both simple TIFF series and BioFormats import.
%
% KEY FEATURES:
%   - Toggle between TIFF and BioFormats import (useBioFormats flag)
%   - Uses Process framework correctly
%   - Validates parameters through u-shape3D's system
%   - Compatible with Lattice Light Sheet microscopy
%
% CONTROL PARAMETERS:
%   - p.control.computeMIP: Generate Maximum Intensity Projections
%   - p.control.mesh: Generate 3D surface mesh using threeLevelSegmentation
%   - p.control.meshThres: Create threshold-based segmentation mask
%
% If there are Problems with Tiff-Reader, its caused due to hidden files.
% Fix: find /Volumes/T7/Images_Tim/Segment -name "._*" -delete
%
% Author: Philipp Kaintoch
% Date: 2025-12-07, Version 2.0

%% Set directories
% imageDirectory = '/Volumes/T7/Spinning Disc/2025_11_11_Neutro/Decon/01';
saveDirectory = '/Volumes/T9/LLSM/2025-12-08_09-05-39/251104_BMMC_1_raws_decon/06_Surface_Rendering/02/Batch_3';

%% Set Image Metadata
pixelSizeXY = 103;   % nm
pixelSizeZ = 216;    % nm
timeInterval = 2;

%% Import Configuration
useBioFormats = true;  % Toggle: false = TIFF directory, true = BioFormats file

% For BioFormats mode
imageFile = '/Volumes/T9/LLSM/2025-12-08_09-05-39/251104_BMMC_1_raws_decon/03_Decon/251104_BMMC_mTdt_2_2s_interval__CamA_ch0_decon.tif';
importMetadata = true;

%% Turn processes on and off
p.control.resetMD = 0;
p.control.deconvolution = 0;         p.control.deconvolutionReset = 0;
p.control.computeMIP = 1;            p.control.computeMIPReset = 0;
p.control.mesh = 1;                  p.control.meshReset = 0;
p.control.meshThres = 0;             p.control.meshThresReset = 0;
p.control.surfaceSegment = 0;        p.control.surfaceSegmentReset = 0;
p.control.patchDescribeForMerge = 0; p.control.patchDescribeForMergeReset = 0;
p.control.patchMerge = 0;            p.control.patchMergeReset = 0;
p.control.patchDescribe = 0;         p.control.patchDescribeReset = 0;
p.control.motifDetect = 0;           p.control.motifDetectReset = 0;
p.control.meshMotion = 0;            p.control.meshMotionReset = 0;
p.control.intensity = 0;             p.control.intensityReset = 0;
p.control.intensityBlebCompare = 0; p.control.intensityBlebCompareReset = 0;

cellSegChannel = 1;
collagenChannel = 1;
p = setChannels(p, cellSegChannel, collagenChannel);

addpath('/Users/philippkaintoch/Documents/Projects/02_Codebase/Pipeline/Scripts/Surface_Extraction') % Experimental for integrating Silicon-Compatiblity Smoothing  

%% Override Default Parameters
% ALL 18 PARAMETERS with defaults from Mesh3DProcess.getDefaultParams()

% Basic mesh generation mode
p.mesh.meshMode = 'threeLevelSurface';              % Default: 'otsu'
p.mesh.useUndeconvolved = 1;                        % Default: 0

% Three-level segmentation parameters
p.mesh.insideGamma = 0.7;                           % Default: 0.6
p.mesh.insideBlur = 2;                              % Default: 2
p.mesh.filterScales = [1.5, 2, 3];                  % Default: [1.5, 2, 4]
p.mesh.filterNumStdSurface = 2;                   % Default: 2
p.mesh.insideDilateRadius = 5;                      % Default: 5
p.mesh.insideErodeRadius = 6.5;                     % Default: 6.5

% Image preprocessing
p.mesh.imageGamma = 1;                              % Default: 1
p.mesh.scaleOtsu = 1;                               % Default: 1
p.mesh.smoothImageSize = 0;                         % Default: 0

% Mesh smoothing
p.mesh.smoothMeshMode = 'none';                     % Default: 'curvature' | Options: 'curvature', 'none', 'appleSilicon'
p.mesh.smoothMeshIterations = 6;                    % Default: 6

% Curvature computation parameters
p.mesh.curvatureMedianFilterRadius = 2;             % Default: 2
p.mesh.curvatureSmoothOnMeshIterations = 20;        % Default: 20

% Image registration (not used in typical workflow)
p.mesh.registerImages = 0;                          % Default: 0
p.mesh.saveRawImages = 0;                           % Default: 0
p.mesh.registrationMode = 'translation';            % Default: 'translation' | Options: 'translation', 'rigid', 'affine'

%% Run the analysis

% load the movie
if ~isfolder(saveDirectory), mkdir(saveDirectory); end

if useBioFormats
    MD = MovieData(imageFile, importMetadata, 'outputDirectory', saveDirectory);
    if MD.pixelSize_ == 1000
        MD.pixelSize_ = pixelSizeXY;
        MD.pixelSizeZ_ = pixelSizeZ;
        MD.timeInterval_ = timeInterval;
        MD.save;
    end
else
    MD = makeMovieDataOneChannel(imageDirectory, saveDirectory, pixelSizeXY, pixelSizeZ, timeInterval);
end

% analyze the cell
morphology3D(MD, p)

% make figures
plotMeshMD(MD, 'surfaceMode', 'curvature'); title('Curvature');

% Export comprehensive metadata for Python analysis
try
    % Prepare metadata export options with script-level information
    metadataOpts = struct();
    metadataOpts.sourceImageFile = imageFile;
    metadataOpts.pixelSizeXY = pixelSizeXY;
    metadataOpts.pixelSizeZ = pixelSizeZ;
    metadataOpts.timeInterval = timeInterval;
    metadataOpts.meshParams = p.mesh;
    metadataOpts.verbose = true;

    exportSuccess = exportMetadata(MD, saveDirectory, metadataOpts);
    if exportSuccess
        fprintf('\n=== Comprehensive Metadata Export Complete ===\n');
        fprintf('Python-readable metadata saved to:\n');
        fprintf('  %s/metadata_export.mat\n', saveDirectory);
        fprintf('Contains: acquisition params, processing params, provenance\n\n');
    end
catch ME
    warning('Metadata export failed: %s', ME.message);
    fprintf('Continuing with existing workflow...\n');
end
