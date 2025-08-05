function MeshGenerationScript()

% Mesh Generation Script - Proper u-shape3D Architecture
% 
% This script properly uses u-shape3D's object-oriented architecture while
% maintaining a simple, straightforward coding style.
%
% Key improvements:
%   - Uses Process framework correctly
%   - Validates parameters through u-shape3D's system
%   - Maintains simple structure for easy understanding
%   - Full parameter access with proper validation
%
% If there are Problems with Tiff-Reader, its caused due to hidden files.
% Fix: find /Volumes/T7/Images_Tim/Segment -name "._*" -delete
% 
% Author: Philipp Kaintoch 
% Date: 2025-08-05, Version 1.0 

%% Set directories
imageDirectory = '/Volumes/T7/Images_Tim/Segment';
saveDirectory = '/Users/philippkaintoch/Desktop/Results';

%% Set Image Metadata 
pixelSizeXY = 166.1; % nm
pixelSizeZ = 500;    % nm
timeInterval = 1; 

%% Turn processes on and off
p.control.resetMD = 0; 
p.control.deconvolution = 0;         p.control.deconvolutionReset = 0;
p.control.computeMIP = 1;            p.control.computeMIPReset = 0;
p.control.mesh = 1;                  p.control.meshReset = 0;
p.control.meshThres = 1;             p.control.meshThresReset = 0;
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

%% Override Default Parameters
p.mesh.meshMode = 'threeLevelSurface';
p.mesh.useUndeconvolved = 1;

% Three-level segmentation parameters
p.mesh.insideGamma = 0.7;
p.mesh.insideBlur = [5, 5, 2];
p.mesh.filterScales = [1.5, 2, 4];
p.mesh.filterNumStdSurface = 1.5;
p.mesh.insideDilateRadius = 3;
p.mesh.insideErodeRadius = 2;

% Image preprocessing
p.mesh.imageGamma = 1;
p.mesh.scaleOtsu = 1;
p.mesh.smoothImageSize = 0;

% Mesh smoothing
p.mesh.smoothMeshMode = 'none';
p.mesh.smoothMeshIterations = 10;

%% Run the analysis

% load the movie
if ~isfolder(saveDirectory), mkdir(saveDirectory); end
MD = makeMovieDataOneChannel(imageDirectory, saveDirectory, pixelSizeXY, pixelSizeZ, timeInterval);

% analyze the cell
morphology3D(MD, p)

% make figures
plotMeshMD(MD, 'surfaceMode', 'curvature'); title('Curvature');