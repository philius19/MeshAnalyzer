function success = exportMetadata(MD, outputDir, options)
% EXPORTMETADATA Export comprehensive metadata in Python-friendly format
%
% Creates a single comprehensive metadata file containing all acquisition
% parameters, processing settings, and provenance information needed for
% Python analysis. Designed to eliminate redundant metadata loading.
%
% Parameters:
%   MD         - MovieData object
%   outputDir  - Directory where metadata_export.mat will be saved
%   options    - (Optional) struct with fields:
%                .sourceImageFile   - Full path to source image file
%                .pixelSizeXY       - XY pixel size in nm (from script)
%                .pixelSizeZ        - Z pixel size in nm (from script)
%                .timeInterval      - Time interval in seconds (from script)
%                .meshParams        - Mesh parameters struct from script
%                .includeChannelInfo (default: true)
%                .includeMeshStats (default: false)
%                .verbose (default: false)
%
% Output:
%   success    - Boolean indicating export success
%
% Creates:
%   {outputDir}/metadata_export.mat containing 'metadata' struct
%
% Example:
%   opts.sourceImageFile = imageFile;
%   opts.pixelSizeXY = pixelSizeXY;
%   opts.pixelSizeZ = pixelSizeZ;
%   opts.meshParams = p.mesh;
%   exportMetadata(MD, saveDirectory, opts);
%
% Philipp Kaintoch, 2025-12-11 (Updated)

    % Parse optional arguments
    if nargin < 3
        options = struct();
    end

    includeChannelInfo = getOption(options, 'includeChannelInfo', true);
    includeMeshStats = getOption(options, 'includeMeshStats', false);
    verbose = getOption(options, 'verbose', false);

    try
        % Initialize metadata struct
        metadata = struct();

        % === TIER 1: Essential Acquisition (prefer script values over MD) ===
        % Use script-provided values if available, fall back to MD
        if isfield(options, 'pixelSizeXY') && ~isempty(options.pixelSizeXY)
            metadata.pixel_size_xy_nm = double(options.pixelSizeXY);
        else
            metadata.pixel_size_xy_nm = double(MD.pixelSize_);
        end

        if isfield(options, 'pixelSizeZ') && ~isempty(options.pixelSizeZ)
            metadata.pixel_size_z_nm = double(options.pixelSizeZ);
        else
            metadata.pixel_size_z_nm = double(MD.pixelSizeZ_);
        end

        if isfield(options, 'timeInterval') && ~isempty(options.timeInterval)
            metadata.time_interval_sec = double(options.timeInterval);
        else
            metadata.time_interval_sec = double(MD.timeInterval_);
        end

        % Image dimensions
        metadata.image_dimensions = struct();
        metadata.image_dimensions.nX = MD.imSize_(2);
        metadata.image_dimensions.nY = MD.imSize_(1);
        metadata.image_dimensions.nZ = MD.zSize_;
        metadata.image_dimensions.nFrames = MD.nFrames_;
        metadata.image_dimensions.nChannels = length(MD.channels_);

        % === TIER 2: Processing Parameters (prefer script params over process) ===
        % Prioritize script-provided parameters, fall back to Mesh3DProcess
        if isfield(options, 'meshParams') && ~isempty(options.meshParams)
            % Use parameters directly from script
            params = options.meshParams;
            metadata.mesh_parameters = struct();
            metadata.mesh_parameters.mesh_mode = params.meshMode;
            metadata.mesh_parameters.inside_gamma = params.insideGamma;
            metadata.mesh_parameters.inside_blur = params.insideBlur;
            metadata.mesh_parameters.filter_scales = params.filterScales;
            metadata.mesh_parameters.filter_num_std_surface = params.filterNumStdSurface;
            metadata.mesh_parameters.inside_dilate_radius = params.insideDilateRadius;
            metadata.mesh_parameters.inside_erode_radius = params.insideErodeRadius;
            metadata.mesh_parameters.smooth_mesh_mode = params.smoothMeshMode;
            metadata.mesh_parameters.smooth_mesh_iterations = params.smoothMeshIterations;
            metadata.mesh_parameters.use_undeconvolved = params.useUndeconvolved;
            metadata.mesh_parameters.image_gamma = params.imageGamma;
            metadata.mesh_parameters.scale_otsu = params.scaleOtsu;
            metadata.mesh_parameters.smooth_image_size = params.smoothImageSize;
            metadata.mesh_parameters.curvature_median_filter_radius = params.curvatureMedianFilterRadius;
            metadata.mesh_parameters.curvature_smooth_on_mesh_iterations = params.curvatureSmoothOnMeshIterations;
            metadata.mesh_parameters.register_images = params.registerImages;
            metadata.mesh_parameters.save_raw_images = params.saveRawImages;
            metadata.mesh_parameters.registration_mode = params.registrationMode;
        else
            % Fall back to Mesh3DProcess
            meshProcess = findMesh3DProcess(MD);
            if ~isempty(meshProcess)
                params = meshProcess.funParams_;
                metadata.mesh_parameters = struct();
                metadata.mesh_parameters.mesh_mode = params.meshMode;
                metadata.mesh_parameters.inside_gamma = params.insideGamma;
                metadata.mesh_parameters.inside_blur = params.insideBlur;
                metadata.mesh_parameters.filter_scales = params.filterScales;
                metadata.mesh_parameters.filter_num_std_surface = params.filterNumStdSurface;
                metadata.mesh_parameters.inside_dilate_radius = params.insideDilateRadius;
                metadata.mesh_parameters.inside_erode_radius = params.insideErodeRadius;
                metadata.mesh_parameters.smooth_mesh_mode = params.smoothMeshMode;
                metadata.mesh_parameters.smooth_mesh_iterations = params.smoothMeshIterations;
                metadata.mesh_parameters.use_undeconvolved = params.useUndeconvolved;
                metadata.mesh_parameters.image_gamma = params.imageGamma;
                metadata.mesh_parameters.scale_otsu = params.scaleOtsu;
                metadata.mesh_parameters.smooth_image_size = params.smoothImageSize;
                metadata.mesh_parameters.curvature_median_filter_radius = params.curvatureMedianFilterRadius;
                metadata.mesh_parameters.curvature_smooth_on_mesh_iterations = params.curvatureSmoothOnMeshIterations;
                metadata.mesh_parameters.register_images = params.registerImages;
                metadata.mesh_parameters.save_raw_images = params.saveRawImages;
                metadata.mesh_parameters.registration_mode = params.registrationMode;

                if isfield(params, 'otsuThreshold')
                    metadata.otsu_threshold = params.otsuThreshold;
                end
            else
                if verbose
                    warning('exportMetadata:NoMeshProcess', ...
                        'No Mesh3DProcess found and no script params provided.');
                end
            end
        end

        % === TIER 3: Mesh Statistics (Optional) ===
        if includeMeshStats && ~isempty(meshProcess)
            metadata.mesh_stats = extractMeshStatistics(MD, meshProcess);
        end

        % === TIER 4: Provenance ===
        if includeChannelInfo
            metadata.channel_info = extractChannelInfo(MD);
        end

        % Source image information (prefer script-provided)
        if isfield(options, 'sourceImageFile') && ~isempty(options.sourceImageFile)
            metadata.source_image_path = options.sourceImageFile;
            [~, name, ext] = fileparts(options.sourceImageFile);
            metadata.source_image_name = [name ext];
        else
            metadata.source_image_path = MD.channels_(1).channelPath_;
            [~, name, ext] = fileparts(MD.channels_(1).channelPath_);
            metadata.source_image_name = [name ext];
        end

        % Processing information
        metadata.processing_date = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        metadata.matlab_version = version;
        metadata.output_directory = outputDir;
        metadata.frames_processed = MD.nFrames_;

        % === EXPORT ===
        outputPath = fullfile(outputDir, 'metadata_export.mat');
        save(outputPath, 'metadata', '-v7.3');

        if verbose
            fprintf('Metadata exported successfully to:\n  %s\n', outputPath);
        end

        success = true;

    catch ME
        warning('exportMetadata:ExportFailed', ...
            'Failed to export metadata: %s', ME.message);
        success = false;
    end
end

% Helper function: Find Mesh3DProcess
function meshProcess = findMesh3DProcess(MD)
    meshProcess = [];
    for i = 1:length(MD.processes_)
        if isa(MD.processes_{i}, 'Mesh3DProcess')
            meshProcess = MD.processes_{i};
            break;
        end
    end
end

% Helper function: Extract mesh statistics
function stats = extractMeshStatistics(MD, meshProcess)
    % Extract mesh stats from saved mesh files
    % Note: This is a placeholder - full implementation depends on mesh file structure

    stats = struct();
    stats.nVertices = [];
    stats.nFaces = [];
    stats.volume = [];
    stats.surfaceArea = [];

    % Try to load first frame mesh to extract basic stats
    try
        meshDir = meshProcess.outFilePaths_{1};
        meshFiles = dir(fullfile(meshDir, 'surface_*.mat'));

        if ~isempty(meshFiles)
            firstMesh = load(fullfile(meshDir, meshFiles(1).name));
            if isfield(firstMesh, 'surface')
                stats.nVertices = size(firstMesh.surface.vertices, 1);
                stats.nFaces = size(firstMesh.surface.faces, 1);
                % Volume and surface area would require more computation
            end
        end
    catch ME
        % Silently fail - mesh stats are optional
    end
end

% Helper function: Extract channel information
function channelInfo = extractChannelInfo(MD)
    nChannels = length(MD.channels_);
    channelInfo = struct([]);

    for i = 1:nChannels
        % Initialize struct fields with defaults if not available
        if isempty(MD.channels_(i).emissionWavelength_)
            wavelength = NaN;
        else
            wavelength = MD.channels_(i).emissionWavelength_;
        end

        if isempty(MD.channels_(i).excitationWavelength_)
            excitation = NaN;
        else
            excitation = MD.channels_(i).excitationWavelength_;
        end

        if isempty(MD.channels_(i).fluorophore_)
            fluorophore = '';
        else
            fluorophore = MD.channels_(i).fluorophore_;
        end

        channelInfo(i).wavelength_nm = wavelength;
        channelInfo(i).excitation_wavelength_nm = excitation;
        channelInfo(i).fluorophore = fluorophore;
    end
end

% Helper function: Get option with default
function value = getOption(options, field, default)
    if isfield(options, field)
        value = options.(field);
    else
        value = default;
    end
end
