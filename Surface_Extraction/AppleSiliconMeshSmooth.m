function smoothedSurface = AppleSiliconMeshSmooth(surface, iterations, lambda)
% AppleSiliconMeshSmooth - Pure MATLAB mesh smoothing for Apple Silicon
%
% This function provides Laplacian mesh smoothing without requiring compiled
% MEX files, making it compatible with Apple Silicon (ARM64) Macs.
%
% SYNTAX:
%   smoothedSurface = AppleSiliconMeshSmooth(surface)
%   smoothedSurface = AppleSiliconMeshSmooth(surface, iterations)
%   smoothedSurface = AppleSiliconMeshSmooth(surface, iterations, lambda)
%
% INPUTS:
%   surface     - Struct with fields:
%                 .vertices: [Nx3] vertex positions
%                 .faces:    [Mx3] triangle indices
%   iterations  - Number of smoothing iterations (default: 6)
%   lambda      - Smoothing factor, 0-1 (default: 0.5)
%                 0 = no smoothing, 1 = maximum smoothing
%
% OUTPUT:
%   smoothedSurface - Smoothed mesh with same structure as input
%
% ALGORITHM:
%   Implements umbrella operator Laplacian smoothing:
%   v_new = (1-λ)*v_old + λ*mean(neighbors(v))
%
% PERFORMANCE:
%   ~2-5 seconds for 30,000 vertex mesh with 6 iterations
%   Slower than compiled MEX but acceptable for most applications
%
% COMPATIBILITY:
%   Works on all MATLAB versions and architectures:
%   - Apple Silicon (ARM64) ✓
%   - Intel Mac (x86_64) ✓
%   - Windows ✓
%   - Linux ✓
%
% COMPARISON TO SMOOTHPATCH:
%   - smoothpatch mode=0: Equivalent to this function
%   - smoothpatch mode=1: Uses curvature flow (more sophisticated)
%   - This function: Laplacian smoothing (simpler but effective)
%
% EXAMPLE:
%   % Load mesh
%   data = load('surface_1_1.mat');
%
%   % Smooth with defaults
%   smoothed = AppleSiliconMeshSmooth(data.surface);
%
%   % Smooth with custom parameters
%   smoothed = AppleSiliconMeshSmooth(data.surface, 10, 0.6);
%
%   % Visualize
%   figure;
%   subplot(1,2,1); patch(data.surface); title('Original');
%   subplot(1,2,2); patch(smoothed); title('Smoothed');
%
% Author: Philipp Kaintoch 
% Date: 2025-12-07
% Purpose: Apple Silicon compatibility for u-shape3D mesh smoothing

    %% Input validation
    if nargin < 1
        error('AppleSiliconMeshSmooth:InvalidInput', 'Surface struct required');
    end

    if ~isfield(surface, 'vertices') || ~isfield(surface, 'faces')
        error('AppleSiliconMeshSmooth:InvalidInput', ...
              'Surface must have .vertices and .faces fields');
    end

    if nargin < 2 || isempty(iterations)
        iterations = 6;  % Default matches u-shape3D default
    end

    if nargin < 3 || isempty(lambda)
        lambda = 0.5;  % Moderate smoothing
    end

    % Validate parameters
    if iterations < 0 || ~isfinite(iterations)
        error('AppleSiliconMeshSmooth:InvalidInput', ...
              'iterations must be non-negative and finite');
    end

    if lambda < 0 || lambda > 1
        error('AppleSiliconMeshSmooth:InvalidInput', ...
              'lambda must be between 0 and 1');
    end

    %% Extract mesh data
    vertices = double(surface.vertices);
    faces = double(surface.faces);
    nVertices = size(vertices, 1);

    fprintf('  Smoothing mesh: %d vertices, %d faces, %d iterations, λ=%.2f\n', ...
            nVertices, size(faces, 1), iterations, lambda);

    %% Build adjacency list (vertex neighbors)
    fprintf('  Building adjacency list...\n');
    tic;

    % Initialize adjacency structure
    adjacency = cell(nVertices, 1);
    for i = 1:nVertices
        adjacency{i} = [];
    end

    % Populate from face list
    for i = 1:size(faces, 1)
        v1 = faces(i, 1);
        v2 = faces(i, 2);
        v3 = faces(i, 3);

        % Each vertex in triangle is neighbor to the other two
        adjacency{v1} = [adjacency{v1}, v2, v3];
        adjacency{v2} = [adjacency{v2}, v1, v3];
        adjacency{v3} = [adjacency{v3}, v1, v2];
    end

    % Remove duplicates and self-references
    for i = 1:nVertices
        adjacency{i} = unique(adjacency{i});
        adjacency{i}(adjacency{i} == i) = [];  % Remove self
    end

    tAdj = toc;
    fprintf('  Adjacency built in %.2f seconds\n', tAdj);

    %% Laplacian smoothing iterations
    fprintf('  Performing smoothing iterations...\n');
    tic;

    newVertices = vertices;

    for iter = 1:iterations
        tempVertices = newVertices;

        % Update each vertex
        for i = 1:nVertices
            neighbors = adjacency{i};

            if ~isempty(neighbors)
                % Compute mean position of neighbors
                neighborPos = mean(newVertices(neighbors, :), 1);

                % Weighted update: mix old position with neighbor average
                tempVertices(i, :) = (1 - lambda) * newVertices(i, :) + ...
                                     lambda * neighborPos;
            end
            % If no neighbors, vertex stays at original position
        end

        newVertices = tempVertices;

        % Progress indicator
        if mod(iter, max(1, floor(iterations/5))) == 0
            fprintf('    Iteration %d/%d\n', iter, iterations);
        end
    end

    tSmooth = toc;
    fprintf('  Smoothing completed in %.2f seconds\n', tSmooth);

    %% Calculate smoothing statistics
    displacement = newVertices - vertices;
    meanDisplacement = mean(sqrt(sum(displacement.^2, 2)));
    maxDisplacement = max(sqrt(sum(displacement.^2, 2)));

    fprintf('  Mean vertex displacement: %.3f voxels\n', meanDisplacement);
    fprintf('  Max vertex displacement: %.3f voxels\n', maxDisplacement);

    %% Return smoothed surface
    smoothedSurface = surface;
    smoothedSurface.vertices = newVertices;

    fprintf('  Total time: %.2f seconds\n', tAdj + tSmooth);
end
