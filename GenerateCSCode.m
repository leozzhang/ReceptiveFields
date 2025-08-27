% ==========================================================================
% CENTER-SURROUND RECEPTIVE FIELD GENERATOR
% ==========================================================================

% Configuration - Easy to modify all parameters here
function config = getRFConfig()
    config.image_size = 25;
    config.center_position_range = [8, 17];  % min, max for both x and y
    config.center_radius_range = [1.5, 4.0];  % radius in pixels
    config.surround_multiplier_range = [1.3, 2.5];  % surround = center * multiplier
    config.amplitude_range = [0.5, 1.5];  % strength of response
    config.noise_level_range = [0.0, 0.2];  % noise as fraction of signal
    config.normalize_energy = true;  % normalize to unit energy
    config.center_value = 1.0;  % positive value for center
    config.surround_value = -0.5;  % negative value for surround
end

% ==========================================================================
% CORE FUNCTIONS
% ==========================================================================

function [rf, params] = generateSingleCenterSurround(varargin)
    % Generate a single center-surround receptive field
    % Optional parameters: 'center_x', 'center_y', 'center_radius', 
    %                     'surround_radius', 'amplitude', 'is_on_center'
    
    % Parse inputs
    p = inputParser;
    config = getRFConfig();
    
    addParameter(p, 'center_x', [], @isnumeric);
    addParameter(p, 'center_y', [], @isnumeric);
    addParameter(p, 'center_radius', [], @isnumeric);
    addParameter(p, 'surround_radius', [], @isnumeric);
    addParameter(p, 'amplitude', [], @isnumeric);
    addParameter(p, 'is_on_center', true, @islogical);
    addParameter(p, 'config', config, @isstruct);
    
    parse(p, varargin{:});
    
    % Extract values or sample randomly
    image_size = p.Results.config.image_size;
    
    if isempty(p.Results.center_x)
        range = p.Results.config.center_position_range;
        center_x = range(1) + (range(2) - range(1)) * rand();
    else
        center_x = p.Results.center_x;
    end
    
    if isempty(p.Results.center_y)
        range = p.Results.config.center_position_range;
        center_y = range(1) + (range(2) - range(1)) * rand();
    else
        center_y = p.Results.center_y;
    end
    
    if isempty(p.Results.center_radius)
        range = p.Results.config.center_radius_range;
        center_radius = range(1) + (range(2) - range(1)) * rand();
    else
        center_radius = p.Results.center_radius;
    end
    
    if isempty(p.Results.surround_radius)
        range = p.Results.config.surround_multiplier_range;
        multiplier = range(1) + (range(2) - range(1)) * rand();
        surround_radius = center_radius * multiplier;
    else
        surround_radius = p.Results.surround_radius;
    end
    
    if isempty(p.Results.amplitude)
        range = p.Results.config.amplitude_range;
        amplitude = range(1) + (range(2) - range(1)) * rand();
    else
        amplitude = p.Results.amplitude;
    end
    
    is_on_center = p.Results.is_on_center;
    
    % Create coordinate grids
    [X, Y] = meshgrid(1:image_size, 1:image_size);
    
    % Calculate distances from center
    distances = sqrt((X - center_x).^2 + (Y - center_y).^2);
    
    % Initialize RF
    rf = zeros(image_size, image_size);
    
    % Create center region
    center_mask = distances <= center_radius;
    if is_on_center
        center_val = p.Results.config.center_value;
    else
        center_val = -p.Results.config.center_value;
    end
    rf(center_mask) = center_val * amplitude;
    
    % Create surround region (annular ring)
    surround_mask = (distances > center_radius) & (distances <= surround_radius);
    if is_on_center
        surround_val = p.Results.config.surround_value;
    else
        surround_val = -p.Results.config.surround_value;
    end
    rf(surround_mask) = surround_val * amplitude;
    
    % Store parameters
    params = struct();
    params.center_x = center_x;
    params.center_y = center_y;
    params.center_radius = center_radius;
    params.surround_radius = surround_radius;
    params.amplitude = amplitude;
    params.is_on_center = is_on_center;
    params.image_size = image_size;
end

function rf_noisy = addNoise(rf, varargin)
    % Add Gaussian noise to receptive field
    
    p = inputParser;
    config = getRFConfig();
    addParameter(p, 'noise_level', [], @isnumeric);
    addParameter(p, 'config', config, @isstruct);
    parse(p, varargin{:});
    
    if isempty(p.Results.noise_level)
        range = p.Results.config.noise_level_range;
        noise_level = range(1) + (range(2) - range(1)) * rand();
    else
        noise_level = p.Results.noise_level;
    end
    
    signal_strength = std(rf(:));
    noise = randn(size(rf)) * noise_level * signal_strength;
    
    rf_noisy = rf + noise;
end

function rf_normalized = normalizeFilter(rf)
    % Normalize filter to unit energy (like your existing MATLAB code)
    rf_normalized = rf / norm(rf(:));
end

function [batch, params_list] = generateBatch(n_samples, varargin)
    % Generate a batch of center-surround RFs with automatic visualization
    
    p = inputParser;
    config = getRFConfig();
    addParameter(p, 'config', config, @isstruct);
    addParameter(p, 'visualize', true, @islogical);  % option to turn off visualization
    addParameter(p, 'max_visualize', 10, @isnumeric);  % max number to show
    parse(p, varargin{:});
    
    image_size = p.Results.config.image_size;
    batch = zeros(n_samples, image_size, image_size);
    params_list = cell(n_samples, 1);
    
    fprintf('Generating %d center-surround RFs...\n', n_samples);
    
    for i = 1:n_samples
        % Random ON/OFF choice - NEW random each iteration
        is_on_center = rand() > 0.5;
        
        % Generate clean RF - NEW random parameters each iteration
        [rf, params] = generateSingleCenterSurround('is_on_center', is_on_center, ...
                                                   'config', p.Results.config);
        
        % Add noise - NEW random noise each iteration
        rf = addNoise(rf, 'config', p.Results.config);
        
        % Normalize if specified
        if p.Results.config.normalize_energy
            rf = normalizeFilter(rf);
        end
        
        batch(i, :, :) = rf;
        params_list{i} = params;
    end
    
    % Automatic visualization (capped at max_visualize)
    if p.Results.visualize
        n_to_show = min(p.Results.max_visualize, n_samples);
        visualizeSamples(batch, params_list, 'n_show', n_to_show);
        
        % Print summary statistics
        on_center_count = sum(cellfun(@(p) p.is_on_center, params_list));
        fprintf('Generated %d RFs: %d ON-center, %d OFF-center\n', ...
                n_samples, on_center_count, n_samples - on_center_count);
    end
end

% ==========================================================================
% VISUALIZATION FUNCTIONS
% ==========================================================================

function visualizeSamples(batch, params_list, varargin)
    % Visualize a grid of generated RFs
    
    p = inputParser;
    addParameter(p, 'n_show', 9, @isnumeric);
    addParameter(p, 'colormap_name', 'seismic', @ischar);
    parse(p, varargin{:});
    
    n_show = min(p.Results.n_show, size(batch, 1));
    grid_size = ceil(sqrt(n_show));
    
    figure;
    for i = 1:n_show
        subplot(grid_size, grid_size, i);
        
        rf = squeeze(batch(i, :, :));
        imagesc(rf);
        
        % Set colormap (try to match your seismic colormap)
        if strcmp(p.Results.colormap_name, 'seismic')
            colormap(gray);  % or use your preferred red-blue colormap
        else
            colormap(p.Results.colormap_name);
        end
        
        axis off;
        
        % Add title with key parameters
        params = params_list{i};
        title_str = sprintf('R=%.1f, ON=%d', params.center_radius, params.is_on_center);
        title(title_str, 'FontSize', 8);
    end
    
    sgtitle('Generated Center-Surround RFs');
end

[batch, params] = generateBatch(5)