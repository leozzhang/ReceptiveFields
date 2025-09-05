% ==========================================================================
% ORIENTED/GABOR RECEPTIVE FIELD GENERATOR
% ==========================================================================

% Configuration - Easy to modify all parameters here
function config = getOrientedRFConfig()
    config.image_size = 25;
    config.center_position_range = [8, 17];  % min, max for both x and y
    config.orientation_steps = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5];  % degrees
    config.frequency_range = [1.0, 5.0];  % cycles across the RF
    config.sigma_x_range = [1.5, 3.5];  % width of Gaussian envelope (along bars)
    config.sigma_y_ratio_range = [2.0, 4.0];  % sigma_y = sigma_x * ratio (elongated)
    config.amplitude_range = [0.5, 1.5];  % strength of response
    config.noise_level_range = [0.0, 0.2];  % noise as fraction of signal
    config.normalize_energy = true;  % normalize to unit energy
    config.phases = [0, 90];  % 0 = bright center, 90 = dark center (degrees)
end

% ==========================================================================
% CORE FUNCTIONS
% ==========================================================================

function [rf, params] = generateSingleOriented(varargin)
    % Generate a single oriented/Gabor receptive field
    % Optional parameters: 'center_x', 'center_y', 'orientation', 'frequency',
    %                     'sigma_x', 'sigma_y', 'amplitude', 'phase'
    
    % Parse inputs
    p = inputParser;
    config = getOrientedRFConfig();
    
    addParameter(p, 'center_x', [], @isnumeric);
    addParameter(p, 'center_y', [], @isnumeric);
    addParameter(p, 'orientation', [], @isnumeric);  % degrees
    addParameter(p, 'frequency', [], @isnumeric);
    addParameter(p, 'sigma_x', [], @isnumeric);
    addParameter(p, 'sigma_y', [], @isnumeric);
    addParameter(p, 'amplitude', [], @isnumeric);
    addParameter(p, 'phase', [], @isnumeric);  % 0 or 90 degrees
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
    
    if isempty(p.Results.orientation)
        orientation = p.Results.config.orientation_steps(randi(length(p.Results.config.orientation_steps)));
    else
        orientation = p.Results.orientation;
    end
    
    if isempty(p.Results.frequency)
        range = p.Results.config.frequency_range;
        frequency = range(1) + (range(2) - range(1)) * rand();
    else
        frequency = p.Results.frequency;
    end
    
    if isempty(p.Results.sigma_x)
        range = p.Results.config.sigma_x_range;
        sigma_x = range(1) + (range(2) - range(1)) * rand();
    else
        sigma_x = p.Results.sigma_x;
    end
    
    if isempty(p.Results.sigma_y)
        range = p.Results.config.sigma_y_ratio_range;
        ratio = range(1) + (range(2) - range(1)) * rand();
        sigma_y = sigma_x * ratio;
    else
        sigma_y = p.Results.sigma_y;
    end
    
    if isempty(p.Results.amplitude)
        range = p.Results.config.amplitude_range;
        amplitude = range(1) + (range(2) - range(1)) * rand();
    else
        amplitude = p.Results.amplitude;
    end
    
    if isempty(p.Results.phase)
        phase = p.Results.config.phases(randi(length(p.Results.config.phases)));
    else
        phase = p.Results.phase;
    end
    
    % Create coordinate grids
    [X, Y] = meshgrid(1:image_size, 1:image_size);
    
    % Center coordinates
    X_centered = X - center_x;
    Y_centered = Y - center_y;
    
    % Convert orientation to radians
    theta = deg2rad(orientation);
    
    % Rotate coordinates
    X_rot = X_centered * cos(theta) + Y_centered * sin(theta);
    Y_rot = -X_centered * sin(theta) + Y_centered * cos(theta);
    
    % Create Gaussian envelope (elliptical)
    gaussian = exp(-(X_rot.^2 / (2 * sigma_x^2) + Y_rot.^2 / (2 * sigma_y^2)));
    
    % Create sinusoidal grating
    wavelength = image_size / frequency;  % pixels per cycle
    grating = sin(2 * pi * X_rot / wavelength + deg2rad(phase));
    
    % Combine Gaussian envelope with grating
    rf = amplitude * gaussian .* grating;
    
    % Store parameters used
    params = struct();
    params.center_x = center_x;
    params.center_y = center_y;
    params.orientation = orientation;
    params.frequency = frequency;
    params.sigma_x = sigma_x;
    params.sigma_y = sigma_y;
    params.amplitude = amplitude;
    params.phase = phase;
    params.image_size = image_size;
end

function rf_noisy = addOrientedNoise(rf, varargin)
    % Add multiple types of random noise/distortions to oriented RF
    
    p = inputParser;
    config = getOrientedRFConfig();
    addParameter(p, 'noise_level', [], @isnumeric);
    addParameter(p, 'config', config, @isstruct);
    addParameter(p, 'gaussian_noise', 0.05, @isnumeric);  % Max Gaussian noise
    addParameter(p, 'salt_pepper', 0, @isnumeric);    % Max salt & pepper
    addParameter(p, 'blur_amount', [0, 0.8], @isnumeric); % Random blur range
    addParameter(p, 'contrast_range', [0.8, 1.2], @isnumeric); % Contrast variation
    parse(p, varargin{:});
    
    rf_noisy = rf;
    
    % Original noise (keep this for backward compatibility)
    if ~isempty(p.Results.noise_level)
        noise_level = p.Results.noise_level;
    else
        range = p.Results.config.noise_level_range;
        noise_level = range(1) + (range(2) - range(1)) * rand();
    end
    signal_strength = std(rf(:));
    rf_noisy = rf_noisy + randn(size(rf)) * noise_level * signal_strength;
    
    % NEW: Additional noise types (same as center-surround)
    % 1. Extra Gaussian noise
    if p.Results.gaussian_noise > 0
        extra_noise = p.Results.gaussian_noise * rand();
        rf_noisy = rf_noisy + randn(size(rf)) * extra_noise;
    end
    
    % 2. Salt and pepper noise
    if p.Results.salt_pepper > 0
        sp_level = p.Results.salt_pepper * rand();
        rf_noisy = imnoise(rf_noisy, 'salt & pepper', sp_level);
    end
    
    % 3. Random blur
    blur_range = p.Results.blur_amount;
    blur_sigma = blur_range(1) + (blur_range(2) - blur_range(1)) * rand();
    if blur_sigma > 0
        rf_noisy = imgaussfilt(rf_noisy, blur_sigma);
    end
    
    % 4. Random contrast
    contrast_range = p.Results.contrast_range;
    contrast_factor = contrast_range(1) + (contrast_range(2) - contrast_range(1)) * rand();
    rf_noisy = rf_noisy * contrast_factor;
    
    % Keep values reasonable
    rf_noisy = max(-3, min(3, rf_noisy));
end

function rf_normalized = normalizeOrientedFilter(rf)
    % Normalize filter to unit energy
    rf_normalized = rf / norm(rf(:));
end

function [batch, params_list] = generateOrientedBatch(n_samples, varargin)
    % Generate a batch of oriented RFs with automatic visualization
    
    p = inputParser;
    config = getOrientedRFConfig();
    addParameter(p, 'config', config, @isstruct);
    addParameter(p, 'visualize', true, @islogical);
    addParameter(p, 'max_visualize', 10, @isnumeric);
    parse(p, varargin{:});
    
    image_size = p.Results.config.image_size;
    batch = zeros(n_samples, image_size, image_size);
    params_list = cell(n_samples, 1);
    
    fprintf('Generating %d oriented RFs...\n', n_samples);
    
    for i = 1:n_samples
        % Generate clean RF - NEW random parameters each iteration
        [rf, params] = generateSingleOriented('config', p.Results.config);
        
        % Add noise - NEW random noise each iteration
        rf = addOrientedNoise(rf, 'config', p.Results.config);
        
        % Normalize if specified
        rf = (rf - min(rf(:))) / (max(rf(:)) - min(rf(:)));  % Scale to [0,1]
        rf = (rf - 0.5) * 2;  % Center around 0, range [-1, 1]
        
        batch(i, :, :) = rf;
        params_list{i} = params;
    end
    
    % Automatic visualization (capped at max_visualize)
    if p.Results.visualize
        n_to_show = min(p.Results.max_visualize, n_samples);
        visualizeOrientedSamples(batch, params_list, 'n_show', n_to_show);
        
        % Print summary statistics
        phases = cellfun(@(p) p.phase, params_list);
        bright_count = sum(phases == 0);
        dark_count = sum(phases == 90);
        orientations = cellfun(@(p) p.orientation, params_list);
        unique_orientations = unique(orientations);
        
        fprintf('Generated %d RFs: %d bright-center, %d dark-center\n', ...
                n_samples, bright_count, dark_count);
        fprintf('Orientations used: %s degrees\n', mat2str(unique_orientations));
    end
end

% ==========================================================================
% VISUALIZATION FUNCTIONS
% ==========================================================================

function visualizeOrientedSamples(batch, params_list, varargin)
    % Visualize a grid of generated oriented RFs
    
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
        
        % Set colormap
        if strcmp(p.Results.colormap_name, 'seismic')
            colormap(gray);
        else
            colormap(p.Results.colormap_name);
        end
        
        axis off;
        
        % Add title with key parameters
        params = params_list{i};
        title_str = sprintf('θ=%.0f°, Ph=%d°', params.orientation, params.phase);
        title(title_str, 'FontSize', 8);
    end
    
    sgtitle('Generated Oriented/Gabor RFs');
end

[batch, params_list] = generateOrientedBatch(10)