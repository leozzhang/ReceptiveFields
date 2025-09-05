function config = getOrientedRFConfig() % oriented generator parameters
    config.image_size = 25;
    config.center_position_range = [8, 17];  % min, max for both x and y
    config.orientation_steps = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5];  % degrees
    config.frequency_range = [5.0, 15.0];  % cycles across the RF
    config.sigma_x_range = [2, 6];  % width of Gaussian envelope (along bars)
    config.sigma_y_ratio_range = [2.0, 4.0];  % sigma_y = sigma_x * ratio (elongated)
    config.amplitude_range = [0.5, 1.5];  % strength of response
    config.noise_level_range = [0.0, 0.2];  % noise as fraction of signal
    config.normalize_energy = true;  % normalize to unit energy
    config.phases = [0, 90];  % 0 = bright center, 90 = dark center (degrees)
end

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
    
    % Create MORE REALISTIC grating (less perfectly linear)
    wavelength = image_size / frequency;  % pixels per cycle
    
    % 1. Add frequency variation across space (less uniform)
    freq_variation = 1 + 0.2 * sin(2 * pi * Y_rot / (wavelength * 2));  % Slight freq changes
    local_wavelength = wavelength ./ freq_variation;
    
    % 2. Add slight curvature to bars (less perfectly straight)
    curvature = 0.1 * (X_rot.^2) / (image_size^2);  % Slight curvature
    curved_phase = deg2rad(phase) + curvature;
    
    % 3. Create grating with variations
    grating = sin(2 * pi * X_rot ./ local_wavelength + curved_phase);
    
    % 4. Add local orientation jitter (bars not perfectly parallel)
    if rand() < 0.4  % 40% chance of orientation jitter
        jitter_strength = 0.05;
        orientation_noise = jitter_strength * randn(size(X_rot));
        
        % Apply local rotations
        X_jitter = X_rot .* cos(orientation_noise) - Y_rot .* sin(orientation_noise);
        grating = sin(2 * pi * X_jitter ./ local_wavelength + curved_phase);
    end
    
    % 5. Make Gaussian envelope less perfectly elliptical
    envelope_noise = 0.1 * randn(size(gaussian));
    gaussian_irregular = gaussian + envelope_noise;
    gaussian_irregular = max(0, gaussian_irregular);  % Keep positive
    
    % Combine irregular envelope with varied grating
    rf = amplitude * gaussian_irregular .* grating;
    
    % 6. Add some "breaks" in the bars occasionally
    if rand() < 0.3  % 30% chance
        num_breaks = randi([1, 2]);
        for brk = 1:num_breaks
            break_x = randi([5, image_size-4]);
            break_y = randi([5, image_size-4]);
            break_size = randi([2, 4]);
            
            % Create small circular "break" in the pattern
            break_distances = sqrt((X - break_x).^2 + (Y - break_y).^2);
            break_mask = break_distances <= break_size;
            rf(break_mask) = rf(break_mask) * 0.3;  % Reduce strength in break area
        end
    end
    
    % 7. Add texture noise to break up perfect sinusoids
    texture_noise = 0.1 * randn(size(rf));
    rf = rf + texture_noise;
    
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
    
    % Make slightly oval instead of perfect circle
    % Add random elliptical distortion
    ellipse_ratio = 0.7 + 0.6 * rand();  % Ratio between 0.7 and 1.3
    ellipse_angle = rand() * 2 * pi;  % Random orientation
    
    % Apply elliptical transformation
    cos_angle = cos(ellipse_angle);
    sin_angle = sin(ellipse_angle);
    
    % Rotate coordinates
    X_rot = (X - center_x) * cos_angle + (Y - center_y) * sin_angle;
    Y_rot = -(X - center_x) * sin_angle + (Y - center_y) * cos_angle;
    
    % Apply elliptical scaling
    X_scaled = X_rot;
    Y_scaled = Y_rot / ellipse_ratio;  % Compress/stretch one axis
    
    % Calculate elliptical distances
    distances = sqrt(X_scaled.^2 + Y_scaled.^2);
    
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
    
    % Make it less blob-like and more realistic
    % 1. Add irregular edges by varying the radii slightly
    [H, W] = size(rf);
    [X, Y] = meshgrid(1:W, 1:H);
    angle = atan2(Y - center_y, X - center_x);
    
    % Add radial irregularity (make edges less perfect)
    irregularity = 0.3 * sin(6 * angle) + 0.2 * sin(4 * angle); % Wavy edges
    center_radius_varied = center_radius + irregularity;
    surround_radius_varied = surround_radius + irregularity;
    
    % Use the elliptical distances (don't recalculate circular distances)
    center_mask_irreg = distances <= center_radius_varied;
    surround_mask_irreg = (distances > center_radius_varied) & (distances <= surround_radius_varied);
    
    % Apply the irregular pattern
    rf = zeros(H, W);
    if is_on_center
        center_val = p.Results.config.center_value;
        surround_val = p.Results.config.surround_value;
    else
        center_val = -p.Results.config.center_value;
        surround_val = -p.Results.config.surround_value;
    end
    
    rf(center_mask_irreg) = center_val * amplitude;
    rf(surround_mask_irreg) = surround_val * amplitude;
    
    % 2. Add some texture/noise to break up smooth regions
    texture_noise = 0.15 * randn(size(rf));
    rf = rf + texture_noise;
    
    % 3. Light blur to soften harsh edges but keep structure
    rf = imgaussfilt(rf, 0.5);  % Less blur than before
    
    % 4. Add some random "holes" or "bumps" occasionally
    if rand() < 0.3  % 30% chance
        num_spots = randi([1, 3]);
        for spot = 1:num_spots
            spot_x = randi([3, W-2]);
            spot_y = randi([3, H-2]);
            spot_size = randi([1, 2]);
            spot_strength = 0.3 * (rand() - 0.5) * amplitude;
            
            % Add small random spots
            spot_distances = sqrt((X - spot_x).^2 + (Y - spot_y).^2);
            spot_mask = spot_distances <= spot_size;
            rf(spot_mask) = rf(spot_mask) + spot_strength;
        end
    end
    
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
    % Add multiple types of random noise/distortions to center-surround RF
    
    p = inputParser;
    config = getRFConfig();
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
    
    % NEW: Additional noise types
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
        rf = (rf - min(rf(:))) / (max(rf(:)) - min(rf(:)));  % Scale to [0,1]
        rf = (rf - 0.5) * 2;  % Center around 0, range [-1, 1]
        
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

% ==========================================================================
% SAVE GENERATED RFs AS PNG FILES (like CIFAR-10 structure)
% ==========================================================================

function saveRFDataset(n_center_surround, n_oriented, base_folder, varargin)
    % Save generated RFs as PNG files in CIFAR-10 style folder structure
    %
    % Usage:
    %   saveRFDataset(1000, 1000, 'rf_dataset');
    %   saveRFDataset(500, 500, 'rf_dataset', 'split_train_test', true);
    
    p = inputParser;
    addParameter(p, 'split_train_test', false, @islogical);  % Create train/test split
    addParameter(p, 'test_fraction', 0.2, @isnumeric);  % Fraction for test set
    addParameter(p, 'cs_config', getRFConfig(), @isstruct);  % Center-surround config
    addParameter(p, 'or_config', getOrientedRFConfig(), @isstruct);  % Oriented config
    parse(p, varargin{:});
    
    fprintf('Generating and saving RF dataset...\n');
    fprintf('Center-surround: %d samples\n', n_center_surround);
    fprintf('Oriented: %d samples\n', n_oriented);
    
    % Create folder structure
    if p.Results.split_train_test
        % Create train/test split like CIFAR-10
        cs_train_folder = fullfile(base_folder, 'train', 'center_surround');
        cs_test_folder = fullfile(base_folder, 'test', 'center_surround');
        or_train_folder = fullfile(base_folder, 'train', 'oriented');
        or_test_folder = fullfile(base_folder, 'test', 'oriented');
        
        mkdir(cs_train_folder);
        mkdir(cs_test_folder);
        mkdir(or_train_folder);
        mkdir(or_test_folder);
    else
        % Single folder structure
        cs_folder = fullfile(base_folder, 'center_surround');
        or_folder = fullfile(base_folder, 'oriented');
        
        mkdir(cs_folder);
        mkdir(or_folder);
    end
    
    % Generate and save center-surround RFs
    fprintf('Generating center-surround RFs...\n');
    [cs_batch, ~] = generateBatch(n_center_surround, 'visualize', false, 'config', p.Results.cs_config);
    
    if p.Results.split_train_test
        n_cs_test = round(n_center_surround * p.Results.test_fraction);
        n_cs_train = n_center_surround - n_cs_test;
        
        % Save training set
        for i = 1:n_cs_train
            img = squeeze(cs_batch(i, :, :));
            img_normalized = normalizeForPNG(img);
            filename = sprintf('cs_train_%04d.png', i);
            imwrite(img_normalized, fullfile(cs_train_folder, filename));
        end
        
        % Save test set
        for i = 1:n_cs_test
            img = squeeze(cs_batch(n_cs_train + i, :, :));
            img_normalized = normalizeForPNG(img);
            filename = sprintf('cs_test_%04d.png', i);
            imwrite(img_normalized, fullfile(cs_test_folder, filename));
        end
        
        fprintf('Saved %d train + %d test center-surround images\n', n_cs_train, n_cs_test);
    else
        for i = 1:n_center_surround
            img = squeeze(cs_batch(i, :, :));
            img_normalized = normalizeForPNG(img);
            filename = sprintf('cs_%04d.png', i);
            imwrite(img_normalized, fullfile(cs_folder, filename));
        end
        fprintf('Saved %d center-surround images\n', n_center_surround);
    end
    
    % Generate and save oriented RFs
    fprintf('Generating oriented RFs...\n');
    [or_batch, ~] = generateOrientedBatch(n_oriented, 'visualize', false, 'config', p.Results.or_config);
    
    if p.Results.split_train_test
        n_or_test = round(n_oriented * p.Results.test_fraction);
        n_or_train = n_oriented - n_or_test;
        
        % Save training set
        for i = 1:n_or_train
            img = squeeze(or_batch(i, :, :));
            img_normalized = normalizeForPNG(img);
            filename = sprintf('or_train_%04d.png', i);
            imwrite(img_normalized, fullfile(or_train_folder, filename));
        end
        
        % Save test set
        for i = 1:n_or_test
            img = squeeze(or_batch(n_or_train + i, :, :));
            img_normalized = normalizeForPNG(img);
            filename = sprintf('or_test_%04d.png', i);
            imwrite(img_normalized, fullfile(or_test_folder, filename));
        end
        
        fprintf('Saved %d train + %d test oriented images\n', n_or_train, n_or_test);
    else
        for i = 1:n_oriented
            img = squeeze(or_batch(i, :, :));
            img_normalized = normalizeForPNG(img);
            filename = sprintf('or_%04d.png', i);
            imwrite(img_normalized, fullfile(or_folder, filename));
        end
        fprintf('Saved %d oriented images\n', n_oriented);
    end
    
    fprintf('Dataset saved to: %s\n', base_folder);
    
end

function img_normalized = normalizeForPNG(img)
    % Force all images to have the same distribution
    % Target: mean = 127.5, std = 40 (or whatever you choose)
    
    target_mean = 150;
    target_std = 20;
    
    % Z-score normalize
    img_zscore = (img - mean(img(:))) / std(img(:));
    
    % Scale to target distribution
    img_scaled = img_zscore * target_std + target_mean;
    
    % Clip to [0, 255]
    img_normalized = uint8(max(0, min(255, img_scaled)));
end

saveRFDataset(500,500,'rf_dataset500', 'split_train_test', true)
%% 
% Re-examine your training data
[cs_batch, ~] = generateBatch(5, 'visualize', true);  % Force visualization
[or_batch, ~] = generateOrientedBatch(5, 'visualize', true);

% Check if they actually look different
figure;
for i = 1:3
    subplot(2, 3, i);
    imagesc(squeeze(cs_batch(i,:,:))); 
    colormap gray; axis off;
    title('Generated CS');
    
    subplot(2, 3, i+3);
    imagesc(squeeze(or_batch(i,:,:))); 
    colormap gray; axis off;
    title('Generated OR');
end
