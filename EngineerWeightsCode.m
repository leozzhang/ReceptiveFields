load("retnet8.mat","retNet8")
rng(2)
function rf = computeFilterRF_LindseyMethod(net, layerName, filterIndex, varargin)
% COMPUTEFILTERRF_LINDSEYMETHOD - Exact replication of Lindsey et al. method
% 
% This replicates the exact gradient ascent method from the paper:
% - Single gradient step
% - Gray initialization  
% - Step size of 1.0
% - Specific spatial position targeting

% Parse inputs
p = inputParser;
addParameter(p, 'InputSize', [32, 32, 1], @isnumeric);
addParameter(p, 'StepSize', 1.0, @isnumeric);
addParameter(p, 'SpatialPos', 'center', @ischar); % 'center' or [x,y]
parse(p, varargin{:});

inputSize = p.Results.InputSize;
stepSize = p.Results.StepSize;
spatialPos = p.Results.SpatialPos;

% Initialize input exactly like the paper: gray image (0.5)
inputImg = dlarray(0.5 * ones([inputSize, 1], 'single'), 'SSCB');

% Single gradient step (like the paper)
[loss, gradients] = dlfeval(@lindseyLossFcn, net, inputImg, layerName, filterIndex, spatialPos);

% Normalize gradients exactly like the paper
gradNorm = sqrt(sum(gradients.^2, 'all'));
if gradNorm > 1e-5
    gradients = gradients / (gradNorm + 1e-5);
end

% Single update step
inputImg = inputImg + stepSize * gradients;

% Extract result
rf = extractdata(squeeze(inputImg(:, :, 1, 1)));

% Post-process exactly like the paper
rf = rf - mean(rf(:));
rf_std = std(rf(:));
if rf_std > 1e-5
    rf = rf / rf_std;
end
rf = rf * 0.1; % Scale like paper
rf = rf + 0.5; % Add offset like paper
rf = max(0, min(1, rf)); % Clip to [0,1]
end

function [loss, gradients] = lindseyLossFcn(net, inputImg, layerName, filterIndex, spatialPos)
% Loss function replicating the paper exactly

% Forward pass
layerOutput = forward(net, inputImg, 'Outputs', layerName);

% Get spatial dimensions
[H, W, C, B] = size(layerOutput);

% Determine spatial position (paper uses center)
if strcmp(spatialPos, 'center')
    pos_x = round(H/2);
    pos_y = round(W/2);
else
    pos_x = spatialPos(1);
    pos_y = spatialPos(2);
end

% Loss: mean activation at specific spatial position and filter
% Paper uses: K.mean(layer_output[:, pos_x, pos_y, filter_index])
loss = mean(layerOutput(pos_x, pos_y, filterIndex, :), 'all');

% Compute gradients
gradients = dlgradient(loss, inputImg);
end

function visualizeFilters_LindseyMethod(net, layerName, numFilters)
% Visualize multiple filters using exact Lindsey method

% Get input size from network
inputSize = net.Layers(1).InputSize;

figure;
for i = 1:numFilters
    fprintf('Processing filter %d/%d (Lindsey method)\n', i, numFilters);
    
    % Compute RF using exact paper method
    rf = computeFilterRF_LindseyMethod(net, layerName, i, 'InputSize', inputSize);
    
    subplot(ceil(sqrt(numFilters)), ceil(sqrt(numFilters)), i);
    imagesc(rf);
    colormap gray;
    axis off;
    title(sprintf('Filter %d', i), 'FontSize', 8);
end
sgtitle(sprintf('Receptive Fields (Lindsey Method) - %s', layerName));
end
function optimized_weights = simpleGradientOptimization(net, target_image_path)
    % Load and preprocess the target image
    target_img = imread(target_image_path);
    
    % Convert to grayscale if needed
    if size(target_img, 3) == 3
        target_img = rgb2gray(target_img);
    end
    
    % Convert to double and normalize
    target_gradient = double(target_img);
    target_gradient = target_gradient / 255;
    target_gradient = target_gradient - mean(target_gradient(:));
    
    fprintf('Loaded target image: %dx%d\n', size(target_gradient,1), size(target_gradient,2));
    
    % Extract current weights from retNet8
    current_conv2_weights = net.Layers(4).Weights; % Adjust index as needed
    current_conv2_bias = net.Layers(4).Bias;
    initial_weights = current_conv2_weights;
    filter_idx = 1;
    
    % Compute current gradient to get size
    current_gradient = computeFilterRF_LindseyMethod(net, 'conv2', filter_idx);
    fprintf('Current gradient size: %dx%d\n', size(current_gradient,1), size(current_gradient,2));
    
    % Resize target to match
    target_gradient = imresize(target_gradient, [size(current_gradient,1), size(current_gradient,2)]);
    fprintf('Resized target to: %dx%d\n', size(target_gradient,1), size(target_gradient,2));
    
    % Simple settings
    max_iterations = 1500;
    
    fprintf('Starting optimization...\n');
    
    for iter = 1:max_iterations
        % Compute current gradient
        current_gradient = computeFilterRF_LindseyMethod(net, 'conv2', filter_idx);
        
        % How different is it from target?
        %1 error = sum((current_gradient(:) - target_gradient(:)).^2);
        %testing error with normalized gradients
        norm_target = (target_gradient - mean(target_gradient(:))) / std(target_gradient(:));
        norm_current = (current_gradient - mean(current_gradient(:))) / std(current_gradient(:));
        % Compute error on normalized versions
        lambda = 0.01;  % Adjust as needed
        pattern_error = sum((norm_current(:) - norm_target(:)).^2);
        deviation_penalty = lambda * sum((current_conv2_weights(:) - initial_weights(:)).^2);
        error = pattern_error + deviation_penalty;

        fprintf('Iteration %d: Error = %.6f\n', iter, error);
        
        % If good enough, stop
        if error < 0.001
            break;
        end
        
        % Try small random perturbations to weights
        weight_update = randn(size(current_conv2_weights)) * 0.5;
        temp_weights = current_conv2_weights + weight_update;
        
        % Create new network with updated weights
        temp_layers = [
            imageInputLayer([32 32 1], 'Name', 'input', 'Normalization', 'none')
            convolution2dLayer(9, 32, 'Name', 'conv1', 'Padding', 'same', ...
                'Weights', net.Layers(2).Weights, 'Bias', net.Layers(2).Bias)
            reluLayer('Name', 'relu1')
            convolution2dLayer(9, 1, 'Name', 'conv2', 'Padding', 'same', ...
                'Weights', temp_weights, 'Bias', current_conv2_bias)
             leakyReluLayer('Name', 'relu2')
        
            % TRAINABLE layers (experience-dependent)
            convolution2dLayer(9, 32, 'Name', 'conv3', 'Padding', 'same')
            reluLayer('Name', 'relu3')
            convolution2dLayer(9, 32, 'Name', 'conv4', 'Padding', 'same')
            reluLayer('Name', 'relu4')
            fullyConnectedLayer(1024, 'Name', 'fc1')
            reluLayer('Name', 'relu5')
            fullyConnectedLayer(10, 'Name', 'fc_output')
            softmaxLayer('Name', 'softmax')
        ];
        
        temp_net = dlnetwork(temp_layers);
        
        % Compute new gradient
        new_gradient = computeFilterRF_LindseyMethod(temp_net, 'conv2', filter_idx);
        %new_error = sum((new_gradient(:) - target_gradient(:)).^2);
        %testing error with normalized gradients
        norm_new = (new_gradient - mean(new_gradient(:))) / std(new_gradient(:));
        pattern_error = sum((norm_new(:) - norm_target(:)).^2);
        deviation_penalty = lambda * sum((temp_weights(:) - initial_weights(:)).^2);
        new_error = pattern_error + deviation_penalty;
        % If better, keep it
        if new_error < error
            current_conv2_weights = temp_weights;
            net = temp_net;
            fprintf('  -> Keeping update (improvement: %.6f)\n', error - new_error);
        end
    end
    
    optimized_weights = current_conv2_weights;
    % Compute what gradient these weights currently produce
    current_gradient = computeFilterRF_LindseyMethod(net, 'conv2', filter_idx);
    
    % Calculate scaling factor needed to match target brightness
    %target_mean = mean(target_gradient(:));
    %current_mean = mean(current_gradient(:));
    %brightness_scale = target_mean / current_mean;
    current_max=max(abs(current_conv2_weights(:)));
    desired_max=0.01;
    brightness_scale=desired_max/current_max;
    % Scale the weights by this factor
    optimized_weights = optimized_weights * brightness_scale;
    % Redefine net to include normalized weights
    temp_layers = [
            imageInputLayer([32 32 1], 'Name', 'input', 'Normalization', 'none')
            convolution2dLayer(9, 32, 'Name', 'conv1', 'Padding', 'same', ...
                'Weights', net.Layers(2).Weights, 'Bias', net.Layers(2).Bias)
            reluLayer('Name', 'relu1')
            convolution2dLayer(9, 1, 'Name', 'conv2', 'Padding', 'same', ...
                'Weights', optimized_weights, 'Bias', current_conv2_bias)
             leakyReluLayer('Name', 'relu2')
        
            % TRAINABLE layers (experience-dependent)
            convolution2dLayer(9, 32, 'Name', 'conv3', 'Padding', 'same')
            reluLayer('Name', 'relu3')
            convolution2dLayer(9, 32, 'Name', 'conv4', 'Padding', 'same')
            reluLayer('Name', 'relu4')
            fullyConnectedLayer(1024, 'Name', 'fc1')
            reluLayer('Name', 'relu5')
            fullyConnectedLayer(10, 'Name', 'fc_output')
            softmaxLayer('Name', 'softmax')
        ];
    net=dlnetwork(temp_layers);

    fprintf('Optimization complete!\n');
    final_gradient = computeFilterRF_LindseyMethod(net, 'conv2', filter_idx);

    %visualize
    figure;
    imagesc(final_gradient);
    colormap gray;
    title('Final Gradient');
end
optimized_conv2_weights18=simpleGradientOptimization(retNet8, "C:\Users\leozi\OneDrive\Desktop\Research\rf_dataset500\train\center_surround\cs_train_0011.png")

%% 
save('optimized_conv2_weights17.mat', 'optimized_conv2_weights17');

%% comparing weights values to find differences.

good_versions = [13];
bad_versions = [12];

fprintf('Version | Type | Mean    | Std     | Min     | Max     | Range\n');
fprintf('--------|------|---------|---------|---------|---------|--------\n');

% Check good versions
for v = good_versions
    filename = sprintf('optimized_conv2_weights%d.mat', v);
    loaded = load(filename);
    varname = sprintf('optimized_conv2_weights%d', v);
    w = loaded.(varname);
    
    fprintf('%7d | GOOD | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f\n', ...
        v, mean(w(:)), std(w(:)), min(w(:)), max(w(:)), max(w(:))-min(w(:)));
end

% Check bad versions
for v = bad_versions
    filename = sprintf('optimized_conv2_weights%d.mat', v);
    loaded = load(filename);
    varname = sprintf('optimized_conv2_weights%d', v);
    w = loaded.(varname);
    
    fprintf('%7d | BAD  | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f\n', ...
        v, mean(w(:)), std(w(:)), min(w(:)), max(w(:)), max(w(:))-min(w(:)));
end
%% 
load("retnet8.mat","retNet8")
retNet8.Layers(4).Weights
%% histogram of weights
filename = 'optimized_conv2_weights17.mat';
load(filename);
w = optimized_conv2_weights17(:);

figure;
subplot(2,1,1);
violinplot(w, 'Orientation', 'horizontal');
title(filename);
xlabel('Weight Value');
set(gca, "FontSize", 14)
%% making methods figure

function generateOptimizationFigureImages(net, target_image_path)
    % Simple function - give it network and target, get 5 figures
    %
    % Inputs:
    %   net - Your network with initial conv2 weights
    %   target_image_path - Path to target center-surround image
    
    % Load target
    target_img = imread(target_image_path);
    if size(target_img, 3) == 3
        target_img = rgb2gray(target_img);
    end
    target = double(target_img) / 255;
    target = target - mean(target(:));
    
    % Get initial weights
    initial_weights = net.Layers(4).Weights;
    initial_bias = net.Layers(4).Bias;
    conv1_weights = net.Layers(2).Weights;
    conv1_bias = net.Layers(2).Bias;
    
    % Compute initial RF
    initial_rf = computeFilterRF_LindseyMethod(net, 'conv2', 1);
    target = imresize(target, size(initial_rf));
    
    % Compute initial loss
    loss_initial = computeLoss(initial_rf, target, initial_weights, initial_weights);
    
    % Create perturbed weights (one iteration)
    perturbed_weights = initial_weights + randn(size(initial_weights)) * 0.5;
    
    % Build temp network with perturbed weights
    temp_layers = [
        imageInputLayer([32 32 1], 'Name', 'input', 'Normalization', 'none')
        convolution2dLayer(9, 32, 'Name', 'conv1', 'Padding', 'same', ...
            'Weights', conv1_weights, 'Bias', conv1_bias)
        reluLayer('Name', 'relu1')
        convolution2dLayer(9, 1, 'Name', 'conv2', 'Padding', 'same', ...
            'Weights', perturbed_weights, 'Bias', initial_bias)
        leakyReluLayer('Name', 'relu2')
        convolution2dLayer(9, 32, 'Name', 'conv3', 'Padding', 'same')
        reluLayer('Name', 'relu3')
        convolution2dLayer(9, 32, 'Name', 'conv4', 'Padding', 'same')
        reluLayer('Name', 'relu4')
        fullyConnectedLayer(1024, 'Name', 'fc1')
        reluLayer('Name', 'relu5')
        fullyConnectedLayer(10, 'Name', 'fc_output')
        softmaxLayer('Name', 'softmax')
    ];
    temp_net = dlnetwork(temp_layers);
    
    % Compute perturbed RF
    perturbed_rf = computeFilterRF_LindseyMethod(temp_net, 'conv2', 1);
    
    % Compute perturbed loss
    loss_perturbed = computeLoss(perturbed_rf, target, perturbed_weights, initial_weights);
    
    % Display 5 figures
    figure;
    imagesc(target);
    colormap('gray');
    axis square off;
    colorbar;
    title('Target RF');
    
    figure;
    imagesc(initial_weights(:,:,1,1));
    colormap('gray');
    axis square off;
    colorbar;
    title('Initial Weights');
    
    figure;
    imagesc(initial_rf);
    colormap('gray');
    axis square off;
    colorbar;
    title(sprintf('Initial RF (Loss = %.4f)', loss_initial));
    
    figure;
    imagesc(perturbed_weights(:,:,1,1));
    colormap('gray');
    axis square off;
    colorbar;
    title('Perturbed Weights');
    
    figure;
    imagesc(perturbed_rf);
    colormap('gray');
    axis square off;
    colorbar;
    title(sprintf('Perturbed RF (Loss = %.4f)', loss_perturbed));
    
    fprintf('Initial loss:   %.6f\n', loss_initial);
    fprintf('Perturbed loss: %.6f\n', loss_perturbed);
end

function loss = computeLoss(rf, target, weights, initial_weights)
    norm_rf = (rf - mean(rf(:))) / std(rf(:));
    norm_target = (target - mean(target(:))) / std(target(:));
    lambda = 0.01;
    pattern_error = sum((norm_rf(:) - norm_target(:)).^2);
    deviation_penalty = lambda * sum((weights(:) - initial_weights(:)).^2);
    loss = pattern_error + deviation_penalty;
end

load("retnet8.mat","retNet8");
generateOptimizationFigureImages(retNet8, "C:\Users\leozi\OneDrive\Desktop\Research\rf_dataset500\train\center_surround\cs_train_0011.png")

%% 
% Simple weight visualization
load('optimized_conv2_weights17.mat');

% Visualize first slice (9x9 from first conv1 channel connection)
figure;
imagesc(optimized_conv2_weights17(:,:,1,1));
colormap('gray');
colorbar;
axis square;
title('Optimized Conv2 Weights (slice 1)');

