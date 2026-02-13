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
sgtitle(sprintf('Receptive Fields (Lindsey Method) - %s - %s', inputname(1), layerName));
end

load("retnet8.mat","retNet8")
genetic_conv1_weights = retNet8.Layers(2).Weights;
genetic_conv1_bias = retNet8.Layers(2).Bias;
genetic_conv2_weights = retNet8.Layers(4).Weights;  
genetic_conv2_bias = retNet8.Layers(4).Bias;
%load optimized reverse engineered weights
%load("optimized_conv2_weights4.mat","optimized_conv2_weights4")

rng(2)
layers = [
    imageInputLayer([32 32 1], 'Name', 'input')
    %retina-net
    convolution2dLayer(9, 32, 'Name', 'conv1', 'Padding', 'same') 
    reluLayer('Name', 'relu1')
    convolution2dLayer(9, 1, 'Name', 'conv2', 'Padding', 'same','WeightL2Factor',1e-3)  %bottleneck
    leakyReluLayer('Name', 'relu2')

    %vvs-net
    convolution2dLayer(9,32,'Name', 'conv3', 'Padding', 'same')
    reluLayer('Name', 'relu3')
    convolution2dLayer(9,32,'Name','conv4', 'Padding', 'same')
    reluLayer('Name','relu4')

    fullyConnectedLayer(1024,'Name', 'fc1')
    reluLayer('Name', 'relu5')
    fullyConnectedLayer(10, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
];


newnet=dlnetwork(layers);

load("retNet21.mat","retNet21")
visualizeFilters_LindseyMethod(retNet48, "conv4", 32)

%% for figure
function visualizeGradientAscentWithGradient(net, layerName, filterIndex, varargin)
% VISUALIZEGRADIENTASCENTWITHGRADIENT - Show the gradient explicitly
%
% Creates a 3-panel or 4-panel figure showing:
%   1. Initial gray image
%   2. Gradient heat map (what the network computed)
%   3. Final RF (after applying gradient)
%   4. Activation values

p = inputParser;
addParameter(p, 'InputSize', [32, 32, 1], @isnumeric);
addParameter(p, 'StepSize', 1.0, @isnumeric);
addParameter(p, 'SpatialPos', 'center', @ischar);
parse(p, varargin{:});

inputSize = p.Results.InputSize;
stepSize = p.Results.StepSize;
spatialPos = p.Results.SpatialPos;

% Initialize gray image
inputImg = dlarray(0.5 * ones([inputSize, 1], 'single'), 'SSCB');

% Get initial activation
[activation_initial, ~] = dlfeval(@lindseyLossFcn, net, inputImg, layerName, filterIndex, spatialPos);

% Compute gradient (this is the key computation!)
[~, gradients] = dlfeval(@lindseyLossFcn, net, inputImg, layerName, filterIndex, spatialPos);

% Normalize gradient (like Lindsey method)
gradNorm = sqrt(sum(gradients.^2, 'all'));
if gradNorm > 1e-5
    gradients_normalized = gradients / (gradNorm + 1e-5);
else
    gradients_normalized = gradients;
end

% Apply gradient
inputImg_updated = inputImg + stepSize * gradients_normalized;

% Get final activation
[activation_final, ~] = dlfeval(@lindseyLossFcn, net, inputImg_updated, layerName, filterIndex, spatialPos);

% Extract for visualization
initial_img = extractdata(squeeze(inputImg(:, :, 1, 1)));
gradient_img = extractdata(squeeze(gradients_normalized(:, :, 1, 1)));
final_img = extractdata(squeeze(inputImg_updated(:, :, 1, 1)));

% Post-process final RF (like Lindsey)
final_rf = postProcessRF(final_img);

% Create figure
figure('Position', [100, 100, 1400, 400]);

% Panel 1: Initial gray image
subplot(1, 3, 1);
imagesc(initial_img);
colormap(gca, gray);
axis off;
title(sprintf('Initial Image\n(Gray)\nActivation: %.3f', extractdata(activation_initial)), ...
    'FontSize', 11, 'FontWeight', 'bold');
colorbar;
clim([0, 1]);

% Panel 2: Gradient heat map
% subplot(1, 4, 2);
% imagesc(gradient_img);
% colormap(gca, jet);  % HEATMAP COLOR
% axis off;
% title(sprintf('Gradient\n∇ Activation'), ...
%     'FontSize', 11, 'FontWeight', 'bold');
% colorbar;
% % For standard heatmap, you might want to NOT center at zero:
% % caxis([min(gradient_img(:)), max(gradient_img(:))]);
% % OR keep it centered to show positive/negative:
% clim([-max(abs(gradient_img(:))), max(abs(gradient_img(:)))]);

% Panel 3: Raw updated image
subplot(1, 3, 2);
imagesc(final_img);
colormap(gca, gray);
axis off;
title(sprintf('Updated Image\n(Gray + Gradient ∇ Activation)\nActivation: %.3f', extractdata(activation_final)), ...
    'FontSize', 11, 'FontWeight', 'bold');
colorbar;
clim([0, 1]);

% Panel 4: Post-processed RF
subplot(1, 3, 3);
imagesc(final_rf);
colormap(gca, gray);
axis off;
title(sprintf('Final GRF\n(Normalized)'), ...
    'FontSize', 11, 'FontWeight', 'bold');
colorbar;
clim([0, 1]);

sgtitle(sprintf('Gradient-Based Receptive Field Computation'), 'FontSize', 14, 'FontWeight', 'bold');
end

function rf = postProcessRF(rawImg)
% Post-process exactly like Lindsey paper
rf = rawImg - mean(rawImg(:));
rf_std = std(rf(:));
if rf_std > 1e-5
    rf = rf / rf_std;
end
rf = rf * 0.1;
rf = rf + 0.5;
rf = max(0, min(1, rf));
end

load('retnet13.mat', 'retNet13');
visualizeGradientAscentWithGradient(retNet13, 'conv2', 1);

%% test for activations and gradients in conv3,4 leading to GRF that is all gray

function [rf, diagnostics] = computeFilterRF_Diagnostic(net, layerName, filterIndex, varargin)
% COMPUTEFILTERRF_DIAGNOSTIC - Diagnostic version of Lindsey method
% Returns both the RF visualization AND raw gradient information

% Parse inputs
p = inputParser;
addParameter(p, 'InputSize', [32, 32, 1], @isnumeric);
addParameter(p, 'StepSize', 1.0, @isnumeric);
addParameter(p, 'SpatialPos', 'center', @ischar);
parse(p, varargin{:});

inputSize = p.Results.InputSize;
stepSize = p.Results.StepSize;
spatialPos = p.Results.SpatialPos;

% Initialize input exactly like the paper: gray image (0.5)
inputImg = dlarray(0.5 * ones([inputSize, 1], 'single'), 'SSCB');

% Forward pass - check layer activation magnitude BEFORE gradient computation
layerOutput = forward(net, inputImg, 'Outputs', layerName);
layerOutputData = extractdata(layerOutput);

% Store forward pass diagnostics
diagnostics.layer_activation_mean = mean(abs(layerOutputData(:)));
diagnostics.layer_activation_max = max(abs(layerOutputData(:)));
diagnostics.layer_activation_std = std(layerOutputData(:));
diagnostics.layer_activation_filter = layerOutputData(:,:,filterIndex,1); % spatial map of this filter
diagnostics.center_activation = layerOutputData(round(size(layerOutputData,1)/2), ...
                                                round(size(layerOutputData,2)/2), ...
                                                filterIndex, 1);
fprintf('  [%s] Filter %d center activation: %.6f\n', layerName, filterIndex, diagnostics.center_activation);
fprintf('  [%s] Layer activation mean: %.6f, max: %.6f\n', layerName, ...
    diagnostics.layer_activation_mean, diagnostics.layer_activation_max);

% Single gradient step
[loss, gradients] = dlfeval(@lindseyLossFcn, net, inputImg, layerName, filterIndex, spatialPos);

% Store raw gradient diagnostics BEFORE normalization
gradData = extractdata(gradients);
diagnostics.raw_grad_norm = sqrt(sum(gradients.^2, 'all'));
diagnostics.raw_grad_norm = extractdata(diagnostics.raw_grad_norm);
diagnostics.raw_grad_mean = mean(abs(gradData(:)));
diagnostics.raw_grad_max = max(abs(gradData(:)));
diagnostics.loss_value = extractdata(loss);
diagnostics.below_threshold = diagnostics.raw_grad_norm <= 1e-5;

fprintf('  [%s] Filter %d raw gradient norm: %.10f\n', layerName, filterIndex, diagnostics.raw_grad_norm);
fprintf('  [%s] Filter %d loss value: %.10f\n', layerName, filterIndex, diagnostics.loss_value);
% Check what fraction of the 9x9 neighborhood around center is positive
[H, W, C, B] = size(layerOutputData);
center_h = round(H/2);
center_w = round(W/2);
half_kernel = 4; % for 9x9 kernel
neighborhood = layerOutputData(...
    max(1,center_h-half_kernel):min(H,center_h+half_kernel), ...
    max(1,center_w-half_kernel):min(W,center_w+half_kernel), :, :);
diagnostics.neighborhood_positive_fraction = mean(neighborhood(:) > 0);
fprintf('  [%s] Fraction of 9x9 neighborhood positive: %.4f\n', ...
    layerName, diagnostics.neighborhood_positive_fraction);
if diagnostics.below_threshold
    fprintf('  *** WARNING: Gradient norm below threshold (%.2e <= 1e-5)! RF will be gray. ***\n', ...
        diagnostics.raw_grad_norm);
else
    fprintf('  [%s] Gradient norm above threshold, RF should be visible\n', layerName);
end

% Normalize gradients
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
rf = rf * 0.1;
rf = rf + 0.5;
rf = max(0, min(1, rf));

diagnostics.rf_std = rf_std;
diagnostics.rf_is_gray = rf_std <= 1e-5;
end


% Load one of the affected retNets
load('retnet43.mat', 'retNet43');  % replace with whichever shows all-gray conv4
net = retNet62;
inputSize = net.Layers(1).InputSize;

fprintf('\n========== DIAGNOSTIC RUN ==========\n');
fprintf('\n--- Checking conv2 ---\n');
[rf2, diag2] = computeFilterRF_Diagnostic(net, 'conv2', 1, 'InputSize', inputSize);

fprintf('\n--- Checking conv3 ---\n');
[rf3, diag3] = computeFilterRF_Diagnostic(net, 'conv3', 1, 'InputSize', inputSize);

fprintf('\n--- Checking conv4 ---\n');
[rf4, diag4] = computeFilterRF_Diagnostic(net, 'conv4', 1, 'InputSize', inputSize);

% Summary
fprintf('\n========== SUMMARY ==========\n');
fprintf('Layer   | Grad Norm      | Center Act  | Below Thresh?\n');
fprintf('conv2   | %.10f | %.6f    | %d\n', diag2.raw_grad_norm, diag2.center_activation, diag2.below_threshold);
fprintf('conv3   | %.10f | %.6f    | %d\n', diag3.raw_grad_norm, diag3.center_activation, diag3.below_threshold);
fprintf('conv4   | %.10f | %.6f    | %d\n', diag4.raw_grad_norm, diag4.center_activation, diag4.below_threshold);
%% 
