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

load("retNet15.mat","retNet15")
visualizeFilters_LindseyMethod(newnet, "conv2", 1)

%% 
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
title(sprintf('Final RF\n(Post-processed)'), ...
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