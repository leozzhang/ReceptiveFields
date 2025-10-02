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
load("optimized_conv2_weights8.mat","optimized_conv2_weights8")

rng(2)
layers = [
        imageInputLayer([32 32 1], 'Name', 'input', 'Normalization', 'none')  % 32x32 like retNet8
        
        % GENETIC layers from retNet8
        convolution2dLayer(9, 32, 'Name', 'conv1', 'Padding', 'same', ...     % 9x9 like retNet8
                          'Weights', genetic_conv1_weights, ...
                          'Bias', genetic_conv1_bias, ...
                          'WeightLearnRateFactor', 0.2, 'BiasLearnRateFactor', 0)
        reluLayer('Name', 'relu1')
        
        convolution2dLayer(9, 1, 'Name', 'conv2', 'Padding', 'same', ...      % Bottleneck like retNet8
                          'Weights', optimized_conv2_weights8, ...
                          'Bias', genetic_conv2_bias, ...
                          'WeightLearnRateFactor', 0.2, 'BiasLearnRateFactor', 0)
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

newnet=dlnetwork(layers);

load("gennet86.mat","genNet86")
visualizeFilters_LindseyMethod(genNet86, "conv2", 1)
%% Classify and softmax score
% Extract RFs from your trained network
load("retnet8.mat","retNet8")
load("rfclassifynet.mat","RFClassifyNet")
net = retNet8;
layerName = "conv2";
numFilters = 1;  % or however many you want to classify

% Map each layer name to its receptive field size
rfSizes = containers.Map( ...
    {'conv1','conv2','conv3','conv4'}, ...
    [9, 17, 25, 33]);

% Extract all RFs as arrays
rfs = zeros(numFilters, 25, 25);  % final size for classifier

for i = 1:numFilters
    rf = computeFilterRF_LindseyMethod(net, layerName, i, 'InputSize', [32, 32, 1]);

    % Get receptive field size for this layer
    cropSize = rfSizes(layerName);

    % Crop center patch of size cropSize × cropSize
    center = floor(size(rf,1)/2) + 1;
    half = floor(cropSize/2);
    rf_cropped = rf(center-half:center+half, center-half:center+half);

    % Resize to 25×25 for classifier
    rf_resized = imresize(rf_cropped, [25, 25]);

    % Store in batch
    rfs(i, :, :) = rf_resized;
end

% Now classify all your CNN filters
for i = 1:numFilters
    single_rf = squeeze(rfs(i, :, :));
    
    % PREPROCESS to match training data distribution
    % Convert to same format as your training data
    single_rf_norm = (single_rf - mean(single_rf(:))) / std(single_rf(:));  % Z-score
    single_rf_scaled = single_rf_norm * 20 + 150;  % Match training: std=20, mean=150
    single_rf_final = uint8(max(0, min(255, single_rf_scaled)));  % Convert to uint8
    
    scores = minibatchpredict(RFClassifyNet, single_rf_final);
    predicted_label = scores2label(scores, ["center_surround", "oriented"]);
    fprintf('Filter %d: %s (CS: %.3f, OR: %.3f)\n', i, predicted_label, scores(1), scores(2));
end
%% 
% Create simple test patterns
blank = ones(25, 25) * 150;  % Solid gray
circle = blank; circle(8:17, 8:17) = 200;  % White square
lines = blank; lines(:, 12:13) = 200;  % Vertical lines

scores_blank = minibatchpredict(RFClassifyNet, uint8(blank));
scores_circle = minibatchpredict(RFClassifyNet, uint8(circle));
scores_lines = minibatchpredict(RFClassifyNet, uint8(lines));

fprintf('Blank: CS=%.3f, OR=%.3f\n', scores_blank(1), scores_blank(2));
fprintf('Circle: CS=%.3f, OR=%.3f\n', scores_circle(1), scores_circle(2));
fprintf('Lines: CS=%.3f, OR=%.3f\n', scores_lines(1), scores_lines(2));
%% 
input_test = dlarray(0.5 * ones([32, 32, 1, 1], 'single'), 'SSCB');

% Check each layer sequentially
layer_names = {'conv1', 'bn1', 'relu1', 'conv2'}; % adjust to your actual layer names

for i = 1:length(layer_names)
    out1 = forward(net1, input_test, 'Outputs', layer_names{i});
    out2 = forward(net2, input_test, 'Outputs', layer_names{i});
    diff = max(abs(out1(:) - out2(:)));
    fprintf('%s output difference: %f\n', layer_names{i}, diff);
end