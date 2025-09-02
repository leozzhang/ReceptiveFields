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


load("retnet8.mat","retNet8")
visualizeFilters_LindseyMethod(retNet8, "conv2", 1)

%% Classify and softmax score
% Extract RFs from your trained network
load("retnet8.mat","retNet8")
net = retNet8;
layerName = "conv4";
numFilters = 32;  % or however many you want to classify

% Extract all RFs as arrays
rfs = [];
for i = 1:numFilters
    rf = computeFilterRF_LindseyMethod(net, layerName, i, 'InputSize', [32, 32, 1]);
    
    % Resize from 32x32 to 25x25 to match your classifier
    rf_resized = imresize(rf, [25, 25]);
    
    % Store in batch format
    rfs(i, :, :) = rf_resized;
end

% Now classify all your CNN filters
for i = 1:numFilters
    single_rf = squeeze(rfs(i, :, :));
    scores = minibatchpredict(RFClassifyNet, single_rf);
    predicted_label = scores2label(scores, ["center_surround", "oriented"]);
    fprintf('Filter %d: %s (CS: %.3f, OR: %.3f)\n', i, predicted_label, scores(1), scores(2));
end
%% 

%noise_img = rand(25, 25);  % Random values 0-1

%scores = minibatchpredict(RFClassifyNet, noise_img);
%fprintf('Random noise: CS=%.3f, OR=%.3f\n', scores(1), scores(2));

% On your training data
cs_img = imread('C:\Users\leozi\OneDrive\Desktop\Research\rf_dataset500\train\center_surround\cs_train_0001.png');
or_img = imread('C:\Users\leozi\OneDrive\Desktop\Research\rf_dataset500\train\oriented\or_train_0001.png');

% Convert to double for stats
cs_img = double(cs_img);
or_img = double(or_img);

fprintf('CS: mean=%.2f, std=%.2f\n', mean(cs_img(:)), std(cs_img(:)));
fprintf('OR: mean=%.2f, std=%.2f\n', mean(or_img(:)), std(or_img(:)));

% On your CNN filters
cnn_rf = computeFilterRF_LindseyMethod(retNet8, "conv3", 1);

% Resize and normalize to 0–255 range (stay in double)
cnn_rf_resized = imresize(cnn_rf, [25, 25]);
cnn_rf_norm = (cnn_rf_resized - min(cnn_rf_resized(:))) / ...
              (max(cnn_rf_resized(:)) - min(cnn_rf_resized(:)));
cnn_rf_double = 255 * cnn_rf_norm;

fprintf('CNN: mean=%.2f, std=%.2f\n', mean(cnn_rf_double(:)), std(cnn_rf_double(:)));
