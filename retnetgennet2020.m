%% OVERNIGHT TRAINING SCRIPT - 40 Models Total
% Phase 1: 20 retNet models (seeds 3-22)
% Phase 2: 20 genNet models with reverse-engineered weights

%% ====================================================================
%  PHASE 1: Train 20 Standard retNet Models
%  ====================================================================

fprintf('\n========================================\n');
fprintf('PHASE 1: Training 20 retNet models\n');
fprintf('========================================\n\n');

for seed_idx = 3:22
    model_num = seed_idx + 20; % retnet23, retnet24, ..., retnet42
    
    fprintf('\n--- Training retNet%d (seed %d) ---\n', model_num, seed_idx);
    
    try
        % Set random seed
        rng(seed_idx)
        
        % Build network
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
        
        net=dlnetwork(layers);
        
        % Scale conv2 weights by 0.1
        conv2_idx = find(strcmp({net.Layers.Name}, 'conv2'));
        conv2 = net.Layers(conv2_idx);
        conv2.Weights = 0.1 * conv2.Weights;
        net = replaceLayer(net, 'conv2', conv2);
        
        exinputsize=net.Layers(1).InputSize;
        
        % Load data
        trainPath='C:\Users\leozi\OneDrive\Desktop\Research\cifar10\cifar10\train';
        testPath='C:\Users\leozi\OneDrive\Desktop\Research\cifar10\cifar10\test';
        imds_train=imageDatastore(trainPath,'IncludeSubfolders',true,'LabelSource','foldernames');
        imds_test=imageDatastore(testPath,"IncludeSubfolders",true, 'LabelSource','foldernames');
        [imds_train_split, imds_val_split] = splitEachLabel(imds_train, 0.8, 'randomized');
        
        imds_train_resized = augmentedImageDatastore(exinputsize(1:2), imds_train_split,"ColorPreprocessing","rgb2gray");
        imds_test_resized = augmentedImageDatastore(exinputsize(1:2), imds_test,"ColorPreprocessing","rgb2gray");
        imds_val_resized = augmentedImageDatastore(exinputsize(1:2), imds_val_split,"ColorPreprocessing","rgb2gray");
        
        % Training options
        batchSize=32;
        maxEpochs=20;
        learningRate=0.0001;
        
        options=trainingOptions('rmsprop','MiniBatchSize',batchSize, ...
            'MaxEpochs',maxEpochs, ...
            'InitialLearnRate',learningRate, 'Shuffle','every-epoch', ...
            'Verbose', true, ...
            'ValidationData', imds_val_resized, ...
            'ValidationFrequency', 50, ...
            'ValidationPatience', 10, ...
            'L2Regularization', 1e-5, ...
            'Plots', 'training-progress', ...
            'Metrics', 'accuracy');
        
        % Train
        fprintf('Starting training for retNet%d...\n', model_num);
        trainedNet = trainnet(imds_train_resized, net,"crossentropy",options);
        
        % Save with dynamic variable name
        save_filename = sprintf('retnet%d.mat', model_num);
        save_varname = sprintf('retNet%d', model_num);
        eval([save_varname ' = trainedNet;']);
        save(save_filename, save_varname);
        
        fprintf('✓ Successfully saved %s\n', save_filename);
        
    catch ME
        fprintf('✗ ERROR training retNet%d: %s\n', model_num, ME.message);
        fprintf('Continuing to next model...\n');
    end
    
    % Clear variables to free memory
    clear net trainedNet imds_train imds_test imds_train_split imds_val_split
    clear imds_train_resized imds_test_resized imds_val_resized
end

fprintf('\n========================================\n');
fprintf('PHASE 1 COMPLETE\n');
fprintf('========================================\n\n');

%% ====================================================================
%  PHASE 2: Train 20 genNet Models with Reverse-Engineered Weights
%  ====================================================================

fprintf('\n========================================\n');
fprintf('PHASE 2: Training 20 genNet models\n');
fprintf('========================================\n\n');

% Load retNet8 once (used as base for all reverse engineering)
load('retnet8.mat', 'retNet8');
genetic_conv1_weights = retNet8.Layers(2).Weights;
genetic_conv1_bias = retNet8.Layers(2).Bias;
genetic_conv2_bias = retNet8.Layers(4).Bias;

for model_idx = 1:20
    weight_num = 17 + model_idx; % optimized_conv2_weights18, 19, ..., 37
    img_num = 127 + model_idx; % cs_train_0128.png, ..., 0147.png
    gennet_num = 827 + model_idx; % gennet828, ..., gennet847
    
    fprintf('\n--- Creating genNet%d (weights%d, image%d) ---\n', ...
        gennet_num, weight_num, img_num);
    
    try
        %% Step 1: Reverse engineer weights
        fprintf('Step 1: Reverse engineering weights...\n');
        
        target_image_path = sprintf('C:\\Users\\leozi\\OneDrive\\Desktop\\Research\\rf_dataset500\\train\\center_surround\\cs_train_%04d.png', img_num);
        
        % Run optimization
        optimized_weights = simpleGradientOptimization(retNet8, target_image_path);
        
        % Save with dynamic variable name
        weight_save_filename = sprintf('optimized_conv2_weights%d.mat', weight_num);
        weight_save_varname = sprintf('optimized_conv2_weights%d', weight_num);
        eval([weight_save_varname ' = optimized_weights;']);
        save(weight_save_filename, weight_save_varname);
        
        fprintf('✓ Saved %s\n', weight_save_filename);
        
        %% Step 2: Build genNet architecture
        fprintf('Step 2: Building genNet architecture...\n');
        
        rng(model_idx + 2) % Use different seed for each genNet
        
        layers = [
            imageInputLayer([32 32 1], 'Name', 'input', 'Normalization', 'none')
            
            % GENETIC layers from retNet8
            convolution2dLayer(9, 32, 'Name', 'conv1', 'Padding', 'same', ...
                'Weights', genetic_conv1_weights, ...
                'Bias', genetic_conv1_bias, ...
                'WeightLearnRateFactor', 0.2, 'BiasLearnRateFactor', 0)
            reluLayer('Name', 'relu1')
            
            convolution2dLayer(9, 1, 'Name', 'conv2', 'Padding', 'same', ...
                'Weights', optimized_weights, ...
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
        
        net = dlnetwork(layers);
        exinputsize = net.Layers(1).InputSize;
        
        %% Step 3: Load and prepare data
        fprintf('Step 3: Loading data...\n');
        
        trainPath='C:\Users\leozi\OneDrive\Desktop\Research\cifar10\cifar10\train';
        testPath='C:\Users\leozi\OneDrive\Desktop\Research\cifar10\cifar10\test';
        imds_train=imageDatastore(trainPath,'IncludeSubfolders',true,'LabelSource','foldernames');
        imds_test=imageDatastore(testPath,"IncludeSubfolders",true, 'LabelSource','foldernames');
        [imds_train_split, imds_val_split] = splitEachLabel(imds_train, 0.8, 'randomized');
        
        imds_train_resized = augmentedImageDatastore(exinputsize(1:2), imds_train_split,"ColorPreprocessing","rgb2gray");
        imds_test_resized = augmentedImageDatastore(exinputsize(1:2), imds_test,"ColorPreprocessing","rgb2gray");
        imds_val_resized = augmentedImageDatastore(exinputsize(1:2), imds_val_split,"ColorPreprocessing","rgb2gray");
        
        %% Step 4: Train
        fprintf('Step 4: Training genNet%d...\n', gennet_num);
        
        batchSize=32;
        maxEpochs=20;
        learningRate=0.0001;
        
        options=trainingOptions('rmsprop','MiniBatchSize',batchSize, ...
            'MaxEpochs',maxEpochs, ...
            'InitialLearnRate',learningRate, 'Shuffle','every-epoch', ...
            'Verbose', true, ...
            'ValidationData', imds_val_resized, ...
            'ValidationFrequency', 50, ...
            'ValidationPatience', 10, ...
            'L2Regularization', 1e-5, ...
            'Plots', 'training-progress', ...
            'Metrics', 'accuracy');
        
        trainedNet = trainnet(imds_train_resized, net,"crossentropy",options);
        
        %% Step 5: Save genNet
        gennet_save_filename = sprintf('gennet%d.mat', gennet_num);
        gennet_save_varname = sprintf('genNet%d', gennet_num);
        eval([gennet_save_varname ' = trainedNet;']);
        save(gennet_save_filename, gennet_save_varname);
        
        fprintf('✓ Successfully saved %s\n', gennet_save_filename);
        
    catch ME
        fprintf('✗ ERROR creating genNet%d: %s\n', gennet_num, ME.message);
        fprintf('Continuing to next model...\n');
    end
    
    % Clear variables to free memory
    clear net trainedNet optimized_weights imds_train imds_test
    clear imds_train_split imds_val_split imds_train_resized imds_test_resized imds_val_resized
end

fprintf('\n========================================\n');
fprintf('PHASE 2 COMPLETE\n');
fprintf('========================================\n\n');

fprintf('\n========================================\n');
fprintf('ALL TRAINING COMPLETE!\n');
fprintf('Total models trained: 40\n');
fprintf('- retNet models: retnet23.mat through retnet42.mat\n');
fprintf('- genNet models: gennet828.mat through gennet847.mat\n');
fprintf('========================================\n');

%% ====================================================================
%  HELPER FUNCTIONS (copied from your code)
%  ====================================================================

function rf = computeFilterRF_LindseyMethod(net, layerName, filterIndex, varargin)
    % COMPUTEFILTERRF_LINDSEYMETHOD - Exact replication of Lindsey et al. method
    
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
    rf = rf * 0.1;
    rf = rf + 0.5;
    rf = max(0, min(1, rf));
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
    loss = mean(layerOutput(pos_x, pos_y, filterIndex, :), 'all');
    
    % Compute gradients
    gradients = dlgradient(loss, inputImg);
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
    
    % Extract current weights from net
    current_conv2_weights = net.Layers(4).Weights;
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
        
        % Normalize for comparison
        norm_target = (target_gradient - mean(target_gradient(:))) / std(target_gradient(:));
        norm_current = (current_gradient - mean(current_gradient(:))) / std(current_gradient(:));
        
        % Compute error with regularization
        lambda = 0.01;
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
    
    % Calculate scaling factor
    current_max = max(abs(current_conv2_weights(:)));
    desired_max = 0.01;
    brightness_scale = desired_max / current_max;
    
    % Scale the weights
    optimized_weights = optimized_weights * brightness_scale;
    
    % Redefine net with normalized weights
    temp_layers = [
        imageInputLayer([32 32 1], 'Name', 'input', 'Normalization', 'none')
        convolution2dLayer(9, 32, 'Name', 'conv1', 'Padding', 'same', ...
            'Weights', net.Layers(2).Weights, 'Bias', net.Layers(2).Bias)
        reluLayer('Name', 'relu1')
        convolution2dLayer(9, 1, 'Name', 'conv2', 'Padding', 'same', ...
            'Weights', optimized_weights, 'Bias', current_conv2_bias)
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
    net = dlnetwork(temp_layers);
    
    fprintf('Optimization complete!\n');
    final_gradient = computeFilterRF_LindseyMethod(net, 'conv2', filter_idx);
    
    % Visualize
    figure;
    imagesc(final_gradient);
    colormap gray;
    title('Final Gradient');
end
%% 
function visualizeFilters_MultipleNets(netNames, layerName, filterIndex)
% VISUALIZEFILTERS_MULTIPLENETS - Visualize same filter across multiple networks
%
% Usage:
%   visualizeFilters_MultipleNets({'retnet23', 'retnet24', ..., 'retnet42'}, 'conv2', 1)
%   visualizeFilters_MultipleNets({'gennet828', 'gennet829', ..., 'gennet847'}, 'conv2', 1)
%
% Inputs:
%   netNames - Cell array of network variable names (e.g., {'retnet23', 'retnet24', ...})
%   layerName - Layer to visualize (e.g., 'conv2')
%   filterIndex - Which filter/channel to visualize (e.g., 1 for conv2 bottleneck)

numNets = length(netNames);

% Create figure
figure('Position', [100, 100, 1200, 800]);

for i = 1:numNets
    fprintf('Processing %s (%d/%d)...\n', netNames{i}, i, numNets);
    
    % Load the network
    load([netNames{i} '.mat'], netNames{i});
    net = eval(netNames{i});
    
    % Get input size from network
    inputSize = net.Layers(1).InputSize;
    
    % Compute RF using Lindsey method
    rf = computeFilterRF_LindseyMethod(net, layerName, filterIndex, 'InputSize', inputSize);
    
    % Plot in grid
    subplot(4, 5, i);  % 4 rows x 5 columns = 20 networks
    imagesc(rf);
    colormap gray;
    axis off;
    axis equal;
    title(netNames{i}, 'FontSize', 8, 'Interpreter', 'none');
    
    % Clear network to save memory
    clear net
end

sgtitle(sprintf('Receptive Fields - %s (Filter %d) Across Networks', layerName, filterIndex), ...
    'FontSize', 12, 'FontWeight', 'bold');
end



% For your 20 retNet models (retnet23 through retnet42)
retnet_names = arrayfun(@(x) sprintf('retNet%d', x), 23:42, 'UniformOutput', false);
visualizeFilters_MultipleNets(retnet_names, 'conv2', 1);

% For your 20 genNet models (gennet828 through gennet847)
gennet_names = arrayfun(@(x) sprintf('genNet%d', x), 828:847, 'UniformOutput', false);
visualizeFilters_MultipleNets(gennet_names, 'conv2', 1);
