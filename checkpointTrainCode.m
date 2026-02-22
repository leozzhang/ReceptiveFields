%% CHECKPOINT TRAINING SCRIPT - 5 retNets + 5 genNets
% Trains networks with checkpoints every 100 iterations
% to examine receptive field development over training

%% ====================================================================
%  PHASE 1: Train 5 retNets with checkpoints (rng 3-7)
%  ====================================================================

fprintf('\n========================================\n');
fprintf('PHASE 1: Training 5 retNet models with checkpoints\n');
fprintf('========================================\n\n');

retnet_seeds = 3:7;
retnet_nums = 43:47;

for i = 1:5
    seed = retnet_seeds(i);
    net_num = retnet_nums(i);
    
    fprintf('\n--- Training retNet%d (seed %d) ---\n', net_num, seed);
    
    try
        % Set random seed
        rng(seed)
        
        % Create checkpoint directory
        checkpoint_dir = sprintf('checkpoints_retNet%d', net_num);
        if ~exist(checkpoint_dir, 'dir')
            mkdir(checkpoint_dir);
        end
        
        % Build network
        layers = [
            imageInputLayer([32 32 1], 'Name', 'input')
            %retina-net
            convolution2dLayer(9, 32, 'Name', 'conv1', 'Padding', 'same')
            reluLayer('Name', 'relu1')
            convolution2dLayer(9, 1, 'Name', 'conv2', 'Padding', 'same','WeightL2Factor',1e-3)
            leakyReluLayer('Name', 'relu2')
            %vvs-net
            convolution2dLayer(9,32,'Name', 'conv3', 'Padding', 'same')
            reluLayer('Name', 'relu3')
            convolution2dLayer(9,32,'Name','conv4', 'Padding', 'same')
            reluLayer('Name','relu4')
            convolution2dLayer(9,32,"Name","conv5","Padding","same")
            reluLayer("Name","reluextra")
            convolution2dLayer(9,32,"Name","conv6","Padding","same")
            reluLayer("Name","reluextra2")
            convolution2dLayer(9,32,"Name","conv7","Padding","same")
            reluLayer("Name","reluextra3")
            fullyConnectedLayer(1024, 'Name', 'fc1')
            reluLayer('Name', 'relu5')
            fullyConnectedLayer(10, 'Name', 'fc_output')
            softmaxLayer('Name', 'softmax')
        ];
        
        net = dlnetwork(layers);
        
        % Scale conv2 weights by 0.1
        conv2_idx = find(strcmp({net.Layers.Name}, 'conv2'));
        conv2 = net.Layers(conv2_idx);
        conv2.Weights = 0.1 * conv2.Weights;
        net = replaceLayer(net, 'conv2', conv2);
        
        exinputsize = net.Layers(1).InputSize;
        
        % Load data
        trainPath='C:\Users\leozi\OneDrive\Desktop\Research\cifar10\cifar10\train';
        testPath='C:\Users\leozi\OneDrive\Desktop\Research\cifar10\cifar10\test';
        imds_train=imageDatastore(trainPath,'IncludeSubfolders',true,'LabelSource','foldernames');
        imds_test=imageDatastore(testPath,"IncludeSubfolders",true, 'LabelSource','foldernames');
        [imds_train_split, imds_val_split] = splitEachLabel(imds_train, 0.8, 'randomized');
        
        imds_train_resized = augmentedImageDatastore(exinputsize(1:2), imds_train_split,"ColorPreprocessing","rgb2gray");
        imds_test_resized = augmentedImageDatastore(exinputsize(1:2), imds_test,"ColorPreprocessing","rgb2gray");
        imds_val_resized = augmentedImageDatastore(exinputsize(1:2), imds_val_split,"ColorPreprocessing","rgb2gray");
        
        % Training options with checkpoints
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
            'Metrics', 'accuracy', ...
            'CheckpointPath', checkpoint_dir, ...
            'CheckpointFrequency', 100, ...
            'CheckpointFrequencyUnit', 'iteration');
        
        % Train
        fprintf('Starting training for retNet%d with checkpoints...\n', net_num);
        trainedNet = trainnet(imds_train_resized, net,"crossentropy",options);
        
        % Save final model
        save_filename = sprintf('retnet%d_checkpointed.mat', net_num);
        save_varname = sprintf('retNet%d', net_num);
        eval([save_varname ' = trainedNet;']);
        save(save_filename, save_varname);
        
        fprintf('✓ Successfully saved %s\n', save_filename);
        fprintf('✓ Checkpoints saved in %s\n', checkpoint_dir);
        
    catch ME
        fprintf('✗ ERROR training retNet%d: %s\n', net_num, ME.message);
        fprintf('Continuing to next model...\n');
    end
    
    % Clear variables to free memory
    clear net trainedNet imds_train imds_test imds_train_split imds_val_split
    clear imds_train_resized imds_test_resized imds_val_resized
end

fprintf('\n========================================\n');
fprintf('PHASE 1 COMPLETE\n');
fprintf('========================================\n\n');

%%

fprintf('\n========================================\n');
fprintf('PHASE 2: Training 5 genNet models with checkpoints\n');
fprintf('========================================\n\n');

gennet_nums = 888:892;
weight_nums = 38:42;

% Load retNet8 weights once
load('retnet8.mat', 'retNet8');
genetic_conv1_weights = retNet8.Layers(2).Weights;
genetic_conv1_bias = retNet8.Layers(2).Bias;
genetic_conv2_bias = retNet8.Layers(4).Bias;

for i = 1:5
    gennet_num = gennet_nums(i);
    weight_num = weight_nums(i);
    
    fprintf('\n--- Training genNet%d (weights %d) ---\n', gennet_num, weight_num);
    
    try
        % Create checkpoint directory
        checkpoint_dir = sprintf('checkpoints_genNet%d', gennet_num);
        if ~exist(checkpoint_dir, 'dir')
            mkdir(checkpoint_dir);
        end
        
        % Load optimized uniform weights
        load_varname = sprintf('optimized_conv2_uniform%d', weight_num);
        load_filename = sprintf('optimized_conv2_uniform%d.mat', weight_num);
        load(load_filename, load_varname);
        optimized_weights = eval(load_varname);
        
        % Build genNet architecture
        rng(i + 100) % Different seed for each genNet
        
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
            
            % TRAINABLE layers
            convolution2dLayer(9, 32, 'Name', 'conv3', 'Padding', 'same')
            reluLayer('Name', 'relu3')
            convolution2dLayer(9, 32, 'Name', 'conv4', 'Padding', 'same')
            reluLayer('Name', 'relu4')
            convolution2dLayer(9,32,"Name","conv5","Padding","same")
            reluLayer("Name","reluextra")
            convolution2dLayer(9,32,"Name","conv6","Padding","same")
            reluLayer("Name","reluextra2")
            convolution2dLayer(9,32,"Name","conv7","Padding","same")
            reluLayer("Name","reluextra3")
            fullyConnectedLayer(1024, 'Name', 'fc1')
            reluLayer('Name', 'relu5')
            fullyConnectedLayer(10, 'Name', 'fc_output')
            softmaxLayer('Name', 'softmax')
        ];
        
        net = dlnetwork(layers);
        exinputsize = net.Layers(1).InputSize;
        
        % Load data
        trainPath='C:\Users\leozi\OneDrive\Desktop\Research\cifar10\cifar10\train';
        testPath='C:\Users\leozi\OneDrive\Desktop\Research\cifar10\cifar10\test';
        imds_train=imageDatastore(trainPath,'IncludeSubfolders',true,'LabelSource','foldernames');
        imds_test=imageDatastore(testPath,"IncludeSubfolders",true, 'LabelSource','foldernames');
        [imds_train_split, imds_val_split] = splitEachLabel(imds_train, 0.8, 'randomized');
        
        imds_train_resized = augmentedImageDatastore(exinputsize(1:2), imds_train_split,"ColorPreprocessing","rgb2gray");
        imds_test_resized = augmentedImageDatastore(exinputsize(1:2), imds_test,"ColorPreprocessing","rgb2gray");
        imds_val_resized = augmentedImageDatastore(exinputsize(1:2), imds_val_split,"ColorPreprocessing","rgb2gray");
        
        % Training options with checkpoints
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
            'Metrics', 'accuracy', ...
            'CheckpointPath', checkpoint_dir, ...
            'CheckpointFrequency', 100, ...
            'CheckpointFrequencyUnit', 'iteration');
        
        % Train
        fprintf('Starting training for genNet%d with checkpoints...\n', gennet_num);
        trainedNet = trainnet(imds_train_resized, net,"crossentropy",options);
        
        % Save final model
        gennet_save_filename = sprintf('gennet%d_checkpointed.mat', gennet_num);
        gennet_save_varname = sprintf('genNet%d', gennet_num);
        eval([gennet_save_varname ' = trainedNet;']);
        save(gennet_save_filename, gennet_save_varname);
        
        fprintf('✓ Successfully saved %s\n', gennet_save_filename);
        fprintf('✓ Checkpoints saved in %s\n', checkpoint_dir);
        
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
fprintf('Total networks trained: 10\n');
fprintf('- retNet models: retnet43-47 (checkpointed)\n');
fprintf('- genNet models: gennet888-892 (checkpointed)\n');
fprintf('Checkpoints saved in folders: checkpoints_retNet## and checkpoints_genNet##\n');
fprintf('========================================\n');
%% visualize

%% CHECKPOINT RECEPTIVE FIELD VISUALIZATION
% Visualizes how receptive fields develop over training
% by loading every 3rd checkpoint and computing Lindsey GRFs

function visualize_checkpoint_RFs(network_name, layer_name, filter_idx)
% VISUALIZE_CHECKPOINT_RFS - Visualize RF development across checkpoints
%
% Usage:
%   visualize_checkpoint_RFs('retNet43', 'conv2', 1)
%   visualize_checkpoint_RFs('genNet888', 'conv3', 5)
%
% Inputs:
%   network_name - Name of network (e.g., 'retNet43', 'genNet888')
%   layer_name - Layer to visualize (e.g., 'conv2', 'conv3', 'conv4')
%   filter_idx - Which filter to visualize (default: 1)

if nargin < 3
    filter_idx = 1;
end

% Get checkpoint directory
checkpoint_dir = sprintf('checkpoints_%s', network_name);

if ~exist(checkpoint_dir, 'dir')
    error('Checkpoint directory not found: %s', checkpoint_dir);
end

% Get list of checkpoint files
checkpoint_files = dir(fullfile(checkpoint_dir, 'net_checkpoint__*.mat'));

if isempty(checkpoint_files)
    error('No checkpoint files found in %s', checkpoint_dir);
end

% Sort by iteration number (extract from filename)
iterations = zeros(length(checkpoint_files), 1);
for i = 1:length(checkpoint_files)
    % Extract iteration number from filename like "net_checkpoint__200__2026_02_21__17_49_53.mat"
    % The pattern is: net_checkpoint__[NUMBER]__[rest]
    tokens = regexp(checkpoint_files(i).name, 'net_checkpoint__(\d+)__', 'tokens');
    if ~isempty(tokens)
        iterations(i) = str2double(tokens{1}{1});
    end
end
[~, sort_idx] = sort(iterations);
checkpoint_files = checkpoint_files(sort_idx);
iterations = iterations(sort_idx);

% Select every 3rd checkpoint
selected_indices = 1:3:length(checkpoint_files);
num_checkpoints = length(selected_indices);

fprintf('Found %d total checkpoints, visualizing %d (every 3rd)\n', ...
    length(checkpoint_files), num_checkpoints);

% Compute grid size
grid_cols = ceil(sqrt(num_checkpoints));
grid_rows = ceil(num_checkpoints / grid_cols);

% Create figure
figure('Position', [100, 100, 200*grid_cols, 200*grid_rows]);

% Process each selected checkpoint
rfs = cell(num_checkpoints, 1);
checkpoint_iterations = zeros(num_checkpoints, 1);

for i = 1:num_checkpoints
    idx = selected_indices(i);
    checkpoint_file = checkpoint_files(idx);
    checkpoint_iterations(i) = iterations(idx);
    
    fprintf('Processing checkpoint %d/%d: %s (iteration %d)...\n', ...
        i, num_checkpoints, checkpoint_file.name, iterations(idx));
    
    % Load checkpoint
    checkpoint_path = fullfile(checkpoint_file.folder, checkpoint_file.name);
    load(checkpoint_path, 'net');
    
    % Get input size
    inputSize = net.Layers(1).InputSize;
    
    % Compute RF using Lindsey method
    rf = computeFilterRF_LindseyMethod(net, layer_name, filter_idx, 'InputSize', inputSize);
    rfs{i} = rf;
    
    % Plot
    subplot(grid_rows, grid_cols, i);
    imagesc(rf);
    colormap gray;
    axis off;
    axis equal;
    title(sprintf('Iter %d', checkpoint_iterations(i)), 'FontSize', 10);
    
    clear net
end

sgtitle(sprintf('%s - %s Filter %d: RF Development', network_name, layer_name, filter_idx), ...
    'FontSize', 14, 'FontWeight', 'bold');


end

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

visualize_checkpoint_RFs('retNet46', 'conv1', 1);

%% 

%% VISUALIZE INITIAL CONV2 WEIGHTS - Failed vs Successful retNets

% Define networks
failed_nets = [43, 45, 46, 47, 48, 49, 51, 54, 59, 60];
successful_nets = [44, 50, 52, 53, 55, 56, 57, 58, 61, 62];
seeds = 3:22; % rng seeds corresponding to retNet43-62

% Storage
failed_weights = cell(length(failed_nets), 1);
successful_weights = cell(length(successful_nets), 1);

%% Extract initial conv2 weights for failed networks
fprintf('Extracting initial weights for FAILED networks...\n');
for i = 1:length(failed_nets)
    net_num = failed_nets(i);
    seed = seeds(net_num - 42); % retNet43 uses seed 3, etc.
    
    fprintf('  retNet%d (seed %d)\n', net_num, seed);
    
    % Recreate initial network
    rng(seed);
    layers = [
        imageInputLayer([32 32 1], 'Name', 'input')
        convolution2dLayer(9, 32, 'Name', 'conv1', 'Padding', 'same')
        reluLayer('Name', 'relu1')
        convolution2dLayer(9, 1, 'Name', 'conv2', 'Padding', 'same','WeightL2Factor',1e-3)
        leakyReluLayer('Name', 'relu2')
        convolution2dLayer(9,32,'Name', 'conv3', 'Padding', 'same')
        reluLayer('Name', 'relu3')
        convolution2dLayer(9,32,'Name','conv4', 'Padding', 'same')
        reluLayer('Name','relu4')
        fullyConnectedLayer(1024, 'Name', 'fc1')
        reluLayer('Name', 'relu5')
        fullyConnectedLayer(10, 'Name', 'fc_output')
        softmaxLayer('Name', 'softmax')
    ];
    net = dlnetwork(layers);
    
    % Get conv2 weights and apply 0.1 scaling
    conv2_idx = find(strcmp({net.Layers.Name}, 'conv2'));
    conv2_weights = net.Layers(conv2_idx).Weights;
    conv2_weights = 0.1 * conv2_weights; % Apply the scaling
    
    failed_weights{i} = conv2_weights;
end

%% Extract initial conv2 weights for successful networks
fprintf('\nExtracting initial weights for SUCCESSFUL networks...\n');
for i = 1:length(successful_nets)
    net_num = successful_nets(i);
    seed = seeds(net_num - 42);
    
    fprintf('  retNet%d (seed %d)\n', net_num, seed);
    
    % Recreate initial network
    rng(seed);
    layers = [
        imageInputLayer([32 32 1], 'Name', 'input')
        convolution2dLayer(9, 32, 'Name', 'conv1', 'Padding', 'same')
        reluLayer('Name', 'relu1')
        convolution2dLayer(9, 1, 'Name', 'conv2', 'Padding', 'same','WeightL2Factor',1e-3)
        leakyReluLayer('Name', 'relu2')
        convolution2dLayer(9,32,'Name', 'conv3', 'Padding', 'same')
        reluLayer('Name', 'relu3')
        convolution2dLayer(9,32,'Name','conv4', 'Padding', 'same')
        reluLayer('Name','relu4')
        fullyConnectedLayer(1024, 'Name', 'fc1')
        reluLayer('Name', 'relu5')
        fullyConnectedLayer(10, 'Name', 'fc_output')
        softmaxLayer('Name', 'softmax')
    ];
    net = dlnetwork(layers);
    
    % Get conv2 weights and apply 0.1 scaling
    conv2_idx = find(strcmp({net.Layers.Name}, 'conv2'));
    conv2_weights = net.Layers(conv2_idx).Weights;
    conv2_weights = 0.1 * conv2_weights;
    
    successful_weights{i} = conv2_weights;
end

%% Visualize - show sum across input channels (9x9 spatial pattern)
figure('Position', [100, 100, 1600, 900]);

% Total number of networks to display
total_nets = length(failed_nets) + length(successful_nets);
grid_cols = 10; % 10 columns
grid_rows = ceil(total_nets / grid_cols); % Calculate needed rows

% Plot failed networks first
for i = 1:length(failed_nets)
    w = failed_weights{i};
    % Sum across all 32 input channels to get overall 9x9 spatial pattern
    w_sum = sum(w, 3);
    
    subplot(grid_rows, grid_cols, i);
    imagesc(w_sum);
    colormap gray;
    colorbar;
    axis square;
    title(sprintf('FAIL: %d', failed_nets(i)), 'Color', 'red', 'FontWeight', 'bold', 'FontSize', 10);
end

% Plot successful networks after failed ones
for i = 1:length(successful_nets)
    w = successful_weights{i};
    w_sum = sum(w, 3);
    
    plot_idx = length(failed_nets) + i;
    subplot(grid_rows, grid_cols, plot_idx);
    imagesc(w_sum);
    colormap gray;
    colorbar;
    axis square;
    title(sprintf('OK: %d', successful_nets(i)), 'FontSize', 8);
end

sgtitle('Initial Conv2 Weights (summed across 32 input channels) - Failed vs Successful', ...
    'FontSize', 14, 'FontWeight', 'bold');

%% Also visualize histograms of weight distributions
figure('Position', [100, 100, 1400, 600]);

subplot(1, 2, 1);
hold on;
for i = 1:length(failed_nets)
    w = failed_weights{i};
    histogram(w(:), 50, 'DisplayName', sprintf('retNet%d', failed_nets(i)), ...
        'FaceAlpha', 0.3);
end
hold off;
title('Failed Networks - Weight Distribution');
xlabel('Weight Value');
ylabel('Count');
legend('Location', 'best');

subplot(1, 2, 2);
hold on;
for i = 1:min(6, length(successful_nets)) % Just show first 6 for clarity
    w = successful_weights{i};
    histogram(w(:), 50, 'DisplayName', sprintf('retNet%d', successful_nets(i)), ...
        'FaceAlpha', 0.3);
end
hold off;
title('Successful Networks - Weight Distribution (first 6)');
xlabel('Weight Value');
ylabel('Count');
legend('Location', 'best');
%% 
%% Check what retNet8 conv1 produces on gray image

% Load retNet8
load('retnet8.mat', 'retNet8');

% Create gray input
inputSize = retNet8.Layers(1).InputSize;
gray_input = dlarray(0.5 * ones([inputSize, 1], 'single'), 'SSCB');

% Get conv1 output (before relu1)
conv1_output = forward(retNet8, gray_input, 'Outputs', 'conv1');
conv1_data = extractdata(conv1_output);

% Statistics
fprintf('\n=== retNet8 Conv1 Output on Gray Image ===\n');
fprintf('Shape: %s\n', mat2str(size(conv1_data)));
fprintf('Mean: %.6f\n', mean(conv1_data(:)));
fprintf('Std: %.6f\n', std(conv1_data(:)));
fprintf('Min: %.6f\n', min(conv1_data(:)));
fprintf('Max: %.6f\n', max(conv1_data(:)));
fprintf('Fraction positive: %.4f\n', mean(conv1_data(:) > 0));

% Visualize a few of the 32 conv1 output channels
figure('Position', [100, 100, 1200, 400]);
for i = 1:min(8, size(conv1_data, 3))
    subplot(2, 4, i);
    imagesc(conv1_data(:,:,i,1));
    colorbar;
    colormap gray;
    axis square;
    title(sprintf('Conv1 Channel %d', i), 'FontSize', 10);
end
sgtitle('retNet8 Conv1 Output on Gray Image (First 8 Channels)', 'FontSize', 12);

% Also check after relu1
relu1_output = forward(retNet8, gray_input, 'Outputs', 'relu1');
relu1_data = extractdata(relu1_output);

fprintf('\n=== retNet8 After ReLU1 ===\n');
fprintf('Mean: %.6f\n', mean(relu1_data(:)));
fprintf('Fraction positive: %.4f\n', mean(relu1_data(:) > 0));


% Load trained retNet43
load('retnet43.mat', 'retNet43');

% Gray input
inputSize = retNet43.Layers(1).InputSize;
gray_input = dlarray(0.5 * ones([inputSize, 1], 'single'), 'SSCB');

% Get conv2 output
conv2_output = forward(retNet43, gray_input, 'Outputs', 'conv2');
conv2_data = extractdata(conv2_output);

fprintf('\n=== TRAINED retNet43 Conv2 Output on Gray Image ===\n');
fprintf('Fraction positive: %.4f\n', mean(conv2_data(:) > 0));
fprintf('Fraction negative: %.4f\n', mean(conv2_data(:) < 0));
fprintf('Mean: %.6f\n', mean(conv2_data(:)));
fprintf('Min: %.6f, Max: %.6f\n', min(conv2_data(:)), max(conv2_data(:)));
fprintf('Std: %.6f\n', std(conv2_data(:)));
fprintf('Fraction positive: %.4f\n', mean(conv2_data(:) > 0));
fprintf('Fraction zero: %.4f\n', mean(conv2_data(:) == 0));
fprintf('Fraction negative: %.4f\n', mean(conv2_data(:) < 0));

% Compare to what we expect from population diagnostic
fprintf('\n(Population diagnostic showed genNet act_pos_frac at conv2 = 0.0)\n');

% Visualize
figure('Position', [100, 100, 800, 400]);

subplot(1, 2, 1);
imagesc(conv2_data(:,:,1,1));
colorbar;
colormap gray;
axis square;
title('Trained genNet888 Conv2 Output');

subplot(1, 2, 2);
histogram(conv2_data(:), 50);
xlabel('Activation Value');
ylabel('Count');
title('Conv2 Output Distribution');
xline(0, 'r--', 'LineWidth', 2);

sgtitle('Trained genNet888 Conv2 Output on Gray Image', 'FontSize', 12);