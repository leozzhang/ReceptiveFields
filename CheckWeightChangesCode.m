clear,clc
% Load both networks
%originalNet = imagePretrainedNetwork("googlenet");
%load('googlenetmnist.mat','trainedNet');
rng(1)
layers = [
    imageInputLayer([28 28 1], 'Name', 'input')
    convolution2dLayer(5, 6, 'Name', 'conv1')      % Just 6 filters - easy to visualize
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    fullyConnectedLayer(10, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
];

originalNet = dlnetwork(layers);
load('scratchlitecifar.mat','sNetliteCIFAR');
% Function to calculate weight differences for each layer
layerNames = {originalNet.Layers.Name};
weightChanges = [];

for i = 1:length(originalNet.Layers)
    if isprop(originalNet.Layers(i), 'Weights') && ~isempty(originalNet.Layers(i).Weights)
        % Calculate the norm of weight differences
        origWeights = originalNet.Layers(i).Weights;
        newWeights = sNetliteCIFAR.Layers(i).Weights;
        
        if isequal(size(origWeights), size(newWeights))
            weightDiff = norm(origWeights(:) - newWeights(:));
            weightNorm = norm(origWeights(:));
            relativeChange = weightDiff / weightNorm * 100; % Percentage change
            
            fprintf('Layer %d (%s): %.2f%% change\n', i, layerNames{i}, relativeChange);
            weightChanges(end+1) = relativeChange;
        else
            fprintf('Layer %d (%s): Structure changed\n', i, layerNames{i});
        end
    end
end
%% 
rng(1)
layers = [
    imageInputLayer([28 28 3], 'Name', 'input')
    convolution2dLayer(3, 32, 'Name', 'conv1')      % Just 6 filters - easy to visualize
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    fullyConnectedLayer(10, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
];

originalNet = dlnetwork(layers);
load('scratchcifar.mat','sNetCIFAR');
function visualizeLayerComparison(originalNet, trainedNet, layerNum)
    origLayer = originalNet.Layers(layerNum);
    newLayer = trainedNet.Layers(layerNum);
    
    if isprop(origLayer, 'Weights')
        origWeights = origLayer.Weights;
        newWeights = newLayer.Weights;
        
        if ndims(origWeights) == 4 % Convolutional layer
            numFiltersToShow = 6;  % Show only first 6
            
            figure('Position', [100, 100, 1200, 400]);
            for i = 1:numFiltersToShow
                % Original filters (top row)
                subplot(2, numFiltersToShow, i);
                imagesc(origWeights(:,:,1,i));
                colormap gray; axis off;
                title(sprintf('Orig F%d', i), 'FontSize', 8);
                
                % New filters (bottom row)
                subplot(2, numFiltersToShow, i + numFiltersToShow);
                imagesc(newWeights(:,:,1,i));
                colormap gray; axis off;
                title(sprintf('Trained F%d', i), 'FontSize', 8);
            end
            sgtitle(sprintf('Layer %d (%s) - First %d Filters: Before vs After', ...
                layerNum, origLayer.Name, numFiltersToShow), 'Interpreter', 'none');
        end
    end
end
function visualizeLayerComparisonRGB(originalNet, trainedNet, layerNum)
    origLayer = originalNet.Layers(layerNum);
    newLayer = trainedNet.Layers(layerNum);
    
    if isprop(origLayer, 'Weights')
        origWeights = origLayer.Weights;
        newWeights = newLayer.Weights;
        
        if ndims(origWeights) == 4 % Convolutional layer
            numFiltersToShow = 6;  % Show fewer filters since we're showing RGB
            
            figure('Position', [100, 100, 1500, 800]);
            for i = 1:numFiltersToShow
                for channel = 1:3
                    % Original filters - RGB channels
                    subplot(6, numFiltersToShow, (channel-1)*numFiltersToShow + i);
                    imagesc(origWeights(:,:,channel,i));
                    colormap gray; axis off;
                    title(sprintf('Orig F%d Ch%d', i, channel), 'FontSize', 8);
                    
                    % Trained filters - RGB channels  
                    subplot(6, numFiltersToShow, (channel+2)*numFiltersToShow + i);
                    imagesc(newWeights(:,:,channel,i));
                    colormap gray; axis off;
                    title(sprintf('Train F%d Ch%d', i, channel), 'FontSize', 8);
                end
            end
            sgtitle('RGB Channels: Before vs After Training', 'Interpreter', 'none');
        end
    end
end
visualizeLayerComparison(originalNet, sNet3liteCIFAR, 2)
%% debugging
% Compare raw weight statistics
before_weights = squeeze(originalNet.Layers(2).Weights);
after_weights = squeeze(scratchNet.Layers(2).Weights);

fprintf('Before weights - Min: %.4f, Max: %.4f, Range: %.4f\n', ...
    min(before_weights(:)), max(before_weights(:)), max(before_weights(:)) - min(before_weights(:)));
fprintf('After weights - Min: %.4f, Max: %.4f, Range: %.4f\n', ...
    min(after_weights(:)), max(after_weights(:)), max(after_weights(:)) - min(after_weights(:)));

% Look at one specific filter in detail
fprintf('\nFilter 1 before:\n');
disp(before_weights(:,:,1));
fprintf('\nFilter 1 after:\n');
disp(after_weights(:,:,1));