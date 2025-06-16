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
    fullyConnectedLayer(84, 'Name', 'fc1')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(10, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
];

originalNet = dlnetwork(layers);
load('scratchmnist.mat','scratchNet');
% Function to calculate weight differences for each layer
layerNames = {originalNet.Layers.Name};
weightChanges = [];

for i = 1:length(originalNet.Layers)
    if isprop(originalNet.Layers(i), 'Weights') && ~isempty(originalNet.Layers(i).Weights)
        % Calculate the norm of weight differences
        origWeights = originalNet.Layers(i).Weights;
        newWeights = scratchNet.Layers(i).Weights;
        
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
function visualizeLayerComparison(originalNet, trainedNet, layerNum)
    origLayer = originalNet.Layers(layerNum);
    newLayer = trainedNet.Layers(layerNum);
    
    if isprop(origLayer, 'Weights')
        origWeights = origLayer.Weights;
        newWeights = newLayer.Weights;
        
        % For conv layers, show first few filters
        if ndims(origWeights) == 4  % Convolutional layer
            numFilters = min(16, size(origWeights, 4));
            
            figure('Position', [100, 100, 1200, 600]);
            for i = 1:numFilters
                % Original filters
                subplot(2, 6, i);
                imagesc(origWeights(:,:,1,i));
                colormap gray; axis off;
                title(sprintf('Orig F%d', i), 'FontSize', 8);
                
                % New filters
                subplot(2, 6, i + numFilters);
                imagesc(newWeights(:,:,1,i));
                colormap gray; axis off;
                title(sprintf('New F%d', i), 'FontSize', 8);
            end
            sgtitle(sprintf('Layer %d (%s) - Before vs After Training', layerNum, origLayer.Name), 'Interpreter', 'none');
        end
    end
end

visualizeLayerComparison(originalNet, scratchNet, 2)
%% for 1x1
function visualizeLayerComparison2(originalNet, trainedNet, layerNum)
    origLayer = originalNet.Layers(layerNum);
    newLayer = trainedNet.Layers(layerNum);
    
    if isprop(origLayer, 'Weights')
        origWeights = origLayer.Weights;
        newWeights = newLayer.Weights;
        
        % Check dimensions and filter size
        fprintf('Layer %d (%s): Weight dimensions = %s\n', layerNum, origLayer.Name, mat2str(size(origWeights)));
        
        % For conv layers, show first few filters
        if ndims(origWeights) == 4 % Convolutional layer
            numFilters = min(16, size(origWeights, 4));
            
            % Handle different filter sizes
            filterSize = size(origWeights, 1);
            if filterSize == 1
                % For 1x1 filters, show them as bars instead
                figure('Position', [100, 100, 1200, 400]);
                
                subplot(2, 1, 1);
                bar(squeeze(origWeights(1,1,1,1:numFilters)));
                title('Original Filters (1x1 shown as bars)', 'Interpreter', 'none');
                
                subplot(2, 1, 2);
                bar(squeeze(newWeights(1,1,1,1:numFilters)));
                title('New Filters (1x1 shown as bars)', 'Interpreter', 'none');
                
                sgtitle(sprintf('Layer %d (%s) - 1x1 Filters', layerNum, origLayer.Name), 'Interpreter', 'none');
            else
                % For larger filters, use enhanced visualization
                figure('Position', [100, 100, 1200, 600]);
                
                % Calculate global min/max for consistent scaling
                allWeights = [origWeights(:); newWeights(:)];
                globalMin = min(allWeights);
                globalMax = max(allWeights);
                
                for i = 1:numFilters
                    % Original filters
                    subplot(4, 8, i);
                    imagesc(origWeights(:,:,1,i), [globalMin, globalMax]);
                    colormap gray; axis off;
                    title(sprintf('Orig F%d', i), 'FontSize', 8);
                    
                    % New filters
                    subplot(4, 8, i + numFilters);
                    imagesc(newWeights(:,:,1,i), [globalMin, globalMax]);
                    colormap gray; axis off;
                    title(sprintf('New F%d', i), 'FontSize', 8);
                end
                
                sgtitle(sprintf('Layer %d (%s) - Before vs After Training', layerNum, origLayer.Name), 'Interpreter', 'none');
                
                % Add colorbar to show scale
                colorbar('Position', [0.92, 0.3, 0.02, 0.4]);
            end
        end
    end
end
visualizeLayerComparison2(originalNet, trainedNet,137)
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