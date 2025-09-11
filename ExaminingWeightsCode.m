clear,clc
load('retnet8.mat','retNet8'); %load transfer learning network

% Extract conv1 weights
conv1_layer = retNet8.Layers(4); % Assuming conv1 is layer 2
conv1_weights = conv1_layer.Weights; % Should be [9, 9, 1, 32]

% Visualize all 32 filters
figure('Position', [100, 100, 1200, 800]);
for i = 1:32
    subplot(6, 6, i);
    filter = conv1_weights(:,:,1,i);
    imagesc(filter);
    colormap gray;
    axis off;
    title(sprintf('Filter %d', i), 'FontSize', 8);
    % Normalize colormap for better visualization
    caxis([min(filter(:)), max(filter(:))]);
end
sgtitle('Conv1 Filters from Trained Network');
%% debug
fprintf('Original weights size: %s\n', mat2str(size(weights)));
filtersToShow=weights(:,:,1,:); 
fprintf('After selecting first channel: %s\n', mat2str(size(filtersToShow)));
filtersToShow=squeeze(filtersToShow); 
fprintf('After squeeze: %s\n', mat2str(size(filtersToShow)));
fprintf('Number of filters to show: %d\n', size(filtersToShow,3));
