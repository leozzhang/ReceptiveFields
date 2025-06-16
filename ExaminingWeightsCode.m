clear,clc
load('scratchmnist.mat','scratchNet'); %load transfer learning network
%trainedNet=imagePretrainedNetwork("googlenet"); %load pretrained network
firstConvLayer=scratchNet.Layers(2);
fprintf('Layer name: %s\n', firstConvLayer.Name); %check we have the right layer

weights=firstConvLayer.Weights;

filtersToShow=weights(:,:,1,:); %take first channel, all 64 filters
filtersToShow=squeeze(filtersToShow); % remove singleton dimension

%visualize
figure
for i=1:6
    subplot(2,3,i);
    imagesc(filtersToShow(:,:,i));
    colormap gray;
    axis off;
    title(sprintf('Filter %d', i), 'FontSize', 8);
end
sgtitle(sprintf('%s Filters', firstConvLayer.Name), 'Interpreter', 'none');
%% debug
fprintf('Original weights size: %s\n', mat2str(size(weights)));
filtersToShow=weights(:,:,1,:); 
fprintf('After selecting first channel: %s\n', mat2str(size(filtersToShow)));
filtersToShow=squeeze(filtersToShow); 
fprintf('After squeeze: %s\n', mat2str(size(filtersToShow)));
fprintf('Number of filters to show: %d\n', size(filtersToShow,3));
