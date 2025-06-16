% Check the label encoding
fprintf('First 20 YTrain values: %s\n', mat2str(double(YTrain(1:20)')));
fprintf('First 20 YTest values: %s\n', mat2str(double(YTest(1:20)')));
fprintf('YTrain range: [%d, %d]\n', min(double(YTrain)), max(double(YTrain)));

% Show a few samples with corrected labels
figure;
for i = 1:6
    subplot(2,3,i);
    imshow(XTrain(:,:,1,i));
    title(sprintf('Label: %d, Actual digit: %d', double(YTrain(i)), double(YTrain(i))-1));
end
%% test rng
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
testNet=dlnetwork(layers)
firstConvLayer=testNet.Layers(2);
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

