clear,clc
%CNN from scratch
rng(1)
layers = [
    imageInputLayer([28 28 1], 'Name', 'input')
    convolution2dLayer(5, 6, 'Name', 'conv1')      % Just 6 filters - easy to visualize
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    fullyConnectedLayer(10, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
];

net=dlnetwork(layers);
exinputsize=net.Layers(1).InputSize;

trainPath='C:\Users\leozi\OneDrive\Desktop\Research\mnist_png\train';
testPath='C:\Users\leozi\OneDrive\Desktop\Research\mnist_png\test';
imds_train=imageDatastore(trainPath,'IncludeSubfolders',true,'LabelSource','foldernames');
imds_test=imageDatastore(testPath,"IncludeSubfolders",true, 'LabelSource','foldernames');
[imds_train_split, imds_val_split] = splitEachLabel(imds_train, 0.8, 'randomized');

% No need for color preprocessing since we're using grayscale input
imds_train_resized = augmentedImageDatastore(exinputsize(1:2), imds_train_split);
imds_test_resized = augmentedImageDatastore(exinputsize(1:2), imds_test);
imds_val_resized = augmentedImageDatastore(exinputsize(1:2), imds_val_split);

batchSize=128;
maxEpochs=5;
learningRate=1e-4;

options=trainingOptions('sgdm','MiniBatchSize',batchSize, ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate',learningRate, 'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'ValidationData',imds_val_resized, ...
    'ValidationFrequency',50,...
    'ValidationPatience',4,...
    'Plots','training-progress','Metrics','accuracy');
%% Train
scratchNetlite = trainnet(imds_train_resized, net,"crossentropy",options);
%% Save
save('scratchlitemnist.mat','scratchNetlite')
%% Evaluate
scores=minibatchpredict(scratchNetlite,imds_test_resized);
classes=categories(imds_test.Labels);
predlabels=scores2label(scores,classes);
testlabels=imds_test.Labels;
accuracy=testnet(scratchNetlite,imds_test_resized,"accuracy")

% Display confusion matrix
figure;
confusionchart(testlabels, predlabels);
title('Confusion Matrix');

