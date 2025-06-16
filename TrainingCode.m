clear,clc
networkused="googlenet";
net=imagePretrainedNetwork(networkused,"NumClasses",10);
exinputsize=net.Layers(1).InputSize;

%images
trainPath='C:\Users\leozi\OneDrive\Desktop\Research\mnist_png\train';
testPath='C:\Users\leozi\OneDrive\Desktop\Research\mnist_png\test';
imds_train=imageDatastore(trainPath,'IncludeSubfolders',true,'LabelSource','foldernames');
imds_test=imageDatastore(testPath,"IncludeSubfolders",true, 'LabelSource','foldernames');
[imds_train_split, imds_val_split] = splitEachLabel(imds_train, 0.8, 'randomized');
imds_train_resized = augmentedImageDatastore(exinputsize(1:2), imds_train_split, 'ColorPreprocessing', 'gray2rgb');
imds_test_resized = augmentedImageDatastore(exinputsize(1:2), imds_test, 'ColorPreprocessing', 'gray2rgb');
imds_val_resized = augmentedImageDatastore(exinputsize(1:2), imds_val_split, 'ColorPreprocessing', 'gray2rgb');

%training options
batchSize=128;
maxEpochs=2;
learningRate=1e-4;

options=trainingOptions('sgdm','MiniBatchSize',batchSize, ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate',learningRate, 'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'ValidationData',imds_val_resized, ...
    'ValidationFrequency',50,...
    'ValidationPatience',3,...
    'Plots','training-progress','Metrics','accuracy');
%% 

trainedNet = trainnet(imds_train_resized, net,"crossentropy",options);
%% 
save('googlenetmnist.mat','trainedNet')
%% evaluate
scores=minibatchpredict(trainedNet,imds_test_resized);
classes=categories(imds_test.Labels);
predlabels=scores2label(scores,classes);
testlabels=imds_test.Labels;
accuracy=testnet(trainedNet,imds_test_resized,"accuracy")

% Display confusion matrix
figure;
confusionchart(testlabels, predlabels);
title('Confusion Matrix');