%clear,clc

%set conv weights as previously trained network weights
%load("retnet8.mat","retNet8")
%genetic_conv1_weights = retNet8.Layers(2).Weights;
%genetic_conv1_bias = retNet8.Layers(2).Bias;
%genetic_conv2_weights = retNet8.Layers(4).Weights;  
%genetic_conv2_bias = retNet8.Layers(4).Bias;
%load optimized reverse engineered weights
%load("optimized_conv2_weights14.mat","optimized_conv2_weights14")
%load("optimized_conv2_weights17.mat","optimized_conv2_weights17")
%bothconv2=cat(4, optimized_conv2_weights17, -1*optimized_conv2_weights14);
%bothbias=reshape([genetic_conv2_bias;genetic_conv2_bias],[1, 1, 2]);

rng(2)
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

% Scale conv2 weights
conv2_idx = find(strcmp({net.Layers.Name}, 'conv2'));
conv2 = net.Layers(conv2_idx);
conv2.Weights = 0.1 * conv2.Weights;
net = replaceLayer(net, 'conv2', conv2);

exinputsize=net.Layers(1).InputSize;

trainPath='C:\Users\leozi\OneDrive\Desktop\Research\cifar10\cifar10\train';
testPath='C:\Users\leozi\OneDrive\Desktop\Research\cifar10\cifar10\test';
imds_train=imageDatastore(trainPath,'IncludeSubfolders',true,'LabelSource','foldernames');
imds_test=imageDatastore(testPath,"IncludeSubfolders",true, 'LabelSource','foldernames');
[imds_train_split, imds_val_split] = splitEachLabel(imds_train, 0.8, 'randomized');

imds_train_resized = augmentedImageDatastore(exinputsize(1:2), imds_train_split,"ColorPreprocessing","rgb2gray");
imds_test_resized = augmentedImageDatastore(exinputsize(1:2), imds_test,"ColorPreprocessing","rgb2gray");
imds_val_resized = augmentedImageDatastore(exinputsize(1:2), imds_val_split,"ColorPreprocessing","rgb2gray");
batchSize=32;
maxEpochs=20;
learningRate=0.0001;

options=trainingOptions('rmsprop','MiniBatchSize',batchSize, ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate',learningRate, 'Shuffle','every-epoch', ...
    'Verbose', true, ...
    'ValidationData', imds_val_resized, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 10, ...  % Stop early if overfitting
    'L2Regularization', 1e-5, ...  
    'Plots', 'training-progress', ...
    'Metrics', 'accuracy');
%% Train
retNet22 = trainnet(imds_train_resized, net,"crossentropy",options);
%% Save
save('retnet22.mat','retNet22')
%% Evaluate
scores=minibatchpredict(genNet8,imds_test_resized);
classes=categories(imds_test.Labels);
scores(1)
predlabels=scores2label(scores,classes);
testlabels=imds_test.Labels;
accuracy=testnet(genNet8,imds_test_resized,"accuracy")

% Display confusion matrix
figure;
confusionchart(testlabels, predlabels);

title('Confusion Matrix');
%% center surround score
load('scratch3litecifar.mat','sNet3liteCIFAR');
function final_score=corr2csscore(filter)
    [h,w]=size(filter);
    best_score=0;
    center_radii = [0.6, 0.9, 1.2];
    surround_ratios = [1.8, 2.2];
    cx_range = 2:(w-1);  % Avoid edges
    cy_range = 2:(h-1);

    for center_r = center_radii
        for surr_ratio = surround_ratios
            surround_r=center_r*surr_ratio;
            for cx = cx_range
                for cy = cy_range
                    template=createCenterSurroundTemplate(h,w,cx,cy,center_r,surround_r);
                    c=corr2(template,filter);
                    current_score=abs(c);

                    if current_score>best_score
                        best_score = current_score;
                    end
                end
            end
        end
    end
    final_score=best_score;
end
weights = retNet2.Layers(4).Weights;
num_filters=size(weights,4); %4th dimension of weights contains num filters
cs_scores=zeros(1,num_filters);
for i=1:num_filters
    filter=weights(:,:,1,i);
    cs_scores(i)=corr2csscore(filter);
    fprintf('Filter %d: %.3f\n',i, cs_scores(i));
end

disp(strjoin(string(cs_scores), sprintf('\t')))
