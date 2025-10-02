%clear,clc
%CS score functions
function template = createCenterSurroundTemplate(h, w, cx, cy, center_r, surround_r)

    % CREATE A SIMPLE CENTER-SURROUND TEMPLATE
    % 
    % This function creates an "ideal" center-surround pattern that we'll
    % compare our CNN filters against
    %
    % Inputs:
    %   h, w = height and width of the template (same size as your filter)
    %   cx, cy = center position (where the center circle should be)
    %   center_r = radius of the positive center circle
    %   surround_r = radius of the negative surround ring
    %
    % Output:
    %   template = a matrix with +1 in center, -0.5 in surround, 0 elsewhere

    % Step 1: Create coordinate grids
    % This creates matrices X and Y where each position contains its coordinates
    [X, Y] = meshgrid(1:w, 1:h);
    
    % Step 2: Calculate distance from each pixel to the center point
    % This tells us how far each pixel is from (cx, cy)
    distances = sqrt((X - cx).^2 + (Y - cy).^2);
    
    % Step 3: Start with a blank template (all zeros)
    template = zeros(h, w);
    
    % Step 4: Make the center positive (+1)
    % Find all pixels that are within center_r distance from center
    center_mask = distances <= center_r;
    template(center_mask) = 1;  % Positive response in center
    
    % Step 5: Make the surround negative (-0.5)
    % Find pixels that are between center_r and surround_r distance from center
    surround_mask = (distances > center_r) & (distances <= surround_r);
    template(surround_mask) = -0.5;  % Negative response in surround
    
    % Step 6: Normalize the template so it has unit energy
    % This ensures fair comparison between different sized templates
    template = template / norm(template(:));
end
function score = computeTemplateMatch(filter, template)
    % FIXED VERSION - Now gives perfect score = 1 for identical patterns
    %
    % Inputs:
    %   filter = a CNN filter (5x5 matrix from your network)
    %   template = our ideal center-surround pattern (also 5x5)
    %
    % Output:
    %   score = similarity score between 0 and 1

    % Remove the mean from both (zero-center them)
    filter_centered = filter - mean(filter(:));
    template_centered = template - mean(template(:));
    
    % Compute normalized cross-correlation
    numerator = sum(filter_centered(:) .* template_centered(:));
    denominator = norm(filter_centered(:)) * norm(template_centered(:));
    
    % Handle edge case where one pattern is completely flat
    if denominator == 0
        score = 0;
    else
        correlation = numerator / denominator;
        score = abs(correlation);  % Take absolute value for ON/OFF invariance
    end
end
function [best_score, best_params] = findBestCenterSurroundMatch_optimized(filter)
    % OPTIMIZED and SIMPLIFIED - just returns the best similarity score
    
    [h, w] = size(filter);
    
    % OPTIMIZED for 5x5 filters MAY NEED TO CHANGE FOR DIFF NETS
    center_radii = [0.6, 0.9, 1.2];    
    surround_ratios = [1.8, 2.2];       
    
    % Position search - avoid edges
    cx_range = 2:(w-1);
    cy_range = 2:(h-1);
    
    best_score = 0;
    best_params = struct();
    
    for center_r = center_radii
        for surr_ratio = surround_ratios
            surround_r = center_r * surr_ratio;
            
            % Skip if surround too big
            if surround_r > min(h,w)/2.2
                continue;
            end
            
            for cx = cx_range
                for cy = cy_range
                    
                    template = createCenterSurroundTemplate(h, w, cx, cy, center_r, surround_r);
                    
                    % Skip templates that are too big
                    active_pixels = sum(template(:) ~= 0);
                    if active_pixels > 15  % Max 60% of 25 pixels
                        continue;
                    end
                    
                    score = computeTemplateMatch(filter, template);
                    
                    if score > best_score
                        best_score = score;
                        best_params = struct('center_x', cx, 'center_y', cy, ...
                            'center_radius', center_r, 'surround_radius', surround_r, ...
                            'active_pixels', active_pixels, 'template', template);
                    end
                end
            end
        end
    end
    
    % Simple output - just the score and basic parameters
    if ~isempty(fieldnames(best_params))
        fprintf('Best CS match: pos(%.1f,%.1f), c_r=%.1f, score=%.3f\n', ...
            best_params.center_x, best_params.center_y, ...
            best_params.center_radius, best_score);
    else
        fprintf('No valid center-surround match found\n');
    end
end
function cs_score = getCenterSurroundScore(filter)
    % SIMPLE FUNCTION: Just returns the center-surround score (0 to 1)
    % This is what you'll use most often
    
    [score, ~] = findBestCenterSurroundMatch_optimized(filter);
    cs_score = score;
end

%set conv weights as previously trained network weights
load("retnet8.mat","retNet8")
genetic_conv1_weights = retNet8.Layers(2).Weights;
genetic_conv1_bias = retNet8.Layers(2).Bias;
genetic_conv2_weights = retNet8.Layers(4).Weights;  
genetic_conv2_bias = retNet8.Layers(4).Bias;
%load optimized reverse engineered weights
load("optimized_conv2_weights8.mat","optimized_conv2_weights8")

rng(2)
layers = [
        imageInputLayer([32 32 1], 'Name', 'input', 'Normalization', 'none')  % 32x32 like retNet8
        
        % GENETIC layers from retNet8
        convolution2dLayer(9, 32, 'Name', 'conv1', 'Padding', 'same', ...     % 9x9 like retNet8
                          'Weights', genetic_conv1_weights, ...
                          'Bias', genetic_conv1_bias, ...
                          'WeightLearnRateFactor', 0.2, 'BiasLearnRateFactor', 0)
        reluLayer('Name', 'relu1')
        
        convolution2dLayer(9, 1, 'Name', 'conv2', 'Padding', 'same', ...      % Bottleneck like retNet8
                          'Weights', optimized_conv2_weights8, ...
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

net=dlnetwork(layers);
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
genNet87 = trainnet(imds_train_resized, net,"crossentropy",options);
%% Save
save('gennet87.mat','genNet87')
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
