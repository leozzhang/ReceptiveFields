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
%% testing create center surround template

% Example usage - let's create and visualize a center-surround template
filter_size = [5, 5];  % Same as your CNN filters
h = filter_size(1);
w = filter_size(2);

% Template parameters
center_x = 3;       % Center position (middle of 5x5 filter)
center_y = 3;
center_radius = 1.2;  % Small center
surround_radius = 1.8*1.2; % Larger surround

% Create the template
template = createCenterSurroundTemplate(h, w, center_x, center_y, center_radius, surround_radius);

% Display it
figure;
imagesc(template);
colormap gray;
colorbar;
title('Example Center-Surround Template');
xlabel('Positive center (+1), Negative surround (-0.5)');

% Print the actual values so you can see the pattern
fprintf('Template values:\n');
disp(template);
%% 
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
%% testing compute template match
template = createCenterSurroundTemplate(h, w, 3, 3, 1, 2.5);
perfect_filter=template;
score1 = computeTemplateMatch(perfect_filter, template);
fprintf('Test 1 - Perfect match: score = %.3f\n', score1);
disp(template)
random_filter = randn(5, 5);
score2 = computeTemplateMatch(random_filter, template);
fprintf('Test 2 - Random noise: score = %.3f\n', score2);
inverted_filter = -template;
score3 = computeTemplateMatch(inverted_filter, template);
fprintf('Test 3 - Inverted (OFF-center): score = %.3f\n', score3);
edge_filter = [-1, -1, -1, -1, -1;
               -1, -1, -1, -1, -1;
               +2, +2, +2, +2, +2;
               -1, -1, -1, -1, -1;
               -1, -1, -1, -1, -1];
score4 = computeTemplateMatch(edge_filter, template);
fprintf('Test 4 - Edge detector: score = %.3f\n', score4);
cnn_filter = [0.1, -0.3, 0.2, -0.1, 0.0;
             -0.2, 0.8, 0.9, 0.7, -0.1;
             0.0, 0.7, 1.2, 0.8, 0.1;
             -0.1, 0.6, 0.9, 0.5, -0.2;
             0.0, -0.2, 0.1, -0.3, 0.1];
score5 = computeTemplateMatch(cnn_filter, template);
fprintf('Test 5 - Simulated CNN filter: score = %.3f\n', score5);
%% 
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

% Simple wrapper function for easy use
function cs_score = getCenterSurroundScore(filter)
    % SIMPLE FUNCTION: Just returns the center-surround score (0 to 1)
    % This is what you'll use most often
    
    [score, ~] = findBestCenterSurroundMatch_optimized(filter);
    cs_score = score;
end
%% testing finding best center surround
clc
test_filter1 = createCenterSurroundTemplate(5, 5, 3, 4, 0.8, 1.8);
score1 = getCenterSurroundScore(test_filter1);
fprintf('Perfect CS filter score: %.3f\n', score1);

% Test 2: Random filter
rng(8);
test_filter2 = randn(5, 5);
score2 = findBestCenterSurroundMatch_optimized(test_filter2);
fprintf('Random filter score: %.3f\n', score2);

% Test 3: actual scratchnetlite filters
load('scratchlitecifar.mat','sNetliteCIFAR');
weights = sNetliteCIFAR.Layers(2).Weights;
num_filters=size(weights,4); %4th dimension of weights contains num filters
cs_scores=zeros(1,num_filters);
for i=1:num_filters
    fprintf('Filter %d ',i);
    filter=weights(:,:,1,i);
    cs_scores(i)=getCenterSurroundScore(filter);
end