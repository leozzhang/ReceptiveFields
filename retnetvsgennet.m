%% POPULATION DIAGNOSTIC - retNet vs genNet
% Compares weight and gradient statistics across populations

%% Setup
retnet_nums = 43:62;   % your 20 retNets
gennet_nums = 888:907; % your 20 genNets

num_nets = 20;
layers_to_check = {'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7'};

% Initialize storage structs
retnet_stats = struct();
gennet_stats = struct();

for l = 1:length(layers_to_check)
    layer = layers_to_check{l};
    retnet_stats.(layer).grad_norm = zeros(num_nets, 1);
    gennet_stats.(layer).grad_norm = zeros(num_nets, 1);
    retnet_stats.(layer).neighborhood_pos_frac = zeros(num_nets, 1);
    gennet_stats.(layer).neighborhood_pos_frac = zeros(num_nets, 1);
    retnet_stats.(layer).act_mean = zeros(num_nets, 1);
    gennet_stats.(layer).act_mean = zeros(num_nets, 1);
    retnet_stats.(layer).act_std = zeros(num_nets, 1);
    gennet_stats.(layer).act_std = zeros(num_nets, 1);
    retnet_stats.(layer).act_pos_frac = zeros(num_nets, 1);
    gennet_stats.(layer).act_pos_frac = zeros(num_nets, 1);
    retnet_stats.(layer).weight_mean = zeros(num_nets, 1);
    gennet_stats.(layer).weight_mean = zeros(num_nets, 1);
    retnet_stats.(layer).weight_std = zeros(num_nets, 1);
    gennet_stats.(layer).weight_std = zeros(num_nets, 1);
    retnet_stats.(layer).weight_skewness = zeros(num_nets, 1);
    gennet_stats.(layer).weight_skewness = zeros(num_nets, 1);
end

retnet_stats.conv4_failed = zeros(num_nets, 1);
gennet_stats.conv4_failed = zeros(num_nets, 1);

%% Collect stats for retNets
fprintf('\n========================================\n');
fprintf('Collecting retNet statistics...\n');
fprintf('========================================\n');

for i = 1:num_nets
    net_num = retnet_nums(i);
    fprintf('\nProcessing retNet%d (%d/20)...\n', net_num, i);
    
    try
        load_varname = sprintf('retNet%d', net_num);
        load(sprintf('retnet%d.mat', net_num), load_varname);
        net = eval(load_varname);
        inputSize = net.Layers(1).InputSize;
        
        inputImg = dlarray(0.5 * ones([inputSize, 1], 'single'), 'SSCB');
        
        for l = 1:length(layers_to_check)
            layerName = layers_to_check{l};
            
            % Forward pass statistics
            layerOutput = forward(net, inputImg, 'Outputs', layerName);
            layerData = safeExtract(layerOutput);
            
            retnet_stats.(layerName).act_mean(i) = mean(layerData(:));
            retnet_stats.(layerName).act_std(i) = std(layerData(:));
            retnet_stats.(layerName).act_pos_frac(i) = mean(layerData(:) > 0);
            
            % Neighborhood positive fraction at center
            [H, W, C, B] = size(layerData);
            center_h = round(H/2);
            center_w = round(W/2);
            half_kernel = 4;
            neighborhood = layerData(...
                max(1,center_h-half_kernel):min(H,center_h+half_kernel), ...
                max(1,center_w-half_kernel):min(W,center_w+half_kernel), :, :);
            retnet_stats.(layerName).neighborhood_pos_frac(i) = mean(neighborhood(:) > 0);
            
            % Gradient statistics
            [~, gradients] = dlfeval(@lindseyLossFcn_diag, net, inputImg, layerName, 1, 'center');
            grad_norm = safeExtract(sqrt(sum(gradients.^2, 'all')));
            retnet_stats.(layerName).grad_norm(i) = grad_norm;
            
            % Weight statistics
            layer_idx = find(strcmp({net.Layers.Name}, layerName));
            if ~isempty(net.Layers(layer_idx).Weights)
                w = safeExtract(net.Layers(layer_idx).Weights);
                retnet_stats.(layerName).weight_mean(i) = mean(w(:));
                retnet_stats.(layerName).weight_std(i) = std(w(:));
                retnet_stats.(layerName).weight_skewness(i) = skewness(w(:));
            end
        end
        
        retnet_stats.conv4_failed(i) = retnet_stats.conv4.grad_norm(i) <= 1e-5;
        
        fprintf('  conv3 neighborhood pos frac: %.4f | conv4 grad norm: %.8f | failed: %d\n', ...
            retnet_stats.conv3.neighborhood_pos_frac(i), ...
            retnet_stats.conv4.grad_norm(i), ...
            retnet_stats.conv4_failed(i));
        
        clear net
        
    catch ME
        fprintf('ERROR on retNet%d: %s\n', net_num, ME.message);
    end
end

%% Collect stats for genNets
fprintf('\n========================================\n');
fprintf('Collecting genNet statistics...\n');
fprintf('========================================\n');

for i = 1:num_nets
    net_num = gennet_nums(i);
    fprintf('\nProcessing genNet%d (%d/20)...\n', net_num, i);
    
    try
        load_varname = sprintf('genNet%d', net_num);
        load(sprintf('gennet%d.mat', net_num), load_varname);
        net = eval(load_varname);
        inputSize = net.Layers(1).InputSize;
        
        inputImg = dlarray(0.5 * ones([inputSize, 1], 'single'), 'SSCB');
        
        for l = 1:length(layers_to_check)
            layerName = layers_to_check{l};
            
            % Forward pass statistics
            layerOutput = forward(net, inputImg, 'Outputs', layerName);
            layerData = safeExtract(layerOutput);
            
            gennet_stats.(layerName).act_mean(i) = mean(layerData(:));
            gennet_stats.(layerName).act_std(i) = std(layerData(:));
            gennet_stats.(layerName).act_pos_frac(i) = mean(layerData(:) > 0);
            
            % Neighborhood positive fraction at center
            [H, W, C, B] = size(layerData);
            center_h = round(H/2);
            center_w = round(W/2);
            half_kernel = 4;
            neighborhood = layerData(...
                max(1,center_h-half_kernel):min(H,center_h+half_kernel), ...
                max(1,center_w-half_kernel):min(W,center_w+half_kernel), :, :);
            gennet_stats.(layerName).neighborhood_pos_frac(i) = mean(neighborhood(:) > 0);
            
            % Gradient statistics
            [~, gradients] = dlfeval(@lindseyLossFcn_diag, net, inputImg, layerName, 1, 'center');
            grad_norm = safeExtract(sqrt(sum(gradients.^2, 'all')));
            gennet_stats.(layerName).grad_norm(i) = grad_norm;
            
            % Weight statistics
            layer_idx = find(strcmp({net.Layers.Name}, layerName));
            if ~isempty(net.Layers(layer_idx).Weights)
                w = safeExtract(net.Layers(layer_idx).Weights);
                gennet_stats.(layerName).weight_mean(i) = mean(w(:));
                gennet_stats.(layerName).weight_std(i) = std(w(:));
                gennet_stats.(layerName).weight_skewness(i) = skewness(w(:));
            end
        end
        
        gennet_stats.conv4_failed(i) = gennet_stats.conv4.grad_norm(i) <= 1e-5;
        
        fprintf('  conv3 neighborhood pos frac: %.4f | conv4 grad norm: %.8f | failed: %d\n', ...
            gennet_stats.conv3.neighborhood_pos_frac(i), ...
            gennet_stats.conv4.grad_norm(i), ...
            gennet_stats.conv4_failed(i));
        
        clear net
        
    catch ME
        fprintf('ERROR on genNet%d: %s\n', net_num, ME.message);
    end
end

%% Print Summary Statistics
fprintf('\n\n========================================\n');
fprintf('POPULATION SUMMARY\n');
fprintf('========================================\n');

fprintf('\nVisualization failure rate:\n');
fprintf('  retNet: %d/20 failed (%.1f%%)\n', sum(retnet_stats.conv4_failed), mean(retnet_stats.conv4_failed)*100);
fprintf('  genNet: %d/20 failed (%.1f%%)\n', sum(gennet_stats.conv4_failed), mean(gennet_stats.conv4_failed)*100);

for l = 1:length(layers_to_check)
    layerName = layers_to_check{l};
    fprintf('\n--- %s ---\n', layerName);
    
    fprintf('  Neighborhood positive fraction:\n');
    fprintf('    retNet: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n', ...
        mean(retnet_stats.(layerName).neighborhood_pos_frac), ...
        std(retnet_stats.(layerName).neighborhood_pos_frac), ...
        min(retnet_stats.(layerName).neighborhood_pos_frac), ...
        max(retnet_stats.(layerName).neighborhood_pos_frac));
    fprintf('    genNet: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n', ...
        mean(gennet_stats.(layerName).neighborhood_pos_frac), ...
        std(gennet_stats.(layerName).neighborhood_pos_frac), ...
        min(gennet_stats.(layerName).neighborhood_pos_frac), ...
        max(gennet_stats.(layerName).neighborhood_pos_frac));
    
    fprintf('  Gradient norm:\n');
    fprintf('    retNet: mean=%.8f, std=%.8f\n', ...
        mean(retnet_stats.(layerName).grad_norm), ...
        std(retnet_stats.(layerName).grad_norm));
    fprintf('    genNet: mean=%.8f, std=%.8f\n', ...
        mean(gennet_stats.(layerName).grad_norm), ...
        std(gennet_stats.(layerName).grad_norm));
    
    fprintf('  Activation positive fraction (full spatial map):\n');
    fprintf('    retNet: mean=%.4f, std=%.4f\n', ...
        mean(retnet_stats.(layerName).act_pos_frac), ...
        std(retnet_stats.(layerName).act_pos_frac));
    fprintf('    genNet: mean=%.4f, std=%.4f\n', ...
        mean(gennet_stats.(layerName).act_pos_frac), ...
        std(gennet_stats.(layerName).act_pos_frac));
    
    fprintf('  Weight skewness (post-training):\n');
    fprintf('    retNet: mean=%.4f, std=%.4f\n', ...
        mean(retnet_stats.(layerName).weight_skewness), ...
        std(retnet_stats.(layerName).weight_skewness));
    fprintf('    genNet: mean=%.4f, std=%.4f\n', ...
        mean(gennet_stats.(layerName).weight_skewness), ...
        std(gennet_stats.(layerName).weight_skewness));
end

%% Visualize
figure('Position', [100, 100, 2400, 1200]);

metrics = {'neighborhood_pos_frac', 'grad_norm', 'act_pos_frac', 'weight_skewness', 'weight_mean'};
metric_labels = {'Neighborhood Positive Fraction', 'Gradient Norm', ...
    'Activation Positive Fraction', 'Weight Skewness', 'Weight Mean'};
failed_net_nums = [43, 46, 47, 48, 51, 54];
retnet_nums = 43:62;  % Match whatever range you used in your analysis
failed_idx = find(ismember(retnet_nums, failed_net_nums));

plot_idx = 1;
for m = 1:length(metrics)
    metric = metrics{m};
    for l = 1:length(layers_to_check)
        layerName = layers_to_check{l};
        
        subplot(length(metrics), length(layers_to_check), plot_idx);
        
        retnet_vals = retnet_stats.(layerName).(metric);
        gennet_vals = gennet_stats.(layerName).(metric);
        
        % Box plot
        boxplot([retnet_vals; gennet_vals], ...
            [ones(num_nets,1); 2*ones(num_nets,1)], ...
            'Labels', {'retNet', 'genNet'});
        
        % Overlay individual points
        hold on;
        scatter(ones(num_nets,1) + 0.1*randn(num_nets,1), retnet_vals, 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
        scatter(2*ones(num_nets,1) + 0.1*randn(num_nets,1), gennet_vals, 30, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
        
        % Highlight failed retNets
        if ~isempty(failed_idx)
            scatter(ones(length(failed_idx),1) + 0.1*randn(length(failed_idx),1), ...
                retnet_vals(failed_idx), 60, 'k', 'filled', 'MarkerEdgeColor', 'red', 'LineWidth', 2);
        end
        hold off;
        
        title(sprintf('%s - %s', metric_labels{m}, layerName), 'FontSize', 8);
        
        % p-value from ranksum test
        [p, ~] = ranksum(retnet_vals, gennet_vals);
        xlabel(sprintf('p=%.4f', p), 'FontSize', 7);
        
        plot_idx = plot_idx + 1;
    end
end

sgtitle('retNet vs genNet Population Diagnostics', 'FontSize', 12, 'FontWeight', 'bold');

%% Helper functions
function data = safeExtract(x)
    if isa(x, 'dlarray')
        data = extractdata(x);
    else
        data = x;
    end
end

function [loss, gradients] = lindseyLossFcn_diag(net, inputImg, layerName, filterIndex, spatialPos)
    layerOutput = forward(net, inputImg, 'Outputs', layerName);
    [H, W, C, B] = size(layerOutput);
    if strcmp(spatialPos, 'center')
        pos_x = round(H/2);
        pos_y = round(W/2);
    else
        pos_x = spatialPos(1);
        pos_y = spatialPos(2);
    end
    loss = mean(layerOutput(pos_x, pos_y, filterIndex, :), 'all');
    gradients = dlgradient(loss, inputImg);
end