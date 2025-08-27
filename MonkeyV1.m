%% Step 7: Look at receptive fields from multiple neurons
clear; clc;

% Update this to your data path
data_path = 'C:\Users\leozi\OneDrive\Desktop\Research\misc_references\data_and_scripts';

fprintf('=== Computing RFs for multiple neurons ===\n\n');

%% Load data (same as before)
stim_file = fullfile(data_path, 'stimuli_movies', 'natural_movie.mat');
stim_data = load(stim_file);
stimulus_movie = stim_data.M;

spike_file = fullfile(data_path, 'spikes_movies', 'data_monkey1_natural_movie.mat');
spike_data = load(spike_file);
events = spike_data.data.EVENTS;

frame_duration = 30 / 750;
frame_times = (0:749) * frame_duration;
mean_stimulus = mean(stimulus_movie, 3);

%% Function to compute RF for one neuron
function [rf, num_spikes] = compute_neuron_rf(events, neuron_idx, stimulus_movie, frame_times, mean_stimulus)
    % Extract all spike times for this neuron
    all_spike_times = [];
    for trial = 1:size(events, 2)
        trial_spikes = events{neuron_idx, trial};
        if ~isempty(trial_spikes) && isnumeric(trial_spikes)
            all_spike_times = [all_spike_times; trial_spikes(:)];
        end
    end
    
    % Find triggered frames
    triggered_frames = [];
    for spike_idx = 1:length(all_spike_times)
        spike_time = all_spike_times(spike_idx);
        target_time = spike_time - 0.05;  % 50ms before spike
        
        if target_time >= 0
            [~, frame_idx] = min(abs(frame_times - target_time));
            triggered_frames(end+1) = frame_idx;
        end
    end
    
    % Compute RF if enough spikes
    if length(triggered_frames) >= 20
        triggered_stimuli = stimulus_movie(:, :, triggered_frames);
        rf = mean(triggered_stimuli, 3) - mean_stimulus;
        num_spikes = length(triggered_frames);
    else
        rf = [];
        num_spikes = length(triggered_frames);
    end
end

%% Try several neurons
neurons_to_try = [1, 5, 10, 15, 20, 25];  % Try 6 different neurons
figure(1); clf;

good_rfs = 0;
for i = 1:length(neurons_to_try)
    neuron_idx = neurons_to_try(i);
    
    [rf, num_spikes] = compute_neuron_rf(events, neuron_idx, stimulus_movie, frame_times, mean_stimulus);
    
    if ~isempty(rf)
        good_rfs = good_rfs + 1;
        
        subplot(2, 3, i);
        imagesc(rf, [-max(abs(rf(:))), max(abs(rf(:)))]);
        title(sprintf('Neuron %d (%d spikes)', neuron_idx, num_spikes));
        colormap gray; axis off;
        
        fprintf('Neuron %d: %d spikes, RF range %.3f to %.3f\n', ...
            neuron_idx, num_spikes, min(rf(:)), max(rf(:)));
    else
        subplot(2, 3, i);
        text(0.5, 0.5, sprintf('Neuron %d\nNot enough spikes\n(%d)', neuron_idx, num_spikes), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        axis off;
        
        fprintf('Neuron %d: Only %d spikes - skipping\n', neuron_idx, num_spikes);
    end
end

sgtitle('Receptive Fields from Different Neurons');

fprintf('\nFound %d neurons with enough spikes for RF computation\n', good_rfs);
fprintf('\nNext: We can classify these RFs and build a dataset\n');
%% Step 3: Look at the actual data inside
clear; clc;

% Update this to your data path
data_path = 'C:\Users\leozi\OneDrive\Desktop\Research\misc_references\data_and_scripts';

fprintf('=== Examining the actual data ===\n\n');

%% Look at the stimulus data
fprintf('1. STIMULUS DATA:\n');
stim_file = fullfile(data_path, 'stimuli_movies', 'noise_movie.mat');
stim_data = load(stim_file);

% M is our stimulus movie: 320x320 pixels, 750 frames
stimulus_movie = stim_data.M;
fprintf('Stimulus movie: %d x %d pixels, %d frames\n', size(stimulus_movie));

% Look at the range of values
fprintf('Pixel values range from %.2f to %.2f\n', min(stimulus_movie(:)), max(stimulus_movie(:)));

% Show a few frames
figure(1); clf;
subplot(1,3,1); imagesc(stimulus_movie(:,:,1)); title('Frame 1'); colormap gray; axis off;
subplot(1,3,2); imagesc(stimulus_movie(:,:,100)); title('Frame 100'); colormap gray; axis off;
subplot(1,3,3); imagesc(stimulus_movie(:,:,200)); title('Frame 200'); colormap gray; axis off;
sgtitle('Sample stimulus frames');

%% Look at the spike data
fprintf('\n2. SPIKE DATA:\n');
spike_file = fullfile(data_path, 'spikes_movies', 'data_monkey1_noise_movie.mat');
spike_data = load(spike_file);

% The data variable is a structure
data_struct = spike_data.data;
fprintf('Data structure fields:\n');
field_names = fieldnames(data_struct);
for i = 1:length(field_names)
    fprintf('  %s\n', field_names{i});
end

% Let's look at what's inside
if isfield(data_struct, 'spikes')
    spikes_info = data_struct.spikes;
    fprintf('\nSpikes structure:\n');
    spike_fields = fieldnames(spikes_info);
    for i = 1:length(spike_fields)
        fprintf('  %s\n', spike_fields{i});
    end
    
    % Look at spike times
    if isfield(spikes_info, 'times')
        spike_times = spikes_info.times;
        fprintf('\nNumber of neurons: %d\n', length(spike_times));
        
        % Show spike times for first few neurons
        for neuron = 1:min(3, length(spike_times))
            num_spikes = length(spike_times{neuron});
            if num_spikes > 0
                first_few_spikes = spike_times{neuron}(1:min(5, num_spikes));
                fprintf('Neuron %d: %d spikes, first few times: [%.3f %.3f %.3f ...]\n', ...
                    neuron, num_spikes, first_few_spikes);
            end
        end
    end
end

fprintf('\nNext: We''ll figure out how to match stimulus frames to spike times\n');