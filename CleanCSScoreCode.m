function template = createCenterSurroundTemplate(h, w, cx, cy, center_r, surround_r)

    % creates an "ideal" center-surround pattern

    % This creates matrices X and Y where each position contains its coordinates
    [X, Y] = meshgrid(1:w, 1:h);
    
    % Calculate distance from each pixel to the center point
    distances = sqrt((X - cx).^2 + (Y - cy).^2);
    template = zeros(h, w);
    % Find all pixels that are within center_r distance from center
    center_mask = distances <= center_r;
    template(center_mask) = 1;  % Positive response in center
    % Find pixels that are between center_r and surround_r distance from center
    surround_mask = (distances > center_r) & (distances <= surround_r);
    template(surround_mask) = -0.5;  % Negative response in surround
    
    %Normalize the template so it has unit energy
    % This ensures fair comparison between different sized templates
    template = template / norm(template(:));
end

%using cross correlation to scan a smaller template over filter
%possible issues with false positives
function score=normxcorr2csscore(filter)
    %normxcorr2 removing padding and getting max score
        
    template=createCenterSurroundTemplate(3,3,2,2,0.6,1);
    c=normxcorr2(template, filter);
    [th,tw]=size(template);
    [fh,fw]=size(c);
    padding_h = floor(th/2);
    padding_w = floor(tw/2);
    valid_rows = 2*padding_h + 1 : fh - 2*padding_h;
    valid_cols = 2*padding_w + 1 : fw - 2*padding_w;
    c=c(valid_rows,valid_cols);
    figure;
    imagesc(c)
    colormap gray
    colorbar
    score=max(abs(c(:)));
end

%using same size template on filter, brute forcing for best match
%less false positive risk
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


%% 

clear,clc
load('scratchmnist.mat','scratchNet');
weights = scratchNet.Layers(2).Weights;
num_filters=size(weights,4); %4th dimension of weights contains num filters
cs_scores=zeros(1,num_filters);
for i=1:num_filters
    filter=weights(:,:,1,i);
    cs_scores(i)=corr2csscore(filter);
    fprintf('Filter %d: %.3f\n',i, cs_scores(i));
end