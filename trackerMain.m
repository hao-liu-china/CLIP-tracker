function [results] = trackerMain(p, im, bg_area, fg_area, area_resize_factor)
%TRACKERMAIN contains the main loop of the tracker, P contains all the parameters set in runTracker
    %% INITIALIZATION
    sigma = 0.3;
    num_frames = numel(p.img_files);
    % used for OTB-13 benchmark
    OTB_rect_positions = zeros(num_frames, 4);
	pos = p.init_pos;
    target_sz = p.target_sz;
    % patch of the target + padding
    patch_padded = getSubwindow(im, pos, p.norm_bg_area, bg_area);
    % initialize hist model
    new_pwp_model = true;
    [bg_hist, fg_hist] = updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence);
    new_pwp_model = false;
    % Hann (cosine) window
    if isToolboxAvailable('Signal Processing Toolbox')
        hann_window = single(hann(p.cf_response_size(1)) * hann(p.cf_response_size(2))');
    else
        hann_window = single(myHann(p.cf_response_size(1)) * myHann(p.cf_response_size(2))');
    end
    % gaussian-shaped desired response, centred in (1,1)
    % bandwidth proportional to target size
    output_sigma = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor / p.hog_cell_size;
    y = gaussianResponse(p.cf_response_size, output_sigma);
    yf = fft2(y);
    
    temp = load('w2crs');
    w2c = temp.w2crs;
    
    %% EdgeBoxes
    model = load('./models/forest/modelBsds'); 
    model = model.model;
    model.opts.multiscale = 0;
    model.opts.sharpen = 0;
    model.opts.nThreads = 4;
    % set up parameters for edgeBoxes (see edgeBoxesTrackParam.m)
    opts = edgeBoxes;
    opts.alpha = 0.85;
    opts.beta=0.8;
    opts.maxBoxes=200;
    opts.minScore = 0.0005;
    opts.kappa =1.4;
%   opts.minBoxArea = 200;
    proposal_num_limit = 100;
    redetection_frequency = 5;
    svmupdate_frame = 1;
    %% SCALE ADAPTATION INITIALIZATION
    if p.scale_adaptation
        scale_factor = 1;
        base_target_sz = target_sz;
        scale_sigma = sqrt(p.num_scales) * p.scale_sigma_factor;
        ss = (1:p.num_scales) - ceil(p.num_scales/2);
        ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
        ysf = single(fft(ys));
        if mod(p.num_scales,2) == 0
            scale_window = single(hann(p.num_scales+1));
            scale_window = scale_window(2:end);
        else
            scale_window = single(hann(p.num_scales));
        end;

        ss = 1:p.num_scales;
        scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss);

        if p.scale_model_factor^2 * prod(p.norm_target_sz) > p.scale_model_max_area
            p.scale_model_factor = sqrt(p.scale_model_max_area/prod(p.norm_target_sz));
        end

        scale_model_sz = floor(p.norm_target_sz * p.scale_model_factor);
        % find maximum and minimum scales
        min_scale_factor = p.scale_step ^ ceil(log(max(5 ./ bg_area)) / log(p.scale_step));
        max_scale_factor = p.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ target_sz)) / log(p.scale_step));
    end
    response_cf_output = zeros(num_frames-1,1);
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
    time = 0;
    %% MAIN LOOP
%     tic;
    for frame = 1:num_frames
        im = imread([p.img_path p.img_files{frame}]);
        tic();
        if frame>1
	    %% TESTING step
            % extract patch of size bg_area and resize to norm_bg_area
            im_patch_cf = getSubwindow(im, pos, p.norm_bg_area, bg_area);
            pwp_search_area = round(p.norm_pwp_search_area / area_resize_factor);
            % extract patch of size pwp_search_area and resize to norm_pwp_search_area
            im_patch_pwp = getSubwindow(im, pos, p.norm_pwp_search_area, pwp_search_area);
            % compute feature map
            xt = getFeatureMap(im_patch_cf, p.feature_type, p.cf_response_size, p.hog_cell_size, w2c);
            % apply Hann window
            xt_windowed = bsxfun(@times, hann_window, xt);
            % compute FFT
            kxtf = fft2(xt_windowed);

            kzf = gaussian_correlation(kxtf, model_xtf, sigma);
            response_cf = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
            max_response_cf = max(response_cf(:));
            response_cf_output(frame-1,1) = max_response_cf;
            
            % Crop square search region (in feature pixels).
            response_cf = cropFilterResponse(response_cf, ...
                floor_odd(p.norm_delta_area / p.hog_cell_size));
            if p.hog_cell_size > 1
                % Scale up to match center likelihood resolution.
                response_cf = mexResize(response_cf, p.norm_delta_area,'auto');
            end

            [likelihood_map] = getColourMap(im_patch_pwp, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);
            % (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
            likelihood_map(isnan(likelihood_map)) = 0;

            % each pixel of response_pwp loosely represents the likelihood that
            % the target (of size norm_target_sz) is centred on it
            response_pwp = getCenterLikelihood(likelihood_map, p.norm_target_sz);

            %% ESTIMATION
            response = mergeResponses(response_cf, response_pwp, p.merge_factor, p.merge_method);

            [row, col] = find(response == max(response(:)), 1);
            center = (1+p.norm_delta_area) / 2;
            pos = pos + ([row, col] - center) / area_resize_factor;

            %% re-detection
            if mod(frame,redetection_frequency) == 0
            if max_response_cf < p.motion_thresh
                search_sz = floor(bg_area * p.search_padding);          %re-detection搜索窗的大小
                search_mid_pt = search_sz * 0.5;                        %搜索窗的中心坐标
                
                edgeBoxes_window = get_subwindow(im, pos, search_sz); 
                if size(im,3) == 1 % for gray sequences
                    edgeBoxes_window = single(edgeBoxes_window / 255);
                    edgeBoxes_window = cat(3, edgeBoxes_window, edgeBoxes_window, edgeBoxes_window);
                end
                opts.minBoxArea= 0.8*target_sz(1)*target_sz(2);
                opts.maxBoxArea=1.2*target_sz(1)*target_sz(2);
                opts.maxAspectRatio= 1.5*max(target_sz(1)/target_sz(2),target_sz(2)/target_sz(1));
                opts.aspectRatio = target_sz(2)/target_sz(1);
                bbs= myedgeBoxes(edgeBoxes_window,model,opts);
               
                bbs1 = [bbs(:,1)+bbs(:,3)./2 bbs(:,2)+bbs(:,4)./2  bbs(:,3) bbs(:,4)];
                [ind,maxProb] = prorej(single(edgeBoxes_window),bbs1,svm_model);
                num = floor(size(bbs,1)./2);
                bbs2 =  bbs(ind(1:num),:);
                
                num_of_proposals = 0;
                proposals = zeros(proposal_num_limit,4); % center_y, center_x, rows, cols
                proposals_xywh = zeros(proposal_num_limit,4);
                
                for i = 1 : min([size(bbs2,1) proposal_num_limit])
                    proposal_sz = [bbs2(i,4) bbs2(i,3)];
                    proposal_pos = [bbs2(i,2) bbs2(i,1)] + floor(proposal_sz/2);
                    num_of_proposals = num_of_proposals + 1;
                    proposals(num_of_proposals,:) = [pos+proposal_pos-search_mid_pt, proposal_sz];
                    proposals_xywh(num_of_proposals,:) = [proposals(num_of_proposals,[2,1]) - proposal_sz([2,1])/2, proposal_sz([2,1])];           
                end
                
                model_alpha = ifft2(model_alphaf);
                response_proposal = zeros(num_of_proposals,1);
                
                 for j = 1 : num_of_proposals
                    % calculate the response of the classifier without considering the cyclic shifts
                    % 计算proposal的bg_area——bg_area_proposal
                    avg_dim_proposal = sum(proposals(j,3:4))/2;
                    bg_area_proposal = round(proposals(j,3:4) + avg_dim_proposal);
                    if(bg_area_proposal(2)>size(im,2)),  bg_area_proposal(2)=size(im,2)-1;    end
                    if(bg_area_proposal(1)>size(im,1)),  bg_area_proposal(1)=size(im,1)-1;    end
                    bg_area_proposal = bg_area_proposal - mod(bg_area_proposal - proposals(j,3:4), 2);
                    patch_proposal = getSubwindow(im, proposals(j,1:2), p.norm_bg_area, bg_area_proposal);
                    xt_proposal = getFeatureMap(patch_proposal, p.feature_type, p.cf_response_size, p.hog_cell_size, w2c);
                    xt_windowed_proposal = bsxfun(@times, hann_window, xt_proposal);
                    xtf_proposal = fft2(xt_windowed_proposal);
                    kz_proposal = gaussian_correlation_nofft(xtf_proposal, model_xtf, sigma);
                    response_proposal(j,1) = model_alpha(:)'*kz_proposal(:);
                 end
                [max_response_proposal,ind1]=max(response_proposal(:));
                
                if(max_response_proposal > max_response_cf) & (max_response_proposal > p.motion_thresh)
                    max_response_cf = max_response_proposal;
                    new_pos = proposals(ind1,1:2);
                    pos = pos+ floor(0.8*(new_pos-pos));
                end
            end
            end
            
            %% SCALE SPACE SEARCH
            if p.scale_adaptation
                im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor * scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
                xsf = fft(im_patch_scale,[],2);
                scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + p.lambda) ));
                recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));
                %set the scale
                scale_factor = scale_factor * scale_factors(recovered_scale);

                if scale_factor < min_scale_factor
                    scale_factor = min_scale_factor;
                elseif scale_factor > max_scale_factor
                    scale_factor = max_scale_factor;
                end
                % use new scale to update bboxes for target, filter, bg and fg models
                target_sz = round(base_target_sz * scale_factor);
                rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
                avg_dim = sum(target_sz)/2;
                bg_area = round(target_sz + avg_dim);
                if(bg_area(2)>size(im,2)),  bg_area(2)=size(im,2)-1;    end
                if(bg_area(1)>size(im,1)),  bg_area(1)=size(im,1)-1;    end

                bg_area = bg_area - mod(bg_area - target_sz, 2);
                fg_area = round(target_sz - avg_dim * p.inner_padding);
                fg_area = fg_area + mod(bg_area - fg_area, 2);
                % Compute the rectangle with (or close to) params.fixed_area and
                % same aspect ratio as the target bboxgetScaleSubwindow
                area_resize_factor = sqrt(p.fixed_area/prod(bg_area));
            end

            if p.visualization_dbg==1
                mySubplot(2,1,5,1,im_patch_cf,'FG+BG','gray');
                mySubplot(2,1,5,2,likelihood_map,'obj.likelihood','parula');
                mySubplot(2,1,5,3,response_cf,'CF response','parula');
                mySubplot(2,1,5,4,response_pwp,'center likelihood','parula');
                mySubplot(2,1,5,5,response,'merged response','parula');
                drawnow
            end
        end

        %% TRAINING
        % extract patch of size bg_area and resize to norm_bg_area
        im_patch_bg = getSubwindow(im, pos, p.norm_bg_area, bg_area);
        % compute feature map, of cf_response_size
        xt = getFeatureMap(im_patch_bg, p.feature_type, p.cf_response_size, p.hog_cell_size, w2c);
        % apply Hann window
        xt = bsxfun(@times, hann_window, xt);
        % compute FFT
        xtf = fft2(xt);
        %% FILTER UPDATE
        kf = gaussian_correlation(xtf, xtf, sigma);
        new_alphaf_num = yf .* kf;
        new_alphaf_den = kf .* (kf + p.lambda);
        if frame == 1
            alphaf_num = new_alphaf_num;
            alphaf_den = new_alphaf_den;
            model_xtf = xtf;
            svm_model = online_svm_train(im,pos,floor(bg_area*p.svm_padding),target_sz,opts,model); 
        else
                chengfa = max_response_cf / max(response_cf_output(:));
                alphaf_num = (1 - chengfa*p.learning_rate_cf) * alphaf_num + chengfa*p.learning_rate_cf * new_alphaf_num;
                alphaf_den = (1 - chengfa*p.learning_rate_cf) * alphaf_den + chengfa*p.learning_rate_cf * new_alphaf_den;
                model_xtf = (1 - chengfa*p.learning_rate_cf) * model_xtf + chengfa*p.learning_rate_cf * xtf;
            %% BG/FG MODEL UPDATE
            % patch of the target + padding
                [bg_hist, fg_hist] = updateHistModel(new_pwp_model, im_patch_bg, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence, bg_hist, fg_hist, chengfa*p.learning_rate_pwp);
                if (max_response_cf > p.appearance_thresh) &&(frame-svmupdate_frame ~= 1)
                    svm_model = online_svm_update(im,pos,floor(bg_area*p.svm_padding),target_sz,opts,model,svm_model);
                    svmupdate_frame = frame;
                end
        end
        model_alphaf = alphaf_num ./ alphaf_den;
        time = time + toc();
        %% SCALE UPDATE
        if p.scale_adaptation
            im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
            xsf = fft(im_patch_scale,[],2);
            new_sf_num = bsxfun(@times, ysf, conj(xsf));
            new_sf_den = sum(xsf .* conj(xsf), 1);
            if frame == 1
                sf_den = new_sf_den;
                sf_num = new_sf_num;
            else
                sf_den = (1 - chengfa*p.learning_rate_scale) * sf_den + chengfa*p.learning_rate_scale * new_sf_den;
                sf_num = (1 - chengfa*p.learning_rate_scale) * sf_num + chengfa*p.learning_rate_scale * new_sf_num;
            end
        end

        % update bbox position
        if frame==1, rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])]; end

        rect_position_padded = [pos([2,1]) - bg_area([2,1])/2, bg_area([2,1])];

        OTB_rect_positions(frame,:) = rect_position;

        if p.fout > 0,  fprintf(p.fout,'%.2f,%.2f,%.2f,%.2f\n', rect_position(1),rect_position(2),rect_position(3),rect_position(4));   end

        %% VISUALIZATION
        if p.visualization == 1
                figure(1)
                imshow(im)
                rectangle('Position',rect_position, 'LineWidth',2, 'EdgeColor','g');
                rectangle('Position',rect_position_padded, 'LineWidth',2, 'LineStyle','--', 'EdgeColor','b');
                drawnow
        end
    end
    results.type = 'rect';
    results.res = OTB_rect_positions;
    results.fps = num_frames/time;
end

% Reimplementation of Hann window (in case signal processing toolbox is missing)
function H = myHann(X)
    H = .5*(1 - cos(2*pi*(0:X-1)'/(X-1)));
end

% We want odd regions so that the central pixel can be exact
function y = floor_odd(x)
    y = 2*floor((x-1) / 2) + 1;
end

function y = ensure_real(x)
    assert(norm(imag(x(:))) <= 1e-5 * norm(real(x(:))));
    y = real(x);
end
