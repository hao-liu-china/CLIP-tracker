function results = run_CLIP(seq, res_path, bSaveImage)
% Entry point for the Wu - CVPR2013 benchmark
    addpath('./utils');
    addpath('./edgesbox');
    addpath(genpath('./Diagnose'));
    addpath('E:\Matlab2016b\bin\vlfeat-master')
    run('E:\Matlab2016b\bin\vlfeat-master\toolbox\vl_setup')
    %% Read params.txt
    params = readParams('params.txt');
    params.img_files = seq.s_frames;
    params.img_path = '';

    im = imread(params.img_files{1});
    % grayscale sequence? --> use 1D instead of 3D histograms
    if(size(im,3)==1)
        params.grayscale_sequence = true;
    end

    region = seq.init_rect;

    if(numel(region)==8)
        % polygon format (VOT14, VOT15)
        [cx, cy, w, h] = getAxisAlignedBB(region);
    else % rectangle format (WuCVPR13)
        x = region(1);
        y = region(2);
        w = region(3);
        h = region(4);
        cx = x+w/2;
        cy = y+h/2;
    end
    
    % init_pos is the centre of the initial bounding box
    params.init_pos = [cy cx];
    params.target_sz = round([h w]);
    params.motion_thresh = 0.15;
    params.appearance_thresh = 0.44;
    params.search_padding = 2;
    
    params.EB_maxAR_factor = 1.5;
    params.EB_minArea_factor = 0.3;
    params.backSup_size = 64;
    
    params.svm_padding = 1.5;
    
    [params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params);

	% in runTracker we do not output anything because it is just for debug
	params.fout = -1;

	% start the actual tracking
	results = trackerMain(params, im, bg_area, fg_area, area_resize_factor);
    fclose('all');
end
