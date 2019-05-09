function out = getFeatureMap(im_patch, feature_type, cf_response_size, hog_cell_size, w2c)

% code from DSST
hog_orientations = 9;
% allocate space
switch feature_type
    case 'fhog'
        if size(im_patch,3)>1
            gray_im_patch = rgb2gray(im_patch);
        else
            gray_im_patch = im_patch;
        end
        hog = fhog(single(gray_im_patch), hog_cell_size, hog_orientations);
        h = cf_response_size(1);
        w = cf_response_size(2);
        hog(:,:,end) = [];
        if hog_cell_size > 1
            im_patch = mexResize(im_patch, [h, w] ,'auto');
        end
        [gray, cn] = get_patch_feature(im_patch, 'gray', 'cn', w2c);
        out = cat(3, hog, gray, cn);
        
    case 'gray'
        if hog_cell_size > 1, im_patch = mexResize(im_patch,cf_response_size,'auto');   end
        if size(im_patch, 3) == 1
            out = single(im_patch)/255 - 0.5;
        else
            out = single(rgb2gray(im_patch))/255 - 0.5;
        end        
end
        
end

