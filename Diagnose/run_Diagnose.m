function results = run_Diagnose(seq, res_path, bSaveImage)
    configGlobalParam;
    config;
    rng(0);
    seq.opt = opt;
    rect=seq.init_rect;
    p = [rect(1)+rect(3)/2, rect(2)+rect(4)/2, rect(3), rect(4)]; % Center x, Center y, height, width
    frame = imread(seq.s_frames{1});
    if size(frame,3)==1
        frame = repmat(frame,[1,1,3]);
    end
    frame = rgb2gray(frame);
    if (seq.opt.useNormalSize)
        scaleHeight = size(frame, 1) / seq.opt.normalHeight;
        scaleWidth = size(frame, 2) / seq.opt.normalWidth;
        p(1) = p(1) / scaleWidth;
        p(3) = p(3) / scaleWidth;
        p(2) = p(2) / scaleHeight;
        p(4) = p(4) / scaleHeight;
    end
    
    duration = 0;
    tic;
    reportRes = [];
    for f = 1:size(seq.s_frames, 1)
        disp(f)
        frame = imread(seq.s_frames{f});
        if size(frame,3)==1
            frame = repmat(frame,[1,1,3]);
        end
        if (seq.opt.useNormalSize)
%           frame = imresize(frame, [seq.opt.normalHeight, seq.opt.normalWidth]);
            frame = mexResize(frame, [seq.opt.normalHeight, seq.opt.normalWidth], 'auto');
        end
        frame = im2double(frame);
        
        if (f ~= 1)
            tmpl    = globalParam.MotionModel(tmpl, prob, seq.opt);
            [feat, seq.opt] = globalParam.FeatureExtractor(frame, tmpl, seq.opt);
            prob    = globalParam.ObservationModelTest(feat, model);    
            
            [maxProb, maxIdx] = max(prob); 
            p = tmpl(maxIdx, :);
            model.lastOutput = p;
            model.lastProb = maxProb;
            if (strcmp(func2str(globalParam.ConfidenceJudger), 'UpdateDifferenceJudger'))
                w1 = max(round(tmpl(:, 1) - tmpl(:, 3) / 2), round(p(:, 1) - p(:, 3) / 2));
                w2 = min(round(tmpl(:, 1) + tmpl(:, 3) / 2), round(p(:, 1) + p(:, 3) / 2));
                h1 = max(round(tmpl(:, 2) - tmpl(:, 4) / 2), round(p(:, 2) - p(:, 4) / 2));
                h2 = min(round(tmpl(:, 2) + tmpl(:, 4) / 2), round(p(:, 2) + p(:, 4) / 2));
                interArea = max(w2 - w1, 0) .* max(h2 - h1, 0);
                jointArea = (round(tmpl(:, 3)) .* round(tmpl(:, 4)) + round(p(3)) * round(p(4))) - interArea;
                overlapRatio = interArea ./ jointArea;
                idx = (overlapRatio < seq.opt.UpdateDifferenceJudger.overlap);
                model.secondProb = max(prob(idx));
            end
        else
            tmpl = globalParam.MotionModel(p, 1, seq.opt);
            prob = ones(1, size(tmpl, 1));
        end     
        
        if (f == 1)
            tmplPos = globalParam.PosSampler(p, seq.opt);
            tmplNeg = globalParam.NegSampler(p, seq.opt);
            [dataPos, seq.opt] = globalParam.FeatureExtractor(frame, tmplPos, seq.opt);
            [dataNeg, seq.opt] = globalParam.FeatureExtractor(frame, tmplNeg, seq.opt);
            model   = globalParam.ObservationModelTrain(dataPos, dataNeg, seq.opt);  
            if (seq.opt.useFirstFrame)
                assert(~strcmp(func2str(globalParam.ObservationModelTrain), 'SOSVMTrain'), ...
                    'SOSVM does not support useFirstFrame option!!');
                dataPosFirstFrame = dataPos;
            end
        else
            if (globalParam.ConfidenceJudger(model, seq.opt))
                tmplPos = globalParam.PosSampler(p, seq.opt);
                tmplNeg = globalParam.NegSampler(p, seq.opt);
                [dataPos, seq.opt] = globalParam.FeatureExtractor(frame, tmplPos, seq.opt);
                [dataNeg, seq.opt] = globalParam.FeatureExtractor(frame, tmplNeg, seq.opt);
%                 disp(f);
                if (seq.opt.useFirstFrame)
                    dataPos.feat = [dataPosFirstFrame.feat, dataPos.feat];
                    dataPos.tmpl = [zeros(size(dataPosFirstFrame.tmpl)); dataPos.tmpl];
                end
                model   = globalParam.ObservationModelTrain(dataPos, dataNeg, seq.opt, model);  
            end
        end
%         
        figure(1),imagesc(frame);
%         pause(0.1);
        imshow(frame); 
        rectangle('position', [p(1) - p(3) / 2, p(2) - p(4) / 2, p(3), p(4)], ...
            'EdgeColor','r', 'LineWidth',2);
        drawnow;
        if (seq.opt.useNormalSize)
            p(1) = p(1) * scaleWidth;
            p(3) = p(3) * scaleWidth;
            p(2) = p(2) * scaleHeight;
            p(4) = p(4) * scaleHeight;
        end
        rect = [p(1) - p(3) / 2, p(2) - p(4) / 2, p(3), p(4)];
        reportRes = [reportRes; round(rect)];
    end
    if (strcmp(func2str(globalParam.ObservationModelTrain), 'SOSVMTrain'))
        mexSOSVMLearn([], [], 'delete');
    end
        
    duration = duration + toc;
    fprintf('%d frames took %.3f seconds : %.3fps\n',f,duration,f/duration);
    results.res=reportRes;
    results.type='rect';
    results.fps = f/duration;
    
end