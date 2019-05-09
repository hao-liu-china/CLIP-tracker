function svm_model = online_svm_train(im,pos,window_sz,target_sz,opts,model)    
          %addpath(genpath('/home/carrierlxk/tracking_algorithm/Diagnose/'));
          %% train svm model using proposals comes from first frame
          [patch,~] = my_get_subwindow(im, pos, window_sz);
            if size(patch,3)<2
                patch = cat(3,patch,patch,patch);
            end
            opts.minBoxArea= 0.8*target_sz(1)*target_sz(2);
            opts.maxBoxArea=1.25*target_sz(1)*target_sz(2);
            opts.maxAspectRatio= 1.5*max(target_sz(1)/target_sz(2),target_sz(2)/target_sz(1));
            opts.aspectRatio = target_sz(2)/target_sz(1);
            bbs2= myedgeBoxes(patch,model,opts);
            gt2 = [(size(patch,2)-target_sz(2))./2 (size(patch,1)-target_sz(1))./2 target_sz(2) target_sz(1)];
%             figure,imshow(patch);
%             rectangle('Position',gt2,'edgecolor','r');
            tmplPos=[];
            tmplNeg=[];
            gt3 = [gt2([1,2]) gt2([1,2])+gt2([3,4])];
            for ii = 1:size(bbs2,1)
               bb=bbs2(ii,1:4);
               bb1=[bb(1) bb(2) bb(3)+bb(1) bb(4)+bb(2)];
               ovlp= boxoverlap(bb1,gt3);
               if ovlp>0.5
                 temp=([bb(1)+bb(3)./2 bb(2)+bb(4)./2 bb(3) bb(4)]);
                 tmplPos=[tmplPos;temp];
               else 
                  temp = ([bb(1)+bb(3)./2 bb(2)+bb(4)./2 bb(3) bb(4)]);
                  tmplNeg=[tmplNeg;temp];
               end
            end
            if isempty(tmplPos)
                pos=([gt2(1)+gt2(3)/2 gt2(2)+gt2(4)/2 gt2(3) gt2(4)]);
                tmplPos = [tmplPos;pos];
            end
            if isempty(tmplNeg)
                neg=([gt2(1)+gt2(3)/2 gt2(2)+gt2(4)/2 gt2(3) gt2(4)]);
                tmplNeg = [tmplNeg;neg];
            end
            tmplPos = double(tmplPos);
            tmplNeg = double(tmplNeg);
            configGlobalParam;
            opt = myconfig;
            seq.opt = opt;
            if ~isempty(tmplPos)&&~isempty(tmplNeg)
            [dataPos, seq.opt] = globalParam.FeatureExtractor(single(patch), tmplPos, seq.opt);
            [dataNeg, seq.opt] = globalParam.FeatureExtractor(single(patch), tmplNeg, seq.opt);
            svm_model   = globalParam.ObservationModelTrain(dataPos, dataNeg, seq.opt);   
            end