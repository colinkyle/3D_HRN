clc; clear; close all
mask_dir = 'G:\VSI_DATA\MMU 07119\__masks';
data_dir = 'D:\__Atlas__\data\07119\masks';

mask_vals = [10, 30, 50, 70, 100, 130, 160];
mask_names = {'DG','CA3','CA2','CA1','SUB','preSUB','paraSUB'};
cd(mask_dir)

[~,imgs] = file('*.tif');
for im = 6:numel(imgs)
    cd(mask_dir)
    img = rgb2gray(imread(imgs{im}));
    img(img>200)=0;
    [vals,img] = quantize(img);
    MASKS = [];
    for v = 1:numel(vals)
        [m,i] = min(abs(mask_vals - double(vals(v))));
        if m>0
            disp(imgs{im})
            disp(vals(v))
            disp(mask_names{i})
            disp(' ')
        end
        if ~isfield(MASKS,mask_names{i})
            MASKS.(mask_names{i}) = uint8(img==vals(v));
        else
            MASKS.(mask_names{i}) = uint8(MASKS.(mask_names{i})) +  uint8(img==vals(v));
        end
    end
    fnames = fieldnames(MASKS);
    for m = 1:numel(fnames)
        i = find(strcmp(mask_names,fnames{m}));
        CC = bwconncomp(MASKS.(fnames{m}));
        comp_sizes = cellfun(@numel,CC.PixelIdxList);
        CC.PixelIdxList(comp_sizes<300)=[];
        MASKS.(fnames{m}) = zeros(size(img));
        MASKS.(fnames{m})(cat(1,CC.PixelIdxList{:})) = i;
        MASKS.(fnames{m}) = dwn_smpl(MASKS.(fnames{m}));
    end
    img = zeros([265,257]);
    fields = fieldnames(MASKS);
    for f = 1:numel(fields)
        img = img+ MASKS.(fields{f});
    end
    
    figure(1)
    imagesc(img)
    drawnow
    cd(data_dir)
    [path,fname,ext] = fileparts(imgs{im});
    imwrite(uint8(img),[fname,'.png'])
end

    


function img = dwn_smpl(img)
    img = imresize(img,1/95,'nearest');
    img = padarray(img,[round((265-size(img,1))/2),round((257-size(img,2))/2)],'replicate','both');
    img(265,257,:)=img(end,end,:);
    img = img(1:265,1:257,:);
end
function [fids,nums] = get_subject_nums(dir,pattern)
cd(dir)
[~,fids] = folder(pattern);
nums = {};
i=1;
for inds = cellfun(@(S)isstrprop(S,'digit'),fids,'UniformOutput',false)'
    nums = [nums,{fids{i}(inds{1})}];
    i=i+1;
end

end

function [vals,img] = quantize(img)
vals = unique(img(img(:)>0));
for v = 1:numel(vals)
    cnts = sum(img(:)==vals(v));
    if cnts<100
        img(img==vals(v)) = 0;
    end
end
vals = unique(img(img(:)>0));
end