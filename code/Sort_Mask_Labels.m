clc; clear; close all

%load classifier
cd('D:\__Atlas__\model_saves')
load RandomForestClassifier

mask_dir = 'D:\__Atlas__\data\32218\masks';
hist_dir = 'D:\__Atlas__\data\32218\histology\segmented';
cd(hist_dir)
hists = file('*.png');
cd(mask_dir)
masks = file('*.png');

for i = 1:numel(masks)
    for j = 1:numel(hists)
        dmat(i,j) = EditDist(masks{i},hists{j});
    end
    match = find(dmat(i,:)==min(dmat(i,:)));
    if numel(match)==1
        imgs(i).mask = masks{i};
        imgs(i).hist = hists{match};
    else
        disp(hists{match})
        disp(masks{i})
        error('file name resolution required')
    end
end

mask_names = {'DG', 'CA3', 'CA2', 'CA1', 'SUB', 'preSUB', 'paraSUB'};

% do edge case
i = 17%numel(imgs);
mask = imread(imgs(i).mask);
hist = imread(imgs(i).hist);

num_masks = double(max(mask(:)));
xl = 265;
xh = 0;
yl = 257;
yh = 0;

for m = 1:num_masks
    inds{m} = find(mask==m);
    [I,J] = ind2sub(size(mask),inds{m});
    xl = min([min(I),xl]);
    xh = max([max(I),xh]);
    yl = min([min(J),yl]);
    yh = max([max(J),yh]);
end

disp(' ')
disp(imgs(i).mask)
disp([num2str(num_masks),' masks'])

figure(1)
imagesc(rot90(30*mask(xl-10:xh+10,yl-10:yh+10))+rot90(hist(xl-10:xh+10,yl-10:yh+10)))
drawnow;

while ~all(cellfun(@isempty,inds))
    m_list = find(~cellfun(@isempty,inds));
    num_masks = numel(m_list);
    figure(2)
    clf
    subpl = 1;
    for m = m_list
        if ~isempty(inds{m})
            subplot(ceil(num_masks/3),3,subpl)
            subpl=subpl+1;
            
            hist2 = hist;
            hist2(inds{m}) = 0;
            imagesc(rot90(hist2(xl-10:xh+10,yl-10:yh+10)))
            title(num2str(m))
            drawnow
        else
            
        end
    end
    
    dg = input('DG: ');
    ca3 = input('CA3: ');
    ca2 = input('CA2: ');
    ca1 = input('CA1: ');
    sub = input('SUB: ');
    presub = input('preSUB: ');
    parasub = input('paraSUB: ');
    
    
    if ~isempty(dg)
        mask(inds{dg}) = 1;
        inds{dg} = [];
    end
    if ~isempty(ca3)
        mask(inds{ca3}) = 2;
        inds{ca3} = [];
    end
    if ~isempty(ca2)
        mask(inds{ca2}) = 3;
        inds{ca2} = [];
    end
    if ~isempty(ca1)
        mask(inds{ca1}) = 4;
        inds{ca1} = [];
    end
    if ~isempty(sub)
        mask(inds{sub}) = 5;
        inds{sub} = [];
    end
    if ~isempty(presub)
        mask(inds{presub}) = 6;
        inds{presub} = [];
    end
    if ~isempty(parasub)
        mask(inds{parasub}) = 7;
        inds{parasub} = [];
    end
end
imwrite(mask,imgs(i).mask)
mask_old = mask;
for i = i-1:-1:1 % i-1
    mask = imread(imgs(i).mask);
    hist = imread(imgs(i).hist);
    
    
    xl = 265;
    xh = 0;
    yl = 257;
    yh = 0;
    
    
    
    % match_mask
    mask = match_masks(mask_old,mask,ClassTreeEns,i/numel(imgs));
    num_masks = double(max(mask(:)));
    for m = 1:num_masks
        inds{m} = find(mask==m);
        [I,J] = ind2sub(size(mask),inds{m});
        xl = min([min(I),xl]);
        xh = max([max(I),xh]);
        yl = min([min(J),yl]);
        yh = max([max(J),yh]);
    end
    disp(' ')
    disp(imgs(i).mask)
    disp([num2str(num_masks),' masks'])
    
    figure(1)
    imagesc(rot90(30*mask(xl-10:xh+10,yl-10:yh+10))+rot90(hist(xl-10:xh+10,yl-10:yh+10)))
    drawnow;
    
    % must deal with mask not associated with masks and missing mask
    
    while ~all(cellfun(@isempty,inds))
        m_list = find(~cellfun(@isempty,inds));
        num_masks = numel(m_list);
        figure(2)
        clf
        subpl = 1;
        for m = m_list
            if ~isempty(inds{m})
                subplot(ceil(num_masks/3),3,subpl)
                subpl=subpl+1;
                
                hist2 = hist;
                hist2(inds{m}) = 0;
                imagesc(rot90(hist2(xl-10:xh+10,yl-10:yh+10)))
                title(num2str(m))
                drawnow
            else
                
            end
        end
        
        dg = input('DG: ');
        ca3 = input('CA3: ');
        ca2 = input('CA2: ');
        ca1 = input('CA1: ');
        sub = input('SUB: ');
        presub = input('preSUB: ');
        parasub = input('paraSUB: ');
        remov = input('remove: ');
        
        if ~isempty(remov)
            mask(inds{remov}) = 0;
            inds{remov} = [];
        end
        if ~isempty(dg)
            mask(inds{dg}) = 1;
            inds{dg} = [];
        end
        if ~isempty(ca3)
            mask(inds{ca3}) = 2;
            inds{ca3} = [];
        end
        if ~isempty(ca2)
            mask(inds{ca2}) = 3;
            inds{ca2} = [];
        end
        if ~isempty(ca1)
            mask(inds{ca1}) = 4;
            inds{ca1} = [];
        end
        if ~isempty(sub)
            mask(inds{sub}) = 5;
            inds{sub} = [];
        end
        if ~isempty(presub)
            mask(inds{presub}) = 6;
            inds{presub} = [];
        end
        if ~isempty(parasub)
            mask(inds{parasub}) = 7;
            inds{parasub} = [];
        end
    end
    imwrite(mask,imgs(i).mask)
    mask_old = mask;
end


function [mask_new] = match_masks(mask_old,mask_new, ClassTreeEns,s)
[optimizer, metric] = imregconfig('monomodal');
optimizer.MaximumIterations=1000;
tform = imregtform(double(mask_old>0),double(mask_new>0),'translation',optimizer,metric);
moved = imwarp(mask_old,tform,'nearest');
PredictTable  = cell2table(cell(0,8), 'VariableNames', {'YPOS', 'Area', 'CentroidX','CentroidY','MajAx','MinAx','Orientation','Solidity'});


[I,J] = ind2sub(size(mask_new),find(mask_new>0));
[TFORM, ~, ~, ~, ~, MU] = pca([I,J]);

for i = 1:max(moved(:))
    inds_o{i} = find(moved==i);
end
co = 1;
for i = 1:max(mask_new(:))
    %inds{i} = find(mask_new==i);
    cc = bwconncomp(mask_new==i,4);
    rem = cellfun(@numel,cc.PixelIdxList)<4;
    cc.PixelIdxList(rem) = [];
    cc.NumObjects = cc.NumObjects-sum(rem);
    for pp = 1:cc.NumObjects
        inds{co} = cc.PixelIdxList{pp};
        co = co+1;
    end
    if cc.NumObjects
        stats = regionprops(cc,'all');
        addCell = {};
        for ii = 1:numel(stats)
            XY = [fliplr(stats(ii).Centroid)-MU]';
            Center = TFORM*XY;
            addCell(ii,:) = {s,stats(ii).Area,...
                Center(1),Center(2),...
                stats(ii).MajorAxisLength,stats(ii).MinorAxisLength,...
                stats(ii).Orientation,stats(ii).Solidity};
        end
        PredictTable = [PredictTable;addCell];
    end
end
inds(cellfun(@numel,inds)<4) = [];
try
    simMat = zeros(numel(inds),numel(inds_o));
    for i = 1:numel(inds)
        for j = 1:numel(inds_o)
            simMat(i,j) = 2*numel(intersect(inds{i},inds_o{j}))/...
                (numel(inds{i})+numel(inds_o{j}));
            
        end
    end
catch
    simMat = zeros(numel(inds),7);
end

if size(simMat,2)<7
    simMat(:,7) = 0;
end
[Label,Posterior] = predict(ClassTreeEns,PredictTable);
for l = 1:7
    if strcmp(ClassTreeEns.ClassNames{l},'DG')
        sorted(l)=1;
    end
    if strcmp(ClassTreeEns.ClassNames{l} , 'CA3')
        sorted(l)=2;
    end
    if strcmp(ClassTreeEns.ClassNames{l} , 'CA2')
        sorted(l)=3;
    end
    if strcmp(ClassTreeEns.ClassNames{l} , 'CA1')
        sorted(l)=4;
    end
    if strcmp(ClassTreeEns.ClassNames{l} , 'SUB')
        sorted(l)=5;
    end
    if strcmp(ClassTreeEns.ClassNames{l} , 'preSUB')
        sorted(l)=6;
    end
    if strcmp(ClassTreeEns.ClassNames{l} , 'paraSUB')
        sorted(l)=7;
    end
end
Posterior = Posterior(:,sorted)./sum(Posterior,2);
simMat = simMat./[sum(simMat,2)+eps];

simMat = .1*simMat + Posterior;
[~,MAP] = max(simMat,[],2);
%clean MAP
for i = 1:max(MAP)
    if sum(MAP==i)>1
        doubles = find(MAP==i);
        all_others = find(MAP>i);
        to_add = union(doubles(2:end),all_others);
        MAP(to_add) = MAP(to_add)+1;
    end
end

%[~,order] = sort(MAP);
%inds = inds(MAP);

for i = 1:numel(inds)
    mask_new(inds{i})=MAP(i);
end
disp(' ')

end
