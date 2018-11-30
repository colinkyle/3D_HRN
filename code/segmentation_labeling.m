% segmentation labeling ...
clc; clear; close all
mask_dir = 'D:\__Atlas__\data\35765\masks\35765\right';
vsi_dir = 'G:\VSI_DATA\MMU 35765';
save_dir = 'D:\__Atlas__\data\35765\masks';

cd(mask_dir)
[~,psds] = file('*.psd');
loadHandle = @(x)readpsd(x);
if isempty(psds)
    [~,psds] = file('*.tiff');
    loadHandle = @(x)imread(x);
end
cd(vsi_dir)
[~,vsis] = file('*.tiff');
dmat = NaN(numel(psds),numel(vsis));

for i = 1:numel(psds)
    for j = 1:numel(vsis)
        dmat(i,j) = EditDist(psds{i},vsis{j});
    end
    match = find(dmat(i,:)==min(dmat(i,:)));
    if numel(match)==1
        imgs(i).psd = psds{i};
        imgs(i).vsi = vsis{match};
    else
        disp(vsis{match})
        disp(psds{i})
        error('file name resolution required')
    end
end

if numel(imgs) ~= numel(psds)
    error('file name resolution required (imgs vs psds)')
end

for p = 1:numel(imgs)
    cd(mask_dir)
    img = loadHandle(imgs(p).psd);
    figure(1)
    imagesc(img)
    drawnow
    
    % check size
    cd(vsi_dir)
    info = imfinfo(imgs(p).vsi);
    
    if round(info.Height/size(img,2)) == 4
        bgMin = 5000;
        bgThresh = 2000000;
    elseif round(info.Height/size(img,2)) == 2
        bgThresh = 2000000*4;
        bgMin = 5000*4;
    else
        disp(imgs(p).psd)
        disp(['p =',num2str(p)])
        error('mask size doesnt make sense')
    end
    
    % get pure black, red
    BW = (img(:,:,1)<13) & (img(:,:,2)<13) & (img(:,:,3)<13);
    RED = (img(:,:,1)>200) & (img(:,:,2)<13) & (img(:,:,3)<13);
    BW = ~(BW | RED);
    % edge link
    BW = ~filledgegaps(~BW,20);
    
    % remove small regions (drawing errors)
    BW = bwareaopen(BW,bgMin);
    
    %[edgelist edgeim, etype] = edgelink(BW);
    % remove large regions (background)
    cc = bwconncomp(BW,4);
    
    numPix = cellfun(@numel,cc.PixelIdxList);
    inds = find(numPix>bgThresh);
    cc.PixelIdxList(inds) = [];
    cc.NumObjects = cc.NumObjects - numel(inds);
    
    numPix = cellfun(@numel,cc.PixelIdxList);
    inds = find(numPix<bgMin);
    cc.PixelIdxList(inds) = [];
    cc.NumObjects = cc.NumObjects - numel(inds);
    
    if cc.NumObjects==0
        disp(imgs(p).psd)
        disp(['p =',num2str(p)])
        warning('Mask did not create solid objects')
        continue
    elseif cc.NumObjects <7
        disp('Low Object Count:')
        disp(imgs(p).psd)
        disp([num2str(cc.NumObjects),' objects'])
        disp(' ')
    end
    
    labeled = rot90(labelmatrix(cc),-1);
    
    figure(2)
    imagesc(labeled)
    drawnow
    
    % go get matching image from vsi_dir
    
    if round(info.Height/size(labeled,1)) == 4
        labeled = imresize(labeled,2,'nearest');
    end
    BigImg = zeros(info.Height,info.Width);
    BigImg(end-size(labeled,1)+1:end,1:size(labeled,2)) = labeled;
    % down sample
    MASKS = [];
    for m = 1:max(labeled(:))
        MASKS.(['a',num2str(m)]) = double(m)*double(BigImg==m);
        MASKS.(['a',num2str(m)]) = dwn_smpl(MASKS.(['a',num2str(m)]));
    end
    img = zeros([265,257]);
    fields = fieldnames(MASKS);
    for f = 1:numel(fields)
        img = img+ MASKS.(fields{f});
    end
    figure(3)
    imagesc(img)
    drawnow
    cd(save_dir)
    [path,fname,ext] = fileparts(imgs(p).psd);
    imwrite(uint8(img),[fname,'.png'])
    
end

function img = dwn_smpl(img)
    img = imresize(img,1/95,'nearest');
    img = padarray(img,[round((265-size(img,1))/2),round((257-size(img,2))/2)],'replicate','both');
    img(265,257,:)=img(end,end,:);
    img = img(1:265,1:257,:);
end