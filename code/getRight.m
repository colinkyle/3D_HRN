clc; clear;
animals = {'35481'};
for id = 1:numel(animals)
    cd(['G:\VSI_DATA\MMU ',animals{id}])
    imgs = file('*.tiff');
    
    for i = 1:numel(imgs)
        cd(['G:\VSI_DATA\MMU ',animals{id}])
        img = imread(imgs{i});
        img = img(round(size(img,1)/2):end,1:round(size(img,2)/2),:);
        img = imrotate(img,90);
        mkcd('right');
        [path,file,ext]=fileparts(imgs{i});
        imwrite(img,[file,ext]);
    end
end