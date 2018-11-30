cd('D:\__Atlas__\Segmentation\FCN.tensorflow\traindata\annotations\validation')

imgs = file('*.png');
for i = 1:numel(imgs)
    [path,name,ext] = fileparts(imgs{i});
    imname = [path,'\',name,'.png'];
    img = imread(imgs{i});
    img(img>0)=1;
    imwrite(img,imname,'PNG')
    
end