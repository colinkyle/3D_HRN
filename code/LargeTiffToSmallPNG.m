clc; clear; close all

cd('D:\__Atlas__\data\16465_\histology\_images')
[~,imgs] = file('*.png');
for i = 1:numel(imgs)
    cd('D:\__Atlas__\data\16465_\histology\_images')
    img = imread(imgs{i});
    img = imresize(img,1/95);
    img = padarray(img,[round((265-size(img,1))/2),round((257-size(img,2))/2)],0,'both');
    img(265,257,:)=0;
    img = img(1:265,1:257,:);
    annotation = img;
    annotation = rgb2gray(annotation);
    annotation(annotation<=10)=0;
    annotation(annotation>0)=1;
    cd('D:\__Atlas__\Segmentation\FCN.tensorflow\traindata\annotations\training')
    imwrite(annotation,imgs{i})
    cd('D:\__Atlas__\data\16465_\histology')
    img = imread(imgs{i});
    cd('D:\__Atlas__\Segmentation\FCN.tensorflow\traindata\images\training')
    imwrite(img,imgs{i});
end