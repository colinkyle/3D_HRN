clc
clear
%% Set path, variables etc...
vsi_dir = 'G:\VSI_DATA\MMU 35717\MMU35717  B1S2 321N - 341N';
save_dir = 'G:\VSI_DATA\MMU 35717\MMU35717  B1S2 321N - 341N';
%mkdir(save_dir);
javaaddpath('~/SYNC/MyFunctions/bfmatlab')

cd(vsi_dir)
% step 1 do folders
imname = file('*.vsi');
%imname(1) = [];
for iSlide = 1:numel(imname)
    cd(vsi_dir)
    
    %loop and get resolution
    info = infoVSI(imname{iSlide},0);
    clearvars res
    for i = 1:info.numSeries
        info = infoVSI(imname{iSlide},i-1);
        res(i,:) = info.shape;
    end
    [~,ser] = min(sum(abs(res(:,1:2)-[18000,22000]),2));
    ser = ser-1;
    %ser = 2;
    disp(imname{iSlide})
    disp(ser)
    disp(res(ser+1,:))
    try
        info = infoVSI(imname{iSlide},ser);
    catch
        continue
    end
    x = 1;
    xspan = info.shape(1);
    testspan = ceil(info.shape(2)/3);
    y = 1;
    yspan(1) = testspan;
    for i = 2:3
        y(i) = y(i-1)+testspan;
        yspan(i) = testspan;
        if y(i)+testspan>info.shape(2)
            yspan(i) = testspan - (y(i)+testspan-info.shape(2)-1);
        end
    end
    clearvars I1 I2 I3 I
    parfor iPlane = 1:3
        I1 = loadVSI_java(imname{iSlide},ser,iPlane,x,y(1),xspan,yspan(1));
        I2 = loadVSI_java(imname{iSlide},ser,iPlane,x,y(2),xspan,yspan(2));
        I3 = loadVSI_java(imname{iSlide},ser,iPlane,x,y(3),xspan,yspan(3));
        I(:,:,iPlane) = cat(1,I1,I2,I3);
    end
    I = rot90(I,2);
    cd(save_dir)
    imwrite(I,[imname{iSlide}(1:strfind(imname{iSlide},'.vsi')),'tiff']);
end
