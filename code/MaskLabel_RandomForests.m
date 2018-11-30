clc; clear; close all;

train_dirs = {
    'D:\__Atlas__\data\07119\masks',
    'D:\__Atlas__\data\16465\masks',
    'D:\__Atlas__\data\16609\masks',
    'D:\__Atlas__\data\21780\masks',
    'D:\__Atlas__\data\21899\masks',
    'D:\__Atlas__\data\22878\masks',
    'D:\__Atlas__\data\23036\masks',
    'D:\__Atlas__\data\24159\masks'
    };
TrainTable = gather_train_data(train_dirs);
% ClassTreeEns = fitensemble(TrainTable(:,2:end),TrainTable(:,1),...
%    'AdaBoostM2',10000,'Tree');
ClassTreeEns = TreeBagger(100,TrainTable(:,2:end),TrainTable(:,1),'OOBPrediction','On',...
    'Method','classification')

if 1
figure;
oobErrorBaggedEnsemble = oobError(ClassTreeEns);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';
end
if 0
rsLoss = resubLoss(ClassTreeEns,'Mode','Cumulative');

plot(rsLoss);
xlabel('Number of Learning Cycles');
ylabel('Resubstitution Loss');
end
cd('D:\__Atlas__\model_saves')
save('RandomForestClassifier.mat','ClassTreeEns')
function TrainTable = gather_train_data(train_dirs)
label_str = {'DG','CA3','CA2','CA1','SUB','preSUB','paraSUB'};
TrainTable  = cell2table(cell(0,9), 'VariableNames', {'Label', 'YPOS', 'Area', 'CentroidX','CentroidY','MajAx','MinAx','Orientation','Solidity'});

for f =1:numel(train_dirs)
    cd(train_dirs{f})
    sections = file('*.png');
    for s = 1:numel(sections)
        mask = imread(sections{s});
        inds = find(mask>0);
        [I,J] = ind2sub(size(mask),inds);
        [TFORM, ~, ~, ~, ~, MU] = pca([I,J]);
        for sr = 1:7
            cc = bwconncomp(mask==sr,4);
            rem = cellfun(@numel,cc.PixelIdxList)<4;
            cc.PixelIdxList(rem) = [];
            cc.NumObjects = cc.NumObjects-sum(rem);
            if cc.NumObjects
                stats = regionprops(cc,'all');
                addCell = {};
                for i = 1:numel(stats)
                    XY = [fliplr(stats(i).Centroid)-MU]';
                    Center = TFORM*XY;
                    addCell(i,:) = {label_str{sr},s/numel(sections),stats(i).Area,...
                        Center(1),Center(2),...
                        stats(i).MajorAxisLength,stats(i).MinorAxisLength,...
                        stats(i).Orientation,stats(i).Solidity};
                end
                TrainTable = [TrainTable;addCell];
            end
        end
    end
end
end