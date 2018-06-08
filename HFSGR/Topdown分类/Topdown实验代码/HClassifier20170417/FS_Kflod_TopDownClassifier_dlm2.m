%% 10-fold
%% Written by Hong Zhao
% 2017-4-11
%% Input
%  data
%  numFolds
%  indices = varargin{4};
%% Output
function [accuracyMean,accuracyStd,F_LCAMean,FHMean] = FS_Kflod_TopDownClassifier_dlm2(data, numFolds, tree,flag, feature, numberSel,indices)
% if(length(varargin) == 4)
%     data = varargin{1};
%     numFolds = varargin{2};
%     tree = varargin{3};
%     flag = varargin{4};
%  %   indices = varargin{5};
% else
%     if(length(varargin)==5)
%         data = [varargin{1},varargin{2}];
%         numFolds = varargin{3};
%         tree = varargin{4};
%         flag = varargin{5};
%         %   indices = varargin{5};
%     end
% end
[M,N]=size(data);
accuracy_k = zeros(1,numFolds);
% indices = crossvalind('Kfold',data(1:M,N),numFolds);%//进行随机分包 for k=1:10//交叉验证k=10，10个包轮流作为测试集
%     save indices10001 indices;
%        load indices1000;

for k = 1:numFolds
    testID = (indices == k);%//获得test集元素在数据集中对应的单元编号
    trainID = ~testID;%//train集元素的编号为非test元素的编号
    test_data = data(testID,1:N-1);
    test_label = data(testID,N);
    train_data = data(trainID,:);
    
    %% Creat sub table
    [trainDataMod, trainLabelMod] = creatSubTable(train_data, tree);
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Train classifiers of all internal nodes
    numNodes = length(tree(:,1));%ZH: The total of all nodes.
    for i = 1:numNodes
        if (~ismember(i, tree_LeafNode(tree)))
            trainLabel = sparse(trainLabelMod{i});
            trainData = sparse(trainDataMod{i});
            selFeature = feature{i}(1:numberSel(i));
            [model{i}]  = train(double(trainLabel), sparse(trainData(:,selFeature(1:numberSel(i)))), '-c 2 -s 0 -B 1 -q');
        end
    end
    
    %%           Prediction       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (flag==1)
    [predict_label] = FS_topDownPrediction_dlm2(test_data, model, tree,feature,numberSel) ;
    else 
   [predict_label] = topDownPredictionComputeAll(test_data,test_label, model, tree);
    end
    %%          Envaluation       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   [PH(k), RH(k), FH(k)] = EvaHier_HierarchicalPrecisionAndRecall(test_label,predict_label',tree);
   [P_LCA(k),R_LCA(k),F_LCA(k)] = EvaHier_HierarchicalLCAPrecisionAndRecall(test_label,predict_label',tree);
   TIE(k) = EvaHier_TreeInducedError(test_label,predict_label',tree);
    accuracy_k(k) = EvaHier_HierarchicalAccuracy(test_label,predict_label', tree);%王煜
    
end
 accuracyMean = mean(accuracy_k);
 accuracyStd = std(accuracy_k);
F_LCAMean=mean(F_LCA);
FHMean=mean(FH);
end