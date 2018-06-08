clear;clc;close all;
%Computer F1_meature
load('AWAphogTrain.mat')
%% initialization
alpha =1;
beita =1;
Level_num  = max(tree(:,2));
 [~, numSelected] = size(data_array);
numSelected = numSelected -1;
%% Creat sub table
[X, Y] = creatSubTablezh(data_array, tree);
%% Feature selection
  tic;
 [noLeafNode,F]=SubtreeRelationMatrix_F(tree);% Creat subtree relation matrix F
 [feature] = HFSGR_fast(X, Y, numSelected, tree, alpha, beita, noLeafNode, F, 1);
  t=toc;
% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test feature
%load('AWAphogTrainfeature_HFSGR.mat')
% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 load('AWAphogTest.mat');begin1= 25;step1 = 25;end1 =75;
data_array = double(data_array);
 [M,N]=size(data_array);
 numFolds =10;
 k=1;
 ii=2;
 topDown = 1;
for j=begin1:step1:end1
    accuracyMean_selected(k,1)= j;
    accuracyStd_selected(k,1)=j;
    F_LCAMean(k,1)=j;
    FHMean(k,1)=j;
    indices = crossvalind('Kfold',data_array(1:M,N),numFolds);
    [accuracyMean(k,ii),accuracyStd(k,ii),F_LCAMean(k,ii),FHMean(k,ii)] = FS_Kflod_TopDownClassifier(data_array,numFolds,tree,topDown,feature,j,indices);
     fprintf(['A0.1--Accurate rate:',num2str(accuracyMean(k,ii)),'¡À', num2str(accuracyStd(k,ii)),'\n']);
    fprintf(['F_LCAMean:',num2str(F_LCAMean(k,ii)),'\nFHMean:', num2str(FHMean(k,ii)),'\n']);
    k=k+1;
end
 save AWAphog_HFSGR_F1 FHMean
