clear;clc;close all;
% %   title={Hierarchical feature selection with graph regularization based on subtree},
% %   author={Tuo, Qianjuan and Zhao, Hong and Hu, Qinghua},
% % Load dataset
% load('ilsvrc65Train.mat')
% % load('CifarTrain.mat')
% % load('LandscapeTrain_X.mat')%只在用Landscape进行实验
% % load('LandscapeTrain_Y.mat')
% % load('VOCTrain.mat')
% % load('AWAphogTrain.mat')
% % load('SUNTrain.mat')
% %% initialization
% alpha =1;
% beita =1;
% Level_num  = max(tree(:,2));
% [~, numSelected] = size(data_array);
% % [~, numSelected] = size(X{1,94});%只在用Landscape进行实验
% numSelected = numSelected -1;
% %% Creat sub table
% [X, Y] = creatSubTablezh(data_array, tree);
% %% Feature selection
% tic;
% [noLeafNode,F]=SubtreeRelationMatrix_F(tree);% Creat subtree relation matrix F
% [feature] = HFSGR_fast(X, Y, numSelected, tree, alpha, beita, noLeafNode, F, 1);
% t=toc;
% save DDTrainfeature feature
% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test feature
% load('AWAphogTrainfeature_HFSGR.mat')
% load('VOCTrainfeature_HFSGR.mat')
% load('SUNTrainfeature_HFSGR.mat')
% load('CifarTrainfeature_HFSGR.mat')
% load('ilsvrc65Trainfeature_HFSGR.mat')
% load('Cifar100Trainfeature_HFSGR.mat')
%   load('DDTrainfeature_HFSGR.mat')
 load('F194Trainfeature_HFSGR.mat')
% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  load('VOCTest.mat');begin1= 100;step1 = 100;end1 = 300;
%  load('AWAphogTest.mat');begin1= 25;step1 = 25;end1 = 125;
%  load ('SUNTest.mat');begin1= 410;step1 = 410;end1 = 2050;
% load('ilsvrc65Test.mat');begin1= 410;step1 = 410;end1 = 1230;
% load('CifarTest.mat');begin1= 50;step1 = 50;end1 = 250;
% load('Cifar100Test.mat');begin1= 410;step1 = 410;end1 = 1230;
% load('DDTest.mat');begin1= 48;step1 = 48;end1 = 144;
load('F194Test.mat');begin1= 48;step1 = 48;end1 = 144;
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
     fprintf(['A0.1--Accurate rate:',num2str(accuracyMean(k,ii)),'±', num2str(accuracyStd(k,ii)),'\n']);
    fprintf(['F_LCAMean:',num2str(F_LCAMean(k,ii)),'\nFHMean:', num2str(FHMean(k,ii)),'\n']);
    k=k+1;
end
% save AWAphog_HFSGR_F1 FHMean
% save VOC_HFSGR_F1 FHMean
% save SUN_HFSGR_F1 FHMean
% save Cifar_HFSGR_F1 FHMean
% save ilsvr65_HFSGR_F1 FHMean
% save Cifar100_HFSGR_F1 FHMean
% save DD_HFSGR_F1 FHMean
save F194_HFSGR_F1 FHMean
