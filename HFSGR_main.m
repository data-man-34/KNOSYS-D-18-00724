clear;clc;close all;
%   title={Hierarchical feature selection with graph regularization based on subtree},
%   author={Tuo, Qianjuan and Zhao, Hong and Hu, Qinghua},
%% Input:
%  data_array - the data matrix with the label
%      tree - tree structure among the classes
%     alpha - the parameter of optimal
%     beita - the tradeoff parameter  
%% Output:
%    accuracyMean_selected - classification accuracy  
%% Load dataset
%  load ('VOCTrain');
%  load('SUNTrain.mat');
%  load ('LandscapeTrain.mat');
%  load('AWAphogTrain.mat');
%  load('ilsvrc65Train.mat')
%  load('Cifar100Train.mat')
   load('DDTrain.mat')
%  load('F194Train.mat')
%% initialization
alpha =1;
beita =1;
Level_num  = max(tree(:,2));
[~, numSelected] = size(data_array);
numSelected = numSelected -1;
%% Creat sub table
[X, Y] = creatSubTablezh(data_array, tree);
% load('LandscapeTrain_X.mat')%只在用Landscape进行实验
% load('LandscapeTrain_Y.mat')
%% Feature selection
  tic;
  [noLeafNode,F]=SubtreeRelationMatrix_F(tree);% Creat subtree relation matrix F
 [feature] = HFSGR_fast(X, Y, numSelected, tree, alpha, beita, noLeafNode, F, 1);
  t=toc;
% save VOCTrainfeature_HFSGR feature
%  save SUNTrainfeature_HFSGR feature
%   save LandscapeTrainfeature_HFSGR feature
%   save AWAphogTrainfeature_HFSGR feature
%   save ilsvrc65Trainfeature_HFSGR feature
%   save Cifar100Trainfeature_HFSGR feature
   save DDTrainfeature_HFSGR feature
%  save F194Trainfeature_HFSGR feature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Test feature
%  load('SUNTrainfeature_HFSGR.mat')
%  load('VOCTrainfeature_HFSGR.mat')
%  load('AWAphogTrainfeature_HFSGR.mat')
%  load('Cifar100Trainfeature_HFSGR.mat')
% load('ilsvrc65Trainfeature_HFSGR.mat')
  load('DDTrainfeature_HFSGR.mat')
%  load('F194Trainfeature_HFSGR.mat')
% % %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  load ('VOCTest');begin1=100;step1 = 100;end1 =300;
%  load ('SUNTest.mat');begin1= 410;step1 = 410;end1 = 1230;
% load ('LandscapeTest.mat');begin1= 50;step1 = 50;end1 = 150;
%  load ('AWAphogTest.mat');begin1= 25;step1 = 25;end1 = 75;
% load('ilsvrc65Test.mat');begin1= 410;step1 = 410;end1 = 1230;
% load('Cifar100Test.mat');begin1= 410;step1 = 410;end1 = 1230;
load('DDTest.mat');begin1= 48;step1 = 48;end1 = 144;% Corresponding the 10%, 20% and 30% of feature
% load('F194Test.mat');begin1= 48;step1 = 48;end1 = 144;
%  data_array = double(data_array);
[X, Y] = creatSubTablezh(data_array, tree);%split datasets based on the number of non-leaf nodes
% load('LandscapeTest_X.mat')
% load('LandscapeTest_Y.mat')
numFolds =10;%10-fold cross validation
k=1;
EachnodeFeaNmber=zeros(1,length(noLeafNode));
for j=begin1:step1:end1
    accuracyMean_selected(k,1)= j;
    accuracyStd_selected(k,1)=j;
    ii=2;
    for i = 1:length(noLeafNode)
        x=X{noLeafNode(i)};
       [N,~]=size(x);%Record the number of samples for each non-leaf node
       EachnodeFeaNmber(1,i)=N;
        feature_slct = feature{noLeafNode(i)};
        feature_slct = feature_slct(1:j);
        x = x(:,feature_slct);
        y = Y{noLeafNode(i)};
        indices = crossvalind('Kfold',length(y),numFolds);
         [accuracyMean_selected(k,ii),accuracyStd_selected(k,ii)] = Kflod_multclass_svm_testParameters([x,y],numFolds,1,indices,tree);
        % classification accuracy of SVM
        %[accuracyMean_selected(k,ii),accuracyStd_selected(k,ii)] = Kflod_multclass_knn_testParameters([x,y],numFolds,indices,1,tree); 
        % classification accuracy of SVM
ii=ii+1;    
    end
    k=k+1;
end
accuracyMean_selected=accuracyMean_selected(:,2:end);
Accuracy=(accuracyMean_selected*EachnodeFeaNmber')/sum(EachnodeFeaNmber);%weighted average of classification accuracy
%  save SUN_HFSGR_SVM Accuracy; 
% save SUN_HFSGR_KNN Accuracy
%  save VOC_HFSGR_KNN Accuracy
%  save AWAphog_HFSGR_KNN Accuracy
% save AWAphog_HFSGR_KNN Accuracy
% save ilsvrc65_HFSGR_KNN Accuracy;
% save Cifar100_HFSGR_KNN Accuracy;
% save Cifar_HFSGR_SVM Accuracy;
  save DD_HFSGR_SVM_ Accuracy;
%  save DD_HFSGR_KNN Accuracy;
%  save F194_HFSGR_KNN Accuracy;
%  save F194_HFSGR_SVM Accuracy;

