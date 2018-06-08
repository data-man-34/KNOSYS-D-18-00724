clear;clc;close all;
%   title={Hierarchical Feature Selection with Graph Regularization Based on Subtree for Image Classification },
%   author={Tuo, Qianjuan and Zhao, Hong and Hu, Qinghua},
%% Load dataset
 load('AWAphogTrain.mat');
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
 [feature] = HFSGR_fast(X, Y, numSelected, tree, alpha, beita, noLeafNode, F, 0);
  t=toc;
% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test feature
load ('AWAphogTest.mat');begin1= 25;step1 = 25;end1 = 75;
 data_array = double(data_array);
[X, Y] = creatSubTablezh(data_array, tree);
numFolds =10;
k=1;
EachnodeFeaNmber=zeros(1,length(noLeafNode));
for j=begin1:step1:end1
    accuracyMean_selected(k,1)= j;
    accuracyStd_selected(k,1)=j;
    ii=2;
    for i = 1:length(noLeafNode)
        x=X{noLeafNode(i)};
       [N,~]=size(x);%Record the sample number of each non-leaf node
       EachnodeFeaNmber(1,i)=N;
        feature_slct = feature{noLeafNode(i)};
        feature_slct = feature_slct(1:j);
        x = x(:,feature_slct);
        y = Y{noLeafNode(i)};
        indices = crossvalind('Kfold',length(y),numFolds);
         [accuracyMean_selected(k,ii),accuracyStd_selected(k,ii)] = Kflod_multclass_svm_testParameters([x,y],numFolds,1,indices,tree);
%          [accuracyMean_selected(k,ii),accuracyStd_selected(k,ii)] = Kflod_multclass_knn_testParameters([x,y],numFolds,indices,1,tree); 
        ii=ii+1;    
    end
    k=k+1;
end
accuracyMean_selected=accuracyMean_selected(:,2:end);
Accuracy=(accuracyMean_selected*EachnodeFeaNmber')/sum(EachnodeFeaNmber);%Weighted average of classification accuracy
 

