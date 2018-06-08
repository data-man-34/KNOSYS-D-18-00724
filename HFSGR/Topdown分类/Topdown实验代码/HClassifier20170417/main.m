%% Top-down classification with tree hierarchy in Madantory Leaf Node Prediction.
%% Train a multi-class logistic regression classifier for each node.
%% Usage: change the input dataSat on the top,
%%               and you can change the proportion of data for training and testing at parameter 'numTrain'.
%% For performance of the classifier, change the cmd parameter in multiclassMLNP_learning.m
%% For evaluation metrics, res_acc is accuracy,
%% Hieararchical precision and recall, Hieararchical LCA and TIE-Mean, information gain and hierarchical accuracy is at the bottom of the code

clear;
clc;

%% Load information of the dataset
% load Protein27;
load Protein27normalization
% data_array=data_array_train;
tic
topDown = 1;
numFolds = 5;
load ZHProtein27TrainFeature32;
[M,N]=size(data_array);
% indices = crossvalind('Kfold',M,numFolds);%//进行随机分包 for k=1:10//交叉验证k=10，10个包轮流作为测试集
% save Protein27Indices indices;
load Protein27Indices;
numberSel = 300;
[accuracyMean,accuracyStd,F_LCAMean,FHMean] = FS_Kflod_TopDownClassifier(data_array,numFolds,tree,topDown,feature1,numberSel,indices);
t=toc


