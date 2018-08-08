clear;clc;close all;
% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test feature
% load('AWAphogTrainfeature_HiFSRR.mat')
% load('VOCTrainfeature_HiFSRR.mat')
% load('SUNTrainfeature_HiFSRR.mat')
% load('CifarTrainfeature_HiFSRR.mat')
% load('ilsvrc65Trainfeature_HiFSRR.mat')
% load('Cifar100Trainfeature_HiFSRR.mat')
%  load('DDTrainfeature_HiFSRR.mat')
 load('F194Trainfeature_HiFSRR.mat')
% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  load('AWAphogTest.mat');begin1= 25;step1 = 25;end1 = 125;
%  load('VOCTest.mat');begin1= 100;step1 = 100;end1 = 500;
%  load ('SUNTest.mat');begin1= 410;step1 = 410;end1 = 2050;
% load('ilsvrc65Test.mat');begin1= 410;step1 = 410;end1 = 1230;
% load('Cifar100Test.mat');begin1= 410;step1 = 410;end1 = 1230;
% load('CifarTest.mat');begin1= 50;step1 = 50;end1 = 250;
% load('DDTest.mat');begin1= 48;step1 = 48;end1 = 144;
load('F194Test.mat');begin1= 48;step1 = 48;end1 = 144;
data_array = double(data_array);
 [M,N]=size(data_array);
 numFolds =10;
 k=1;
 ii=2;
 topDown = 1;
% data_array = double(data_array);
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
% save AWAphog_HiFSRR_F1 FHMean
% save VOC_HiFSRR_F1 FHMean
% save SUN_HiFSRR_F1 FHMean
% save Cifar_HiFSRR_F1 FHMean
% save ilsvr65_HiFSRR_F1 FHMean
% save Cifar100_HiFSRR_F1 FHMean
% save DD_HiFSRR_F1 FHMean
save F194_HiFSRR_F1 FHMean
