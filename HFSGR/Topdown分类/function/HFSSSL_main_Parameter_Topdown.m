 clear;clc;close all;
%   load('DDTrain.mat')
%  load('Protein194TrainN.mat')
%  load('CLEFTrain_N.mat')k
%  load('AWAphogTrain.mat')
   load ('VOCTrain');
%   load('SUNTrain.mat')
 Level_num  = max(tree(:,2));
 [X, Y] = creatSubTablezh(data_array, tree);
 [noLeafNode]=NonLeafNode(tree);
alpha =10;
% beita =0;
 beita =[1e-2,1e-1,1,1e1,1e2];
 mu =[1e-2,1e-1,1,1e1,1e2];
%   L=[54,60,66,72,78];
  L=[48,96,144,192,240];
 %% Feature selection
  tic;
 for i = 1:length(alpha)
    for k = 1:length(beita)%3
        for kk = 1:length(mu)%5
            for j=1:length(L)
                [Select_feature(j,:),obj(j,:)] = HFSSSL(X, Y, tree,L(j),alpha(i), beita(k), mu(kk), noLeafNode, 1);
            end
            mu_cell{1,kk} = Select_feature;
            mu_cell_obj{1,kk} =obj;
        end
        beita_cell(k,:) = mu_cell;
        beita_cell_obj(k,:) = mu_cell_obj;
    end
   Select_feature_Parameter{1,i}=beita_cell;
   Obj_Parameter{1,i} = beita_cell_obj;
end
  t=toc;
%   save DD_Parameter Select_feature_Parameter Obj_Parameter
%   save DD_ParameterNoB Select_feature_Parameter Obj_Parameter
%   save F194_Parameter Select_feature_Parameter Obj_Parameter
%    save F194_ParameterNoB Select_feature_Parameter Obj_Parameter
%   save CLEF_Parameter Select_feature_Parameter Obj_Parameter
%    save CLEF_ParameterNoB Select_feature_Parameter Obj_Parameter
%  save AWAphog_Parameter Select_feature_Parameter Obj_Parameter
% save AWAphog_Parameterbeita Select_feature_Parameter Obj_Parameter
   save VOC_Parameter Select_feature_Parameter Obj_Parameter
%  save VOC_ParameterNoB Select_feature_Parameter Obj_Parameter
%  save SUN_Parameter Select_feature_Parameter Obj_Parameter
%  save SUN_ParameterNoB Select_feature_Parameter Obj_Parameter
 %% Test feature
%   load ('DDTest.mat');  begin1=54;step1 = 6;end1 = 78;
%  load('Protein194TestN.mat'); begin1=54;step1 = 6;end1 = 78;
%  load('CLEFTest_N.mat'); begin1=54;step1 = 6;end1 = 78;
%  load('AWAphogTest.mat');begin1=20;step1 = 20;end1 =200;
  load ('VOCTest');begin1=48;step1 = 48;end1 =240;
%  load('SUNTest.mat');begin1=40;step1 = 40;end1 =200;
[M,N]=size(data_array);
%    load('DD_Parameter.mat')
%  load('DD_ParameterNoB.mat')
%  load('F194_Parameter.mat')
%  load('F194_ParameterNoB.mat')
% load('CLEF_Parameter.mat')
%  load('CLEF_ParameterNoB.mat')
% load('AWAphog_Parameter.mat')
% load('AWAphog_Parameterbeita.mat')
  load('VOC_Parameter.mat')
% load('VOC_ParameterNoB.mat')
% load('SUN_Parameter.mat')
% load('SUN_ParameterNoB.mat')
numFolds =10;
topDown = 1;
for i=1:length(alpha)
   for k=1:length(beita)
     for kk = 1:length(mu)
         h=1;
         for j=begin1:step1:end1 
          accuracyMean(h,1)=j;
          accuracyStd(h,1)=j;
          F_LCAMean(h,1)=j;
          FHMean(h,1)=j;
          indices = crossvalind('Kfold',data_array(1:M,N),numFolds);
          [accuracyMean(h,kk+1),accuracyStd(h,kk+1),F_LCAMean(h,kk+1),FHMean(h,kk+1)] = FS_Kflod_TopDownClassifier(data_array,numFolds,tree,topDown,Select_feature_Parameter{1,1}{k,kk}(h,:),j,indices);
          h=h+1;
         end
     end
     beita_accuracy{1,k} = accuracyMean; beita_accuracyStd{1,k}= accuracyStd; beita_F_LCAMean{1,k}= F_LCAMean; beita_FHMean{1,k}=FHMean;
   end
   Final_accuracyMean{i,:}=beita_accuracy;Final_accuracyStd{i,:}=beita_accuracyStd; Final_F_LCAMean{i,:}=beita_F_LCAMean;  Final_FHMean{i,:}=beita_FHMean;
end



 