%% 训练阶段和HFSSL_main_Parameter阶段相同，得到XX_Parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc;close all;
%% Test feature
% load ('DDTest.mat'); begin1= 16;step1 = 8;end1 = 48;
%  load('Protein194TestN.mat'); begin1= 16;step1 = 8;end1 = 48;
% load('CLEFTest_N.mat'); begin1= 16;step1 = 8;end1 = 48;
% load ('VOCTest');begin1=40;step1 = 40;end1 =200;
% load('AWAphogTest.mat');begin1=40;step1 = 40;end1 =200;
load('SUNTest.mat');begin1=40;step1 = 40;end1 =200;
[noLeafNode]=NonLeafNode(tree);
[X, Y] = creatSubTablezh(data_array, tree);
alpha =[1e-2,1e-1,1,1e1,1e2];
beita =[1e1,1e2,1e3,1e4,1e5,1e6];
mu=[1e1,1e2,1e3,1e4,1e5,1e6];
% load('VOC_Parameter.mat')
% load('DD_Parameter.mat')
% load('F194_Parameter.mat')
% load('CLEF_Parameter.mat')
% load('AWAphog_Parameter.mat')
load('SUN_Parameter.mat')
numFolds =10;
for a=1:length(alpha)
   for k=1:length(beita)
     for kk = 1:length(mu)
          h=1;
         for j=begin1:step1:end1
             accuracyMean_selected(h,1)= j;
             accuracyStd_selected(h,1)=j;
             ii=2;
           for i=1:length(noLeafNode)
               x=X{noLeafNode(i)};
%                [H,d1]=size(X{noLeafNode(i)});
%               if H~=0%此条件针对CLEF中的空节点79和80
               feature_slct = Select_feature_Parameter{1,1}{k,kk}(h,noLeafNode(i));
               x = x(:,feature_slct{1,1});
               y = Y{noLeafNode(i)};
               indices = crossvalind('Kfold',length(y),numFolds);
               [accuracyMean_selected(h,ii),accuracyStd_selected(h,ii)] = Kflod_multclass_svm_testParameters([x,y],numFolds,1,indices,tree);
               fprintf(['A0.1--Accurate rate:',num2str(accuracyMean_selected(h,ii)),'±', num2str(accuracyStd_selected(h,ii)),'\n']);%12.5954，12.8267，12.2982，15.0538，11.5385
               ii=ii+1;
%               end
           end
          h=h+1;
         end
        accuracyMean=mean(accuracyMean_selected(:,2:end),2);%按列计算所有节点的均值（特征数由多到少）
        accuracyStd=mean(accuracyStd_selected(:,2:end),2);

%          A=sum(accuracyMean_selected(:,(2:6)),2);%43-48行只适用于F194数据集
%          B=sum(accuracyMean_selected(:,(8:end)),2);
%          C=sum(accuracyStd_selected(:,(2:6)),2);
%          D=sum(accuracyStd_selected(:,(8:end)),2);
%          accuracyMean=(A+B)/7;
%          accuracyStd=(C+D)/7;

%          A=sum(accuracyMean_selected(:,(2:12)),2);%49-58行只适用于CLEF数据集
%          B=sum(accuracyMean_selected(:,(14:19)),2);
%          E=sum(accuracyMean_selected(:,(21)),2);
%          F=sum(accuracyMean_selected(:,(23:end)),2);
%          C=sum(accuracyStd_selected(:,(2:12)),2);
%          D=sum(accuracyStd_selected(:,(14:19)),2);
%          G=sum(accuracyStd_selected(:,(21)),2);
%          H=sum(accuracyStd_selected(:,(23:end)),2);
%          accuracyMean=(A+B+E+F)/20;
%          accuracyStd=(C+D+G+H)/20;
     end
       beita_accuracy{1,k} = accuracyMean; beita_accuracyStd{1,k}= accuracyStd;
   end
   Final_accuracyMean{a,:}=beita_accuracy;Final_accuracyStd{a,:}=beita_accuracyStd; 
end









