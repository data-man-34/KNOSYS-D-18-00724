%% Top-down prediction
% Written by Yu Wang 
% Modified by Hong Zhao
% 2017-4-11
%% Inputs:
% input_data: training data without labels
% model: 
% tree: the tree hierarchy
%% Output

function [predict_label] = topDownPrediction(input_data, model, tree)
    
    [m,~]=size(input_data);
    root = find(tree(:,1)==0);     
	for j=1:m%The number of samples
%% 先从树根开始
        [~,~,d_v] = predict(1,sparse(input_data(j,:)), model{root}, '-b 1 -q');%第一个参数没有作用。
        [~,currentNodeID] = max(d_v);
        currentNode=model{root}.Label(currentNodeID) ;
    %% 递归调用中间层直到叶子结点
       while(~ismember(currentNode,tree_LeafNode(tree)))
          [~,~,d_v] = predict(1,  sparse(input_data(j,:)),  model{currentNode}, '-b 1 -q');
          [~,currentNodeID] = max(d_v);
          currentNode=model{currentNode}.Label(currentNodeID);
        end
        predict_label(j)=currentNode;        
    end %%endfor    
end
