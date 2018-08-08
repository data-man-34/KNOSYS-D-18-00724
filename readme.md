title={Hierarchical feature selection with graph regularization based on subtree},
author={Tuo, Qianjuan and Zhao, Hong and Hu, Qinghua},

Here, we mainly introduce the three programs contained in the folder function.

(1).SubtreeRelationMatrix_F
%% Input:
%       tree - tree structure among the classes  
%% Output:
% noLeafNode - the all non-leaf nodes, which according to the number of layers, the middle nodes are sorted from small to large.  
%          F - the relation matrix among the subtrees

 
(2).HFSGR_fast
%% Input:
%     X - the data matrix without the label
%     Y - labels
%     k - the number of selected features
%     Child_num - the number of the level 2 ( that is the number of children
%     of the root node    %%Child_num在这里表示除根以外的非叶子结点数目
%     alpha - the parameter of optimal
%     beita - the tradeoff parameter 
%     flag - draw the objective value
%% Output:
%     feature_slct - The selected feature subset 


(3).HFSGR_main (it is the main code, we run it)
%% Input:
%  data_array - the data matrix with the label
%      tree - tree structure among the classes
%     alpha - the parameter of optimal
%     beita - the tradeoff parameter  
%% Output:
%    accuracyMean_selected - classification accuracy  

