function [info_gain_total, info_gain_mean] = information_gain(label_set, pred_set, tree)
info_gain_total = 0;
[r,c] = size(pred_set);
info_gain_set = zeros(r,c);
leafSet = tree_LeafNode(tree);
num_leafNodes = length(leafSet);
LCA_node_set = zeros(r,c);
for cc = 1:c
    temp_pred_set = tree_Ancestor(tree,pred_set(:,cc),1);%最后一个参数1是代表包括自己，如果是0则不包括自己 
    temp_label_set = tree_Ancestor(tree,label_set(:,cc),1);
    common_ans_set = intersect(temp_pred_set, temp_label_set);
    node_cal = common_ans_set(1);
    if (ismember(node_cal, leafSet))
        cur_leaf_set = node_cal;
    else
        children_set = tree_Descendant(tree, node_cal, 0);
        cur_leaf_set = intersect(children_set, leafSet);
    end
   info_gain_set(:,cc) = log2(num_leafNodes) - log2(length(cur_leaf_set));
   info_gain_total = info_gain_total + info_gain_set(:,cc);
end
info_gain_idealTotal = c * (log2(num_leafNodes) - log2(1));
info_gain_mean = info_gain_total/info_gain_idealTotal;
end