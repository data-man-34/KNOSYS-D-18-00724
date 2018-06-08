%% Ô¤²âµÄ¹ý³Ì
%% ÕÔºì
%% 2017-4-8
function preds = hierarchical_predictionzh(decVal_label,decVal_dv, tree)
root = find(tree(:,1)==0);
    for i=1:length(decVal_dv{root}) %The number of samples
        [~,currentNodeID] = max(decVal_dv{root}(i,:));
        currentNode=decVal_label{root}(currentNodeID);
        while(~ismember(currentNode,tree_LeafNode(tree)))
            [~,currentNodeID] = max(decVal_dv{currentNode}(i,:));
            currentNode=decVal_label{currentNode}(currentNodeID);
        end
        preds(i)=currentNode;
    end
end




