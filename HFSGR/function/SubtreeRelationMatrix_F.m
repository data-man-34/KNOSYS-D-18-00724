function [noLeafNode,F]=SubtreeRelationMatrix_F(tree)
indexRoot = tree_Root(tree);% The root of the tree
leafNode = tree_LeafNode(tree);
%根据层数从小到大对中间结点进行排序
indexRoot=max(indexRoot);
noLeafNode=[indexRoot];
for k=1:max(tree(:,2))-1
    index = setdiff(find(tree(:,2)==k),leafNode); %delete the leaf node.
    noLeafNode=[noLeafNode;index];
end
F=zeros(length(noLeafNode),length(noLeafNode));
for i=1:length(noLeafNode)
    for j=1:length(noLeafNode)
        %如果i是j的父结点，或者j是i的父结点
       if  noLeafNode(i)==tree(noLeafNode(j),1)||noLeafNode(j)==tree(noLeafNode(i),1)
           F(i,j)=1;
       else
           F(i,j)=0;
       end
    end
end

