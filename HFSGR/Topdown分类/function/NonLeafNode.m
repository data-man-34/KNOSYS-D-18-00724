function [noLeafNode]=NonLeafNode(tree)
indexRoot = tree_Root(tree);% The root of the tree
leafNode = tree_LeafNode(tree);
%���ݲ�����С������м����������
noLeafNode=[indexRoot];
for k=1:max(tree(:,2))-1
    index = setdiff(find(tree(:,2)==k),leafNode); %delete the leaf node.
    noLeafNode=[noLeafNode;index];
end

