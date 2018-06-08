%% Compute W_Ri, it presents the relationship weight matrix of the node i
%W_Ri=(W_Pi+sum(W_Ci))/(Ci_num+1)
function [W_R]=W_Relation(tree,i,W,all_Internal,d,s)
%���i�ĸ��ڵ���0��Ҳ����i�Ǹ��ڵ㣬������ʱ��ֻ���������ӽ�㡣
if tree_Parent(tree,i)==0
   ChildrenNode=Child_internalnode(tree,i,all_Internal);%���ؽ��i��Ӧ�ķ�Ҷ�ӽ����ӽ��
   W_ChildrenSum=zeros(d,s);
   for t=1:length(ChildrenNode)
      W_ChildrenSum=W_ChildrenSum+W{ChildrenNode(t)};    
   end
   W_R=W_ChildrenSum/length(ChildrenNode);
else
 %���򣬼������ĸ������ӽ��
   ParentNode=tree_Parent(tree,i);%���ؽ��i��Ӧ�ĸ����
   W_Parent=W{ParentNode};%��Ҷ�ӽ��i�ĸ�����Ȩ��
   ChildrenNode=Child_internalnode(tree,i,all_Internal);%���ؽ��i��Ӧ�ķ�Ҷ�ӽ����ӽ��
   if isempty(ChildrenNode)%�ж�û���ӽ��ʱ��ֻ�͸�����й�
      W_R=W_Parent;
   else%�ж����ӽ���ʱ���������ӽ�㶼�й�ϵ
      W_ChildrenSum=zeros(d,s);
      for t=1:length(ChildrenNode)
         W_ChildrenSum=W_ChildrenSum+W{ChildrenNode(t)};    
      end
      W_R=(W_Parent+W_ChildrenSum)/(length(ChildrenNode)+1);
   end
end
 
