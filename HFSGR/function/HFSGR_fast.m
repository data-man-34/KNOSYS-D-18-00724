%% HiRRpar_FS
% Feature selection based on the relationship between parent and children
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
%% Date:
%     2016-11-30
%% Author:
%     Hong Zhao, modify by qianjuan Tuo 2016-12-29
%% Function
function [feature_slct] = HFSGR_fast(X, Y, numSelected, tree, alpha, beita, noLeafNode, F, flag)
%  rand('seed',9);% When seed is timed, it ensures that the random number generated at random is the same
eps = 1e-8; % set your own tolerance
maxIte = 10;
for i = 1:length(noLeafNode)
    ClassLabel = unique(Y{noLeafNode(i)});
    m(noLeafNode(i)) = length(ClassLabel);
end
maxm=max(m);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k=1;
indexRoot = tree_Root(tree);% The root of the tree
 %indexRoot=max(indexRoot);%only for Landscape
 internalNodes = tree_InternalNodes(tree);
  noLeafNode =[indexRoot;internalNodes];
[~,d] = size(X{indexRoot}); % get the number of features
%% initialize
for j = 1:length(noLeafNode)
    Y{noLeafNode(j)}=conversionY01_extend(Y{noLeafNode(j)},maxm);%extend 2 to [1 0]
    W{noLeafNode(j)} = rand(d, maxm); % initialize W
    %%
    XX{noLeafNode(j)} = X{noLeafNode(j)}' * X{noLeafNode(j)};
    XY{noLeafNode(j)} = X{noLeafNode(j)}' * Y{noLeafNode(j)};
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M=sum(F,2);%W是F的列和
for i = 1:maxIte
    for j = 1:length(noLeafNode)
        %% initialization
        D{noLeafNode(j)} = diag(0.5./max(sqrt(sum(W{noLeafNode(j)}.*W{noLeafNode(j)},2)),eps));%Here eps is the smallest positive number, and its setting avoids the appearance of infinite values
        %% Update the noLeafNode
        W_F_nonj= zeros(d,max(m));%this d is the number of features
        for k=1:length(noLeafNode) 
             W_F_nonj=W_F_nonj + F(j,k)*W{noLeafNode(k)};%when j=k，F(j,j)=0
        end
        W{noLeafNode(j)}=inv(XX{noLeafNode(j)}+alpha * D{noLeafNode(j)}+beita* M(j)* eye(d))*( XY{noLeafNode(j)}+beita*W_F_nonj);
    end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The value of object function
    if (flag ==1)
       for j = 1:length(noLeafNode)
            W_F_value=0;
            for k=1:length(noLeafNode)
               W_F_value= W_F_value+F(j,k)*(norm(W{noLeafNode(j)}-W{noLeafNode(k)}))^2;
            end
            obj(i)=(norm(X{noLeafNode(j)}*W{noLeafNode(j)}-Y{noLeafNode(j)}))^2+alpha*L21(W{noLeafNode(j)})+ beita * W_F_value ;  
       end
    end
end

%obj
for i = 1: length(noLeafNode)
    W1=W{noLeafNode(i)};
    W{noLeafNode(i)} = W1(:,1:m(noLeafNode(i)));
end

clear W1;
for j = 1: length(noLeafNode)
    tempVector = sum(W{noLeafNode(j)}.^2, 2);
    [atemp, value] = sort(tempVector, 'descend'); % sort tempVecror (W) in a descend order
    clear tempVector;
    feature_slct{noLeafNode(j)} = value(1:numSelected);
end
if (flag == 1)
    figure;
    set(gcf,'color','w');
    plot(obj,'LineWidth',4,'Color',[0 0 1]);
    set(gca,'FontName','Times New Roman','FontSize',11);
    xlabel('Iteration number');
    ylabel('Objective function value');
end
end

