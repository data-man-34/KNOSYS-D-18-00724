%% Function
function [feature_slct,obj] = HFSSSL(X, Y,tree,s,alpha, beita, mu, noLeafNode, flag)
% rand('seed',6);%当seed取定时，保证了每次随机产生的随机数是相同的 
eps = 1e-8; % set your own tolerance
maxIte = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%为了输出目标函数的每一项
 One=zeros(1,maxIte);
 Two=zeros(1,maxIte);
 Three=zeros(1,maxIte);
  Four=zeros(1,maxIte);
obj=zeros(1,maxIte);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(noLeafNode)
    ClassLabel = unique(Y{noLeafNode(i)});
    m(noLeafNode(i)) = length(ClassLabel);
end
maxm=max(m);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
indexRoot = tree_Root(tree);% The root of the tree
[~,d] = size(X{indexRoot}); % get the number of features
%% initialize
for j = 1:length(noLeafNode)
    Y{noLeafNode(j)}=conversionY01_extend(Y{noLeafNode(j)},maxm);%extend 2 to [1 0]
    W{noLeafNode(j)} = rand(d, s); % initialize W
    H{noLeafNode(j)} = rand(s, maxm); % initialize H
    %%
   % X{noLeafNode(j)} = X{noLeafNode(j)}*diag(sqrt(1./(diag(X{noLeafNode(j)}'*X{noLeafNode(j)})+eps))); % Normalize X
    XX{noLeafNode(j)} = X{noLeafNode(j)}' * X{noLeafNode(j)};
    XY{noLeafNode(j)} = X{noLeafNode(j)}' * Y{noLeafNode(j)};
    HH{noLeafNode(j)} = H{noLeafNode(j)} * H{noLeafNode(j)}';
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:maxIte
    for j = 1:length(noLeafNode)
        %% initialization
         D{noLeafNode(j)} = diag(0.5./max(sqrt(sum(W{noLeafNode(j)}.*W{noLeafNode(j)},2)),eps));%这里eps是最小的正数，它的设置避免了无穷值的出现
        %% Normalize W_i by 2-norm
         G1 = diag(sqrt(1./(diag(W{noLeafNode(j)}'*W{noLeafNode(j)})+eps)));
         G2 = diag(sqrt(diag(W{noLeafNode(j)}'*W{noLeafNode(j)}))+eps);
         W{noLeafNode(j)} = W{noLeafNode(j)}*G1; 
         H{noLeafNode(j)} = G2*H{noLeafNode(j)};
         %% Compute W_Ri
         W_R{noLeafNode(j)}=W_Relation(tree,noLeafNode(j),W,noLeafNode,d ,s);
        %% Update the W_i and H_i
%          H{noLeafNode(j)}=inv(W{noLeafNode(j)}'*XX{noLeafNode(j)}*W{noLeafNode(j)}+eps)*W{noLeafNode(j)}'*XY{noLeafNode(j)};
         E=W{noLeafNode(j)}'* XY{noLeafNode(j)};
         F=W{noLeafNode(j)}'*XX{noLeafNode(j)}*W{noLeafNode(j)}* H{noLeafNode(j)}+eps;
         H{noLeafNode(j)}= H{noLeafNode(j)}.*E./F;
         A=XY{noLeafNode(j)}*H{noLeafNode(j)}'+beita*W_R{noLeafNode(j)}+mu*W{noLeafNode(j)};
         B=XX{noLeafNode(j)}*W{noLeafNode(j)}*HH{noLeafNode(j)}+alpha* D{noLeafNode(j)}*W{noLeafNode(j)}+beita*W{noLeafNode(j)}+mu*W{noLeafNode(j)}*W{noLeafNode(j)}'*W{noLeafNode(j)}+eps;
         W{noLeafNode(j)}=W{noLeafNode(j)}.*A./B;
    end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The value of object function
    if (flag ==1)
       for j = 1:length(noLeafNode)
            One(i)=One(i)+(norm(X{noLeafNode(j)}*W{noLeafNode(j)}*H{noLeafNode(j)}-Y{noLeafNode(j)}))^2;
            Two(i)=Two(i)+alpha*L21(W{noLeafNode(j)});
            Three(i)=Three(i)+beita *(norm(W{noLeafNode(j)}-W_R{noLeafNode(j)}))^2;
            Four(i)=Four(i)+mu/2*(norm(W{noLeafNode(j)}'*W{noLeafNode(j)}-ones(s)))^2;  
       end
       obj(i)=One(i)+Two(i)+Three(i)+Four(i);
    end;  
    save Valueobj One Two Three Four obj
end

for j = 1: length(noLeafNode)
    tempVector = sum(W{noLeafNode(j)}.^2, 2);
    [atemp, value] = sort(tempVector, 'descend'); % sort tempVecror (W) in a descend order
    clear tempVector;
    feature_slct{noLeafNode(j)} = value(1:s);
end
% if (flag == 1)
%     figure;
%     set(gcf,'color','w');
%     plot(obj,'LineWidth',4,'Color',[0 0 1]);
%     set(gca,'FontName','Times New Roman','FontSize',11);
%     xlabel('Iteration number');
%     ylabel('Objective function value');
% end
end