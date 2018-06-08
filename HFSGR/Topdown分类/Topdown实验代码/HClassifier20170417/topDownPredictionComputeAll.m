%% Top-down prediction
% Written by Yu Wang 
% Modified by Hong Zhao
% 2017-4-11
%% Inputs:
% input_data: training data without labels
% model: 
% tree: the tree hierarchy
%% Output

function [predict_label] = topDownPredictionComputeAll(input_data,inputLabel, model, tree)     
    [decVal_label,decVal_dv] = get_all_decValszh(input_data, inputLabel, model, tree);
    predict_label = hierarchical_predictionzh(decVal_label,decVal_dv, tree);
end
