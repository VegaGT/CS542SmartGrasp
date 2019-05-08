clear all
%% load dataset
load("bigger.mat");
data = bigger;
varNames = data.Properties.VariableNames;

%% extract features and labelsfrom data table
features = zeros(size(data,1),18);
labels = zeros(size(data,1),1);
feat_idxs = [4 5 7 8 10 11 13 14 16 17 19 20 22 23 25 26 28 29];
label_idx = 2;
label_thresh = 95;%median(data.robustness);
for i = 1:size(data,1)
    feat = table2array(data(i,feat_idxs));
    features(i,:) = feat;
    rbst = table2array(data(i,label_idx));
    if(rbst > label_thresh)
        labels(i) = 1;
    else
        labels(i) = 0;
    end
end