clear 
clc
close all

opts.width         = 8;
opts.height        = 6;
opts.fontType      = 'Times';
opts.fontWeight    = 'normal';
opts.fontSize      = 14;

Zones_Names_col    = 1;
Zones_MNF_col      = 43;
Indus_rate_col     = 35;
Commer_rate_col    = 32;
Nonres_rate_col    = 12;
threshold          = 0.1;
data_points        = 20; 

data_length = [30,31,30,31,31,28,31,30,31,30,31,31,24];
data_cumm = cumsum(data_length);
Months = {'Sep ','Oct ','Nov ','Dec ','Jan ','Feb ','Mar ','Apr ','May ','June','July','Aug ','Sept'};
Grouping = cell(sum(data_length),1);

Corr_MNF_Data = readtable('/scratch1/ver100/Water_Project/data/Regression_Model_Reliable_Zones_data.xlsx');


Zones_names     = table2array(Corr_MNF_Data (:,Zones_Names_col));
Zones_MNF       = table2array(Corr_MNF_Data (:,Zones_MNF_col));
Industry_ratio  = table2array(Corr_MNF_Data (:,Indus_rate_col));
Commer_ratio    = table2array(Corr_MNF_Data (:,Commer_rate_col));
Nonres_ratio    = table2array(Corr_MNF_Data (:,Nonres_rate_col));

[SrNonres_ratio,sort_index]  = sort(Nonres_ratio);

Sorted_Comm_ratio    = Commer_ratio(sort_index);
Sorted_Indust_ratio  = Industry_ratio(sort_index);
Sorted_MNF           = Zones_MNF(sort_index);
y                    = Sorted_MNF;
data_index           = find(SrNonres_ratio < threshold);

% plot(y(data_index),'LineWidth',1.7)
plot(y(1:data_points),'LineWidth',2)
hold on

%%%%%%%%%%% Regression for the Industrial Usage
x_industry    = [ones(length(Industry_ratio),1) Sorted_Indust_ratio];
b_industry    = x_industry\y;
yhat_indus    = x_industry*b_industry;
Rsq_Indus     = 1 - sum((y - yhat_indus).^2)/sum((y - mean(y)).^2);
% plot(yhat_indus(data_index),'LineWidth',1.7)
plot(yhat_indus(1:data_points),'LineWidth',2)




%%%%%%%%%%% Regression for the Commercial Usage 
x_commer      = [ones(length(Commer_ratio),1) Sorted_Comm_ratio];
b_commer      = x_commer\y;
yhat_commer   = x_commer*b_commer;
Rsq_commer    = 1 - sum((y - yhat_commer).^2)/sum((y - mean(y)).^2);
% plot(yhat_commer(data_index),'LineWidth',1.7)
plot(yhat_commer(1:data_points),'LineWidth',2)



%%%%%%%%%%% Regression for the Commercial and Industrial Usage 
x_comm_indu    = [ones(length(Commer_ratio),1) Sorted_Indust_ratio Sorted_Comm_ratio];
b_comm_indu    = x_comm_indu\y;
yhat_comm_indu = x_comm_indu*b_comm_indu;
Rsq_comm_indu     = 1 - sum((y - yhat_comm_indu).^2)/sum((y - mean(y)).^2);
% plot(yhat_comm_indu(data_index),'LineWidth',1.7)
plot(yhat_comm_indu(1:data_points),'LineWidth',2)


%%%%%%%%%%% Regression for the Commercial and Industrial Usage w/o intercept
x_comm_indu_no_bo    = [Sorted_Indust_ratio Sorted_Comm_ratio];
b_comm_indu_no_bo    = x_comm_indu_no_bo\y;
yhat_comm_indu_no_bo = x_comm_indu_no_bo*b_comm_indu_no_bo;
Rsq_comm_indu_no_bo     = 1 - sum((y - yhat_comm_indu_no_bo).^2)/sum((y - mean(y)).^2);
% plot(yhat_comm_indu_no_bo(data_index),'LineWidth',1.7)
% plot(yhat_comm_indu_no_bo(1:data_points),'LineWidth',2)

grid on 
xlabel('NonResindetial Ratio','FontSize',14)
ylabel('MNF','FontSize',14)
legend('Ground Truth','Industrial Ratio','Commercial Ratio','Indust and Commer Ratio')


