clear 
clc
close all

opts.width      = 8;
opts.height     = 6;
opts.fontType   = 'Times';
opts.fontWeight   = 'normal';
opts.fontSize   = 14;


Selected_Zones_data =readtable('/scratch1/ver100/Water_Project/data/Selected_Zones_data.xlsx');
data_length = [30,31,30,31,31,28,31,30,31,30,31,31,24];
data_cumm = cumsum(data_length);

Months = {'Sep ','Oct ','Nov ','Dec ','Jan ','Feb ','Mar ','Apr ','May ','June','July','Aug ','Sept'};
Grouping = cell(sum(data_length),1);

Zones_name = Selected_Zones_data.Properties.VariableNames;
Zones_name = Zones_name(2:end)';

Zone1_data = table2array(Selected_Zones_data(3:end,Zones_name(1)));
Zone2_data = table2array(Selected_Zones_data(3:end,Zones_name(2)));
Zone3_data = table2array(Selected_Zones_data(3:end,Zones_name(3)));

%%%%% Replace zeros with nan %%%%%%

Zone1_data(Zone1_data == 0) = nan;
Zone2_data(Zone2_data == 0) = nan;
Zone3_data(Zone3_data == 0) = nan;


Zone1_NC  = table2array(Selected_Zones_data(1,Zones_name(1))); 
Zone1_LM  = table2array(Selected_Zones_data(2,Zones_name(1)));

Zone2_NC  = table2array(Selected_Zones_data(1,Zones_name(2))); 
Zone2_LM  = table2array(Selected_Zones_data(2,Zones_name(2)));

Zone3_NC  = table2array(Selected_Zones_data(1,Zones_name(3))); 
Zone3_LM  = table2array(Selected_Zones_data(2,Zones_name(3)));

%%%% Normalize Data w.r.t Litre/NC/hour-4 hours %%%%

NZone1_data_NC = (Zone1_data * 1000000)/(Zone1_NC * 4);
NZone2_data_NC = (Zone2_data * 1000000)/(Zone2_NC * 4);
NZone3_data_NC = (Zone3_data * 1000000)/(Zone3_NC * 4);

%%%% Normalize Data w.r.t Litre/LM/hour-4 hours %%%%

NZone1_data_LM = (Zone1_data * 1000000)/(Zone1_LM * 4);
NZone2_data_LM = (Zone2_data * 1000000)/(Zone2_LM * 4);
NZone3_data_LM = (Zone3_data * 1000000)/(Zone3_LM * 4);

start_index = 1;

for i=1: numel(data_length)
    Grouping(start_index:data_cumm(i)) = Months(i);
    start_index = data_cumm(i) + 1;   
end



%%%%% Plotting Zone 1 %%%


subplot(1,2,1)
m_nc = nanmean(NZone1_data_NC);
med_nc = nanmedian(NZone1_data_NC);
wm_nc = nanmean(NZone1_data_NC(data_cumm(9)+1: data_cumm(12)));
wmed_nc =  nanmedian(NZone1_data_NC(data_cumm(9)+1: data_cumm(12)));
boxplot(NZone1_data_NC, char(Grouping))
grid on
set(gca,'GridAlpha',0.15,'GridLineStyle','-.')
title(sprintf('Normalization - Number Of Connections: %d',Zone1_NC))
% l= sprintf('Mean %.4f, \nMedian %.4f \nWinter Mean %.4f, \nWinter Median %.4f',m_nc,med_nc,wm_nc,wmed_nc);
l= sprintf('Mean- %.4f \nWinter Mean- %.4f \nMedian- %.4f   \nWinter Median- %.4f ',...
    m_nc,wm_nc,med_nc,wmed_nc);
text(0.2,0.8,l,'Units','Normalized','FontSize',opts.fontSize)
ylabel(Zones_name{1},'FontWeight','bold','Interpreter','none')
xtickangle(45)

subplot(1,2,2)
m_lm = nanmean(NZone1_data_LM);
med_lm = nanmedian(NZone1_data_LM);
wm_lm = nanmean(NZone1_data_LM(data_cumm(9)+1: data_cumm(12)));
wmed_lm =  nanmedian(NZone1_data_LM(data_cumm(9)+1: data_cumm(12)));
boxplot(NZone1_data_LM, char(Grouping))
grid on
set(gca,'GridAlpha',0.15,'GridLineStyle','-.')
title(sprintf('Normalization - Length Of Mains: %d',Zone1_LM))
% l= sprintf('Mean %.4f, \nMedian %.4f \nWinter Mean %.4f, \nWinter Median %.4f',m_lm,med_lm,wm_lm,wmed_lm);
l= sprintf('Mean- %.4f \nWinter Mean- %.4f \nMedian- %.4f   \nWinter Median- %.4f',...
    m_lm,wm_lm,med_lm,wmed_lm);
text(0.2,0.8,l,'Units','Normalized','FontSize',opts.fontSize)
xtickangle(45)

% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.7]);






%%%%% Plotting Zone 2  %%%%%

% subplot(1,2,1)
% m_nc = nanmean(NZone2_data_NC);
% med_nc = nanmedian(NZone2_data_NC);
% wm_nc = nanmean(NZone2_data_NC(data_cumm(9)+1: data_cumm(12)));
% wmed_nc =  nanmedian(NZone2_data_NC(data_cumm(9)+1: data_cumm(12)));
% boxplot(NZone2_data_NC, char(Grouping))
% grid on
% set(gca,'GridAlpha',0.15,'GridLineStyle','-.')
% ylim([150 500])
% title(sprintf('Normalization - Number Of Connections: %d',Zone2_NC))
% % l= sprintf('Mean %.4f, \nMedian %.4f \nWinter Mean %.4f, \nWinter Median %.4f',m_nc,med_nc,wm_nc,wmed_nc);
% l= sprintf('Mean- %.4f \nWinter Mean- %.4f \nMedian- %.4f   \nWinter Median- %.4f ',...
%     m_nc,wm_nc,med_nc,wmed_nc);
% text(0.2,0.8,l,'Units','Normalized','FontSize',opts.fontSize)
% ylabel(Zones_name{2},'FontWeight','bold','Interpreter','none')
% xtickangle(45)
% 
% subplot(1,2,2)
% m_lm = nanmean(NZone2_data_LM);
% med_lm = nanmedian(NZone2_data_LM);
% wm_lm = nanmean(NZone2_data_LM(data_cumm(9)+1: data_cumm(12)));
% wmed_lm =  nanmedian(NZone2_data_LM(data_cumm(9)+1: data_cumm(12)));
% boxplot(NZone2_data_LM, char(Grouping))
% grid on
% set(gca,'GridAlpha',0.15,'GridLineStyle','-.')
% ylim([10 50])
% title(sprintf('Normalization - Length Of Mains: %d',Zone2_LM))
% % l= sprintf('Mean %.4f, \nMedian %.4f \nWinter Mean %.4f, \nWinter Median %.4f',m_lm,med_lm,wm_lm,wmed_lm);
% l= sprintf('Mean- %.4f \nWinter Mean- %.4f \nMedian- %.4f   \nWinter Median- %.4f',...
%     m_lm,wm_lm,med_lm,wmed_lm);
% text(0.2,0.8,l,'Units','Normalized','FontSize',opts.fontSize)
% xtickangle(45)
% 
% % Enlarge figure to full screen.
% set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.7]);
% 






% %%%%% Plotting Zone 3  %%%%%
% % 
% subplot(1,2,1)
% m_nc = nanmean(NZone3_data_NC);
% med_nc = nanmedian(NZone3_data_NC);
% wm_nc = nanmean(NZone3_data_NC(data_cumm(9)+1: data_cumm(12)));
% wmed_nc =  nanmedian(NZone3_data_NC(data_cumm(9)+1: data_cumm(12)));
% boxplot(NZone3_data_NC, char(Grouping))
% grid on
% set(gca,'GridAlpha',0.15,'GridLineStyle','-.')
% ylim([150 500])
% title(sprintf('Normalization - Number Of Connections: %d ',Zone3_NC))
% % l= sprintf('Mean %.4f, \nMedian %.4f \nWinter Mean %.4f, \nWinter Median %.4f',m_nc,med_nc,wm_nc,wmed_nc);
% l= sprintf('Mean- %.4f \nWinter Mean- %.4f \nMedian- %.4f   \nWinter Median- %.4f ',...
%     m_nc,wm_nc,med_nc,wmed_nc);
% text(0.2,0.8,l,'Units','Normalized','FontSize',opts.fontSize)
% ylabel(Zones_name{3},'FontWeight','bold','Interpreter','none')
% xtickangle(45)
% 
% subplot(1,2,2)
% m_lm = nanmean(NZone3_data_LM);
% med_lm = nanmedian(NZone3_data_LM);
% wm_lm = nanmean(NZone3_data_LM(data_cumm(9)+1: data_cumm(12)));
% wmed_lm =  nanmedian(NZone3_data_LM(data_cumm(9)+1: data_cumm(12)));
% boxplot(NZone3_data_LM, char(Grouping))
% grid on
% set(gca,'GridAlpha',0.15,'GridLineStyle','-.')
% ylim([8 30])
% title(sprintf('Normalization - Length Of Mains: %d',Zone3_LM))
% % l= sprintf('Mean %.4f, \nMedian %.4f \nWinter Mean %.4f, \nWinter Median %.4f',m_lm,med_lm,wm_lm,wmed_lm);
% l= sprintf('Mean- %.4f \nWinter Mean- %.4f \nMedian- %.4f   \nWinter Median- %.4f',...
%     m_lm,wm_lm,med_lm,wmed_lm);
% text(0.2,0.8,l,'Units','Normalized','FontSize',opts.fontSize)
% xtickangle(45)
% 
% % Enlarge figure to full screen.
% set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.7]);





%%%%%%% Plot the data using line plot now %%%%%%%%
     %%%%%%% Replace nan with zeros %%%%%%%

%%%%% Plotting Zone 1  %%%%%

% NZone1_data_NC(isnan(NZone1_data_NC)) = 0;
% NZone1_data_LM(isnan(NZone1_data_LM)) = 0;

% plot(NZone1_data_LM,'LineWidth',1.5) 
% hold on
% plot(NZone2_data_LM,'LineWidth',1.5)
% plot(NZone3_data_LM,'LineWidth',1.5)
% %%%% Plot the indices in zero with a different colot %%%%%
% missing_indices = find(isnan(NZone1_data_LM));
% F = fillmissing(NZone1_data_LM,'movmedian',10);
% plot(missing_indices,F(missing_indices),'Color','k','LineWidth',2)
% grid on
% set(gca,'GridAlpha',0.15,'GridLineStyle','-.')
% title('Normalization - Length of Mains')
% % l= sprintf('%s \n%s \n %s \nMissing Values total: %d',Zones_name{1}, Zones_name{2},Zones_name{3},numel(missing_indices));
% legend({Zones_name{1}, Zones_name{2}, Zones_name{3},'Missing Values total: 8'},'Interpreter','none')
% ylabel('MNF')
% xticks(data_cumm)
% xticklabels(Months)
% xtickangle(45)



