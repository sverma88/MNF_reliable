"""
This script trains regression on MNF data
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from utils import *



'''                                                                                                                     
path settings for the data retrival and results accumulation                                                            
Remeber all indexing starts with 0 and not 1, i.e. including headers and columns date                                            
'''

data_dir    = '/scratch1/ver100/Water_Project/data/'
plots_dir   = '/scratch1/ver100/MNF_reliable/plots/'
Drop_MARS   = True

Zones_Names_col      = 1
Zones_MNF_col        = 42
Industry_rate_col    = 34
NonRes_rate_col      = 11
Commer_rate_col      = 31
LengthMains_col      = 44
NumConnect_col       = 43
Commercial_col       = 16
Industrial_col       = 19
DailyCons_Comm       = -2
DailyCons_Inds       = -1


MNF_data             = pd.read_excel(data_dir +  'Regression_Model_Reliable_Zones_data_Verified.xlsx')

print("----------------Regression Data --------------")
MNF_data.fillna(0, inplace = True)  # rip of the Nan values
print(MNF_data.head(10))  # Can put any integer to print the values

Marsfield_index = MNF_data[MNF_data['pressure_zone'] == 'P_MARSFIELD'].index
#### Drop Marsfield as it's an outlier
if Drop_MARS:
    MNF_data.drop(Marsfield_index, inplace = True)

#### Get all data first
Ground_Truth          = np.array(MNF_data.iloc[:, Zones_MNF_col])
NonResr_Ratio         = np.array(MNF_data.iloc[:, NonRes_rate_col])
Data_Ranking          = np.argsort(NonResr_Ratio)
Ranking_lower         = Data_Ranking[:67]
Ranking_higher        = Data_Ranking[-67:]
Ranking_middle        = Data_Ranking[67:-67]


#### This are the variables
LenMains_data    = np.array(MNF_data.iloc[:, LengthMains_col])
NumConnec_data   = np.array(MNF_data.iloc[:, NumConnect_col])
IndustryRatio    = np.array(MNF_data.iloc[:, Industry_rate_col])
CommerctRatio    = np.array(MNF_data.iloc[:, Commer_rate_col])
CommercialVal    = np.array(MNF_data.iloc[:, Commercial_col])
IndustrialVal    = np.array(MNF_data.iloc[:, Industrial_col])
Commer_Daily     = np.array(MNF_data.iloc[:, DailyCons_Comm])
Industy_Daily    = np.array(MNF_data.iloc[:, DailyCons_Inds])

NonZero_IndustryRatio  = np.nonzero(IndustryRatio > 0)
Zero_IndustryRatio     = np.nonzero(IndustryRatio == 0)

#### Normalize Commercial Daily and Industrial Daily
# Commer_Daily  = Commer_Daily/ (max(Commer_Daily) -  min(Commer_Daily))
# Industy_Daily = Industy_Daily/ (max(Industy_Daily) - min(Industy_Daily))

###### Plot the MNF and NonResidental Ratio

fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(20, 10))
fig.suptitle('Ground Truth Plots')
ax1.hist(Ground_Truth[Ranking_lower])
ax1.set(xlabel='Ground Truth MNF',ylabel='Frequency')
ax1.set(title='Bottom 1/3')
ax1.text(0.7,0.7,'Mean: {} '.format(round(np.mean(Ground_Truth[Ranking_lower]),2)), horizontalalignment='center',
         verticalalignment='center', transform=ax1.transAxes)


ax2.hist(Ground_Truth[Ranking_middle])
ax2.set(xlabel='Ground Truth MNF',ylabel='Frequency')
ax2.set(title='Middle 1/3')
ax2.text(0.7,0.7,'Mean: {} '.format(round(np.mean(Ground_Truth[Ranking_middle]),2)), horizontalalignment='center',
         verticalalignment='center', transform=ax2.transAxes)

ax3.hist(Ground_Truth[Ranking_higher])
ax3.set(xlabel='Ground Truth MNF',ylabel='Frequency')
ax3.set(title='Top 1/3')
ax3.text(0.7,0.7,'Mean: {} '.format(round(np.mean(Ground_Truth[Ranking_higher]),2)), horizontalalignment='center',
         verticalalignment='center', transform=ax3.transAxes)


ax4.hist(NonResr_Ratio[Ranking_lower])
ax4.set(xlabel='NonResidential Rate',ylabel='Frequency')
ax5.hist(NonResr_Ratio[Ranking_middle])
ax5.set(xlabel='NonResidential Rate',ylabel='Frequency')
ax6.hist(NonResr_Ratio[Ranking_higher])
ax5.set(xlabel='NonResidential Rate',ylabel='Frequency')

plt.savefig('/scratch1/ver100/MNF_Regression/Regression_plots/MNF_NRR_plots.png')


#### Plot the Daily Commercial Consumption Rate and Daily Industrial Consumption Rate
fig, ((ax4,ax5,ax6), (ax7, ax8, ax9)) = plt.subplots(2,3, figsize=(20, 10))
fig.suptitle('Daily Consumption Rates')

ax4.set(title='Bottom 1/3')
ax5.set(title='Middle 1/3')
ax6.set(title='Top 1/3')

ax4.hist(Commer_Daily[Ranking_lower])
ax4.text(0.71,0.78,'Commercial',horizontalalignment='center',
         verticalalignment='center', transform=ax4.transAxes)
ax4.set(xlabel='Consumption Rate', ylabel='Consumption Frequency')
ax5.hist(Commer_Daily[Ranking_middle])
ax5.set(xlabel='Consumption Rate', ylabel='Consumption Frequency')
ax5.text(0.71,0.78,'Commercial',horizontalalignment='center',
         verticalalignment='center', transform=ax5.transAxes)
ax6.hist(Commer_Daily[Ranking_higher])
ax6.set(xlabel='Consumption Rate', ylabel='Consumption Frequency')
ax6.text(0.71,0.78,'Commercial',horizontalalignment='center',
         verticalalignment='center', transform=ax6.transAxes)


ax7.hist(Industy_Daily[Ranking_lower])
ax7.set(xlabel='Consumption Rate', ylabel='Consumption Frequency')
ax7.text(0.71,0.78,'Industrial',horizontalalignment='center',
         verticalalignment='center', transform=ax7.transAxes)
ax8.hist(Industy_Daily[Ranking_middle])
ax8.set(xlabel='Consumption Rate', ylabel='Consumption Frequency')
ax8.text(0.71,0.78,'Industrial',horizontalalignment='center',
         verticalalignment='center', transform=ax8.transAxes)
ax9.hist(Industy_Daily[Ranking_higher])
ax9.set(xlabel='Consumption Rate', ylabel='Consumption Frequency')
ax9.text(0.71,0.78,'Industrial',horizontalalignment='center',
         verticalalignment='center', transform=ax9.transAxes)


# print(MNF_data.iloc[:,-1])
plt.savefig('/scratch1/ver100/MNF_Regression/Regression_plots/Daily_Consumption.png')

plt.show()

# print(Industy_Daily[Ranking_lower])
