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
alpha       = 0.1
fit_inter   = False
Drop_MARS   = True
fit_lasso   = False

Zones_Names_col      = 1
Zones_MNF_col        = 42
Industry_rate_col    = 34
NonRes_rate_col      = 11
Commer_rate_col      = 31
LengthMains_col      = 44
NumConnect_col       = 43
Commercial_col       = 16
Industrial_col       = 19


if fit_lasso:
    RegModel = Lasso(alpha = alpha, fit_intercept = fit_inter)
else:
    RegModel = LinearRegression(fit_intercept = fit_inter)

MNF_data             = pd.read_excel(data_dir +  'Regression_Model_Reliable_Zones_data.xlsx')

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
Ranking_lower30       = Data_Ranking[:30]
Ranking_lower50       = Data_Ranking[:50]
Ranking_lower80       = Data_Ranking[:80]
Ranking_higher30      = Data_Ranking[-30:]
Ranking_higher50      = Data_Ranking[-50:]
Ranking_higher80      = Data_Ranking[-80:]


#### This are the variables
LenMains_data    = np.array(MNF_data.iloc[:, LengthMains_col])
NumConnec_data   = np.array(MNF_data.iloc[:, NumConnect_col])
IndustryRatio    = np.array(MNF_data.iloc[:, Industry_rate_col])
CommerctRatio    = np.array(MNF_data.iloc[:, Commer_rate_col])
CommercialVal    = np.array(MNF_data.iloc[:, Commercial_col])
IndustrialVal    = np.array(MNF_data.iloc[:, Industrial_col])

# print(Ranking_lower)
# print(IndustrialVal)

### Print Data Statistics
# print("Data Statistics")
# compute_mean_stats(Ground_Truth,Ranking_lower30)

##### Create Data and normalize them
Data1  = np.stack((CommerctRatio, IndustryRatio), axis=1)
Scaler1 = StandardScaler()
# Scaler1 = MinMaxScaler()
Scaler1.fit(Data1)
Data1 = Scaler1.transform(Data1)

Data2  = np.stack((CommerctRatio, IndustryRatio, LenMains_data), axis=1)
# Scaler2 = StandardScaler()
Scaler2 = MinMaxScaler()
Scaler2.fit(Data2)
Data2 = Scaler2.transform(Data2)

Data3  = np.stack((CommerctRatio, IndustryRatio, NumConnec_data), axis=1)
# Scaler3 = StandardScaler()
Scaler3 = MinMaxScaler()
Scaler3.fit(Data3)
Data3 = Scaler3.transform(Data3)

Data4  = np.stack((CommerctRatio, IndustryRatio, NumConnec_data, LenMains_data), axis=1)
# Scaler4 = StandardScaler()
Scaler4 = MinMaxScaler()
Scaler4.fit(Data4)
Data4 = Scaler4.transform(Data4)

Data5  = np.reshape(NonResr_Ratio,(-1,1))
# Scaler5 = StandardScaler()
Scaler5 = MinMaxScaler()
Scaler5.fit(Data5)
Data5 = Scaler5.transform(Data5)

#### Regression on Data 1
Yhat_1      = RegModel.fit(Data1, Ground_Truth).predict(Data1)
R2_D1       = r2_score(Yhat_1, Ground_Truth)
MAE_D1      = mean_absolute_error(Yhat_1, Ground_Truth)
MAE_D1_bot  = mean_absolute_error(Yhat_1[Ranking_lower30], Ground_Truth[Ranking_lower30])
MAE_D1_top  = mean_absolute_error(Yhat_1[Ranking_higher30], Ground_Truth[Ranking_higher30])
print("-----------Regression Results with Resgressors Commercial Ratio and Industrial Ratio---------\n")
print("Coefficient 1---->",RegModel.coef_)
print("Rsq: {} \t MAE: {}".format(R2_D1, MAE_D1))
E1 = compute_error(Ground_Truth,Yhat_1)

#### Regression on Data 2
Yhat_2      = RegModel.fit(Data2, Ground_Truth).predict(Data2)
R2_D2       = r2_score(Yhat_2, Ground_Truth)
MAE_D2      = mean_absolute_error(Yhat_2, Ground_Truth)
MAE_D2_bot  = mean_absolute_error(Yhat_2[Ranking_lower30], Ground_Truth[Ranking_lower30])
MAE_D2_top  = mean_absolute_error(Yhat_2[Ranking_higher30], Ground_Truth[Ranking_higher30])
print("-----------Regression Results with Resgressors Commercial Ratio, Industrial Ratio, and LengthMains---------\n")
print("Coefficient 2---->",RegModel.coef_)
print("Rsq: {} \t MAE: {}".format(R2_D2, MAE_D2))
E2 = compute_error(Ground_Truth,Yhat_2)


#### Regression on Data 3
Yhat_3      = RegModel.fit(Data3, Ground_Truth).predict(Data3)
R2_D3       = r2_score(Yhat_3, Ground_Truth)
MAE_D3      = mean_absolute_error(Yhat_3, Ground_Truth)
MAE_D3_bot  = mean_absolute_error(Yhat_3[Ranking_lower30], Ground_Truth[Ranking_lower30])
MAE_D3_top  = mean_absolute_error(Yhat_3[Ranking_higher30], Ground_Truth[Ranking_higher30])
print("-----------Regression Results with Resgressors Commercial Ratio, Industrial Ratio, and NumConn---------\n")
print("Coefficient 3---->",RegModel.coef_)
print("Rsq: {} \t MAE: {}".format(R2_D3, MAE_D3))
E3 = compute_error(Ground_Truth,Yhat_3)

#### Regression on Data 4
Yhat_4      = RegModel.fit(Data4, Ground_Truth).predict(Data4)
R2_D4       = r2_score(Yhat_4, Ground_Truth)
MAE_D4      = mean_absolute_error(Yhat_4, Ground_Truth)
MAE_D4_bot  = mean_absolute_error(Yhat_4[Ranking_lower30], Ground_Truth[Ranking_lower30])
MAE_D4_top  = mean_absolute_error(Yhat_4[Ranking_higher30], Ground_Truth[Ranking_higher30])
print("-----------Regression Results with Resgressors Commercial Ratio, Industrial Ratio, LengthMains, and NumConn---------\n")
print("Coefficient 4---->",RegModel.coef_)
print("Rsq: {} \t MAE: {}".format(R2_D4, MAE_D4))
E4 = compute_error(Ground_Truth,Yhat_4)


#### Regression on Data 5
Yhat_5      = RegModel.fit(Data5, Ground_Truth).predict(Data5)
R2_D5       = r2_score(Yhat_5, Ground_Truth)
MAE_D5      = mean_absolute_error(Yhat_5, Ground_Truth)
MAE_D5_bot  = mean_absolute_error(Yhat_5[Ranking_lower30], Ground_Truth[Ranking_lower30])
MAE_D5_top  = mean_absolute_error(Yhat_5[Ranking_higher30], Ground_Truth[Ranking_higher30])
print("-----------Regression Results with Resgressor Non-Residential Ratio---------\n")
print("Coefficient 4---->",RegModel.coef_)
print("Rsq: {} \t MAE: {}".format(R2_D5, MAE_D5))
E5 = compute_error(Ground_Truth,Yhat_5)


GT_Mu_bot = np.sum(Ground_Truth[Ranking_lower50])
GT_Mu_top = np.sum(Ground_Truth[Ranking_higher50])


fig, (ax1,ax2) = plt.subplots(1,2)
fig.suptitle('Zones Ranked according to Non-Residential Ratio, Left-Bottom, Right-Top')
ax1.plot(Ground_Truth[Ranking_lower30])
ax1.plot(Yhat_1[Ranking_lower30])
ax1.plot(Yhat_2[Ranking_lower30])
ax1.plot(Yhat_3[Ranking_lower30])
ax1.plot(Yhat_4[Ranking_lower30])
ax1.plot(Yhat_5[Ranking_lower30])

ax1.legend(labels=(['GT :' + str(round(GT_Mu_bot,2)), 'Com + Ind :' + str(round(MAE_D1_bot*50,2)), 'Com + Ind + LM :' + str(round(MAE_D2_bot*50,2)),
            'Com + Ind + NC :' + str(round(MAE_D3_bot*50,2)), 'Com + Ind + LM + NC :' + str(round(MAE_D4_bot*50,2)), 'NonRes-Ratio :' + str(round(MAE_D5_bot*50,2))]), loc='upper left')

ax2.plot(Ground_Truth[Ranking_higher30])
ax2.plot(Yhat_1[Ranking_higher30])
ax2.plot(Yhat_2[Ranking_higher30])
ax2.plot(Yhat_3[Ranking_higher30])
ax2.plot(Yhat_4[Ranking_higher30])
ax2.plot(Yhat_5[Ranking_higher30])

ax2.legend(labels=(['GT :' + str(round(GT_Mu_top,2)), 'Com + Ind :' + str(round(MAE_D1_top*50,2)), 'Com + Ind + LM :' + str(round(MAE_D2_top*50,2)),
            'Com + Ind + NC :' + str(round(MAE_D3_top*50,2)), 'Com + Ind + LM + NC :' + str(round(MAE_D4_top*50,2)), 'NonRes-Ratio :' + str(round(MAE_D5_top*50,2))]), loc='upper left')
# plt.show()


fig, ((ax1, ax2, ax5), (ax3,ax4,ax6)) = plt.subplots(2,3)
fig.suptitle('MNF Correction with different predictors')
ax1.plot(Ground_Truth[Data_Ranking])
ax1.plot(E1[Data_Ranking])
ax1.legend(labels=("GT-MNF", "Corr-MNF"), loc='upper left')
ax1.title.set_text('Com + Ind')

ax2.plot(Ground_Truth[Data_Ranking])
ax2.plot(E2[Data_Ranking])
ax2.legend(labels=("GT-MNF", "Corr-MNF"), loc='upper left')
ax2.title.set_text('Com + Ind + LM')

ax3.plot(Ground_Truth[Data_Ranking])
ax3.plot(E3[Data_Ranking])
ax3.legend(labels=("GT-MNF", "Corr-MNF"), loc='upper left')
ax3.title.set_text('Com + Ind + NC')

ax4.plot(Ground_Truth[Data_Ranking])
ax4.plot(E4[Data_Ranking])
ax4.legend(labels=("GT-MNF", "Corr-MNF"), loc='upper left')
ax4.title.set_text('Com + Ind + LM + NC')

ax5.plot(Ground_Truth[Data_Ranking])
ax5.plot(E5[Data_Ranking])
ax5.legend(labels=("GT-MNF", "Corr-MNF"), loc='upper left')
ax5.title.set_text('NonRes-Ratio')

# plt.show()



###### Histogram plot for all data
fig, ax = plt.subplots(6,6)

##### Ground Truth
fig.suptitle('Histogram plots of the MNF')
ax[0,0].hist(Ground_Truth[Ranking_lower30], normed=True, bins=30)
ax[0,1].hist(Ground_Truth[Ranking_lower50], normed=True, bins=30)
ax[0,2].hist(Ground_Truth[Ranking_lower80], normed=True, bins=30)
ax[0,3].hist(Ground_Truth[Ranking_higher30], normed=True, bins=30)
ax[0,4].hist(Ground_Truth[Ranking_higher50], normed=True, bins=30)
ax[0,5].hist(Ground_Truth[Ranking_higher80], normed=True, bins=30)

##### Yhat1
ax[1,0].hist(Yhat_1[Ranking_lower30], normed=True, bins=30)
ax[1,1].hist(Yhat_1[Ranking_lower50], normed=True, bins=30)
ax[1,2].hist(Yhat_1[Ranking_lower80], normed=True, bins=30)
ax[1,3].hist(Yhat_1[Ranking_higher30], normed=True, bins=30)
ax[1,4].hist(Yhat_1[Ranking_higher50], normed=True, bins=30)
ax[1,5].hist(Yhat_1[Ranking_higher80], normed=True, bins=30)

##### Yhat2
ax[2,0].hist(Yhat_2[Ranking_lower30], normed=True, bins=30)
ax[2,1].hist(Yhat_2[Ranking_lower50], normed=True, bins=30)
ax[2,2].hist(Yhat_2[Ranking_lower80], normed=True, bins=30)
ax[2,3].hist(Yhat_2[Ranking_higher30], normed=True, bins=30)
ax[2,4].hist(Yhat_2[Ranking_higher50], normed=True, bins=30)
ax[2,5].hist(Yhat_2[Ranking_higher80], normed=True, bins=30)


##### Yhat3
ax[3,0].hist(Yhat_3[Ranking_lower30], normed=True, bins=30)
ax[3,1].hist(Yhat_3[Ranking_lower50], normed=True, bins=30)
ax[3,2].hist(Yhat_3[Ranking_lower80], normed=True, bins=30)
ax[3,3].hist(Yhat_3[Ranking_higher30], normed=True, bins=30)
ax[3,4].hist(Yhat_3[Ranking_higher50], normed=True, bins=30)
ax[3,5].hist(Yhat_3[Ranking_higher80], normed=True, bins=30)

##### Yhat4
ax[4,0].hist(Yhat_4[Ranking_lower30], normed=True, bins=30)
ax[4,1].hist(Yhat_4[Ranking_lower50], normed=True, bins=30)
ax[4,2].hist(Yhat_4[Ranking_lower80], normed=True, bins=30)
ax[4,3].hist(Yhat_4[Ranking_higher30], normed=True, bins=30)
ax[4,4].hist(Yhat_4[Ranking_higher50], normed=True, bins=30)
ax[4,5].hist(Yhat_4[Ranking_higher80], normed=True, bins=30)

##### Yhat5
ax[5,0].hist(Yhat_5[Ranking_lower30], normed=True, bins=30)
ax[5,1].hist(Yhat_5[Ranking_lower50], normed=True, bins=30)
ax[5,2].hist(Yhat_5[Ranking_lower80], normed=True, bins=30)
ax[5,3].hist(Yhat_5[Ranking_higher30], normed=True, bins=30)
ax[5,4].hist(Yhat_5[Ranking_higher50], normed=True, bins=30)
ax[5,5].hist(Yhat_5[Ranking_higher80], normed=True, bins=30)

# plt.show()



##### Print stats of Marfield if not dropped
if not(Drop_MARS):
    bar_data = [np.array(MNF_data.iloc[Marsfield_index,-5]),E1[Marsfield_index],E2[Marsfield_index],E3[Marsfield_index],E4[Marsfield_index],E5[Marsfield_index]]
    y_pos = np.arange(len(bar_data))
    bars = ('GT', 'Com_Ind', 'Com_Ind_LM', 'Com_Ind_NC', 'Com_Ind_LM_NC', 'NonRes-Ratio')
    plt.bar(y_pos,bar_data)
    plt.title('Regression Analysis for Marfield')
    plt.xticks(y_pos,bars,rotation=15)
    # plt.show()


fig, (ax1,ax2) = plt.subplots(1,2)
fig.suptitle('Zones Ranked according to Non-Residential Ratio, Left-Bottom, Right-Top')
ax1.plot(Ground_Truth[Ranking_lower50])
ax1.plot(Yhat_1[Ranking_lower50])
ax1.plot(E1[Ranking_lower50])



ax2.plot(Ground_Truth[Ranking_higher50])
ax2.plot(Yhat_1[Ranking_higher50])
ax2.plot(E1[Ranking_higher50])



plt.show()