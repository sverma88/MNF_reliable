"""
This script trains regression on MNF data
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso, ElasticNet
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

Zones_Names_col      = 1
Zones_MNF_col        = 42
Industry_rate_col    = 34
NonRes_rate_col      = 11
Commer_rate_col      = 31
LengthMains_col      = 44
NumConnect_col       = 43
lasso                = Lasso(alpha = alpha, fit_intercept = fit_inter)

MNF_data             = pd.read_excel(data_dir +  'Regression_Model_Reliable_Zones_data.xlsx')

print("----------------Regression Data --------------")
MNF_data.fillna(0, inplace = True)  # rip of the Nan values
print(MNF_data.head(10))  # Can put any integer to print the values

Marsfield_index = MNF_data[MNF_data['pressure_zone'] == 'P_MARSFIELD'].index
#### Drop Marsfield as it's an outlier
if Drop_MARS:
    MNF_data.drop(Marsfield_index, inplace = True)

#### Get all data first
Ground_Truth     = np.array(MNF_data.iloc[:, Zones_MNF_col])
NonResr_Ratio    = np.array(MNF_data.iloc[:, NonRes_rate_col])
Data_Ranking     = np.argsort(NonResr_Ratio)
Ranking_lower    = Data_Ranking[:30]
Ranking_higher   = Data_Ranking[-30:]

print("Ranking Lower--->",Ranking_lower)
print("Ranking Higher--->",Ranking_higher)

#### This are the variables
LenMains_data    = np.array(MNF_data.iloc[:, LengthMains_col])
NumConnec_data   = np.array(MNF_data.iloc[:, NumConnect_col])
IndustryRatio    = np.array(MNF_data.iloc[:, Industry_rate_col])
CommerctRatio    = np.array(MNF_data.iloc[:, Commer_rate_col])

# print(Ranking_lower)
# print(IndustryRatio)

### Print Data Statistics
print("Data Statistics")
compute_mean_stats(Ground_Truth,Ranking_lower)

##### Create Data and normalize them
Data1  = np.stack((CommerctRatio, IndustryRatio), axis=1)
Data_unscaled = np.copy(Data1)
# Scaler1 = StandardScaler()
Scaler1 = MinMaxScaler()
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

#### Regression on Data 1
# Yhat_1      = lasso.fit(Data1, Ground_Truth).predict(Data1)
Yhat_1      = lasso.fit(Data1, Ground_Truth).predict(Data_unscaled)
print("Coefficient 1---->",lasso.coef_)
R2_D1       = r2_score(Yhat_1, Ground_Truth)
MAE_D1      = mean_absolute_error(Yhat_1, Ground_Truth)
MAE_D1_bot  = mean_absolute_error(Yhat_1[Ranking_lower], Ground_Truth[Ranking_lower])
MAE_D1_top  = mean_absolute_error(Yhat_1[Ranking_higher], Ground_Truth[Ranking_higher])
print("Rsq: {} \t MAE: {}".format(R2_D1, MAE_D1))

#### Regression on Data 2
Yhat_2      = lasso.fit(Data2, Ground_Truth).predict(Data2)
print("Coefficient 2---->",lasso.coef_)
R2_D2       = r2_score(Yhat_2, Ground_Truth)
MAE_D2      = mean_absolute_error(Yhat_2, Ground_Truth)
MAE_D2_bot  = mean_absolute_error(Yhat_2[Ranking_lower], Ground_Truth[Ranking_lower])
MAE_D2_top  = mean_absolute_error(Yhat_2[Ranking_higher], Ground_Truth[Ranking_higher])
print("Rsq: {} \t MAE: {}".format(R2_D2, MAE_D2))


#### Regression on Data 3
Yhat_3      = lasso.fit(Data3, Ground_Truth).predict(Data3)
print("Coefficient 3---->",lasso.coef_)
R2_D3       = r2_score(Yhat_3, Ground_Truth)
MAE_D3      = mean_absolute_error(Yhat_3, Ground_Truth)
MAE_D3_bot  = mean_absolute_error(Yhat_3[Ranking_lower], Ground_Truth[Ranking_lower])
MAE_D3_top  = mean_absolute_error(Yhat_3[Ranking_higher], Ground_Truth[Ranking_higher])
print("Rsq: {} \t MAE: {}".format(R2_D3, MAE_D3))


#### Regression on Data 4
Yhat_4      = lasso.fit(Data4, Ground_Truth).predict(Data4)
print("Coefficient 4---->",lasso.coef_)
R2_D4       = r2_score(Yhat_4, Ground_Truth)
MAE_D4      = mean_absolute_error(Yhat_4, Ground_Truth)
MAE_D4_bot  = mean_absolute_error(Yhat_4[Ranking_lower], Ground_Truth[Ranking_lower])
MAE_D4_top  = mean_absolute_error(Yhat_4[Ranking_higher], Ground_Truth[Ranking_higher])
print("Rsq: {} \t MAE: {}".format(R2_D4, MAE_D4))



GT_Mu_bot = np.sum(Ground_Truth[Ranking_lower])
GT_Mu_top = np.sum(Ground_Truth[Ranking_higher])

#
# plt.plot(Ground_Truth[Ranking_lower])
# plt.plot(Yhat_1[Ranking_lower])
# plt.plot(Yhat_2[Ranking_lower])
# plt.plot(Yhat_3[Ranking_lower])
# plt.plot(Yhat_4[Ranking_lower])
# plt.xlabel("Each unit is a zone in ascending order")
# plt.title("Smallest 30 Zones according to Non-Residential Ratio")
# plt.ylabel("Actual and Predicted MNF")
# plt.legend(['GT :' + str(round(GT_Mu_bot,2)), 'ComRat + IndRat: ' + str(round(MAE_D1_bot*30,2)), 'ComRat + IndRat + LenMains: ' + str(round(MAE_D2_bot*30,2)),
#             'ComRat + IndRat + NumConn: ' + str(round(MAE_D3_bot*30,2)), 'ComRat + IndRat + LenMains + NumConn: ' + str(round(MAE_D4_bot*30,2))])
# # plt.savefig(plots_dir + 'Lasso_{}_Bottom.pdf'.format(alpha))
# plt.show()
# plt.close()
#
#
#
# plt.plot(Ground_Truth[Ranking_higher])
# plt.plot(Yhat_1[Ranking_higher])
# plt.plot(Yhat_2[Ranking_higher])
# plt.plot(Yhat_3[Ranking_higher])
# plt.plot(Yhat_4[Ranking_higher])
# plt.xlabel("Each unit is a zone in ascending order")
# plt.title("Highest 30 Zones according to Non-Residential Ratio")
# plt.legend(['GT :' + str(round(GT_Mu_top,2)), 'ComRat + IndRat :' + str(round(MAE_D1_top*30,2)), 'ComRat + IndRat + LenMains :' + str(round(MAE_D2_top*30,2)),
#             'ComRat + IndRat + NumConn :' + str(round(MAE_D3_top*30,2)), 'ComRat + IndRat + LenMains + NumConn :' + str(round(MAE_D4_top*30,2))])
# # plt.savefig(plots_dir + 'Lasso_{}_Top.pdf'.format(alpha))
# plt.show()
# plt.close()


fig, (ax1,ax2) = plt.subplots(1,2)
fig.suptitle('Zones Ranked according to Non-Residential Ratio, Left-Bottom, Right-Top')
ax1.plot(Ground_Truth[Ranking_lower])
ax1.plot(Yhat_1[Ranking_lower])
ax1.plot(Yhat_2[Ranking_lower])
ax1.plot(Yhat_3[Ranking_lower])
ax1.plot(Yhat_4[Ranking_lower])

ax1.legend(labels=(['GT :' + str(round(GT_Mu_bot,2)), 'Com + Ind :' + str(round(MAE_D1_bot*30,2)), 'Com + Ind + LM :' + str(round(MAE_D2_bot*30,2)),
            'Com + Ind + NC :' + str(round(MAE_D3_bot*30,2)), 'Com + Ind + LM + NC :' + str(round(MAE_D4_bot*30,2))]), loc='upper left')

ax2.plot(Ground_Truth[Ranking_higher])
ax2.plot(Yhat_1[Ranking_higher])
ax2.plot(Yhat_2[Ranking_higher])
ax2.plot(Yhat_3[Ranking_higher])
ax2.plot(Yhat_4[Ranking_higher])
ax2.legend(labels=(['GT :' + str(round(GT_Mu_top,2)), 'Com + Ind :' + str(round(MAE_D1_top*30,2)), 'Com + Ind + LM :' + str(round(MAE_D2_top*30,2)),
            'Com + Ind + NC :' + str(round(MAE_D3_top*30,2)), 'Com + Ind + LM + NC :' + str(round(MAE_D4_top*30,2))]), loc='upper left')
plt.show()


E1 = compute_error(Ground_Truth,Yhat_1)
E2 = compute_error(Ground_Truth,Yhat_2)
E3 = compute_error(Ground_Truth,Yhat_3)
E4 = compute_error(Ground_Truth,Yhat_4)


fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2)
fig.suptitle('MNF Correction with different predictors')
ax1.plot(Ground_Truth[Data_Ranking])
ax1.plot(E1[Data_Ranking])
ax1.legend(labels=("GT-MNF", "Corr-MNF"), loc='lower left')
ax1.title.set_text('Com + Ind')

ax2.plot(Ground_Truth[Data_Ranking])
ax2.plot(E2[Data_Ranking])
ax2.legend(labels=("GT-MNF", "Corr-MNF"), loc='lower left')
ax2.title.set_text('Com + Ind + LM')

ax3.plot(Ground_Truth[Data_Ranking])
ax3.plot(E3[Data_Ranking])
ax3.legend(labels=("GT-MNF", "Corr-MNF"), loc='lower left')
ax4.title.set_text('Com + Ind + NC')

ax4.plot(Ground_Truth[Data_Ranking])
ax4.plot(E4[Data_Ranking])
ax4.legend(labels=("GT-MNF", "Corr-MNF"), loc='lower left')
ax4.title.set_text('Com + Ind + LM + NC')

plt.show()




##### Print stats of Marfield if not dropped
if not(Drop_MARS):
    bar_data = [np.array(MNF_data.iloc[Marsfield_index,-5]),E1[Marsfield_index],E2[Marsfield_index],E3[Marsfield_index],E4[Marsfield_index]]
    print(list(bar_data))
    y_pos = np.arange(len(bar_data))
    bars = ('GT', 'Com_Ind', 'Com_Ind_LM', 'Com_Ind_NC', 'Com_Ind_LM_NC')
    plt.bar(y_pos,bar_data)
    plt.title('Regression Analysis for Marfield')
    plt.xticks(y_pos,bars,rotation=15)
    plt.show()

