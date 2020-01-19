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



'''                                                                                                                     
path settings for the data retrival and results accumulation                                                            
Remeber all indexing starts with 0 and not 1, i.e. including headers and columns date                                            
'''

data_dir    = '/scratch1/ver100/Water_Project/data/'
plots_dir    = '/scratch1/ver100/MNF_reliable/plots/'
alpha       = 0.5

Zones_Names_col      = 1
Zones_MNF_col        = 42
Industry_rate_col    = 34
NonRes_rate_col      = 11
Commer_rate_col      = 31
LengthMains_col      = 44
NumConnect_col       = 43
lasso                = Lasso(alpha = alpha)

MNF_data             = pd.read_excel(data_dir +  'Regression_Model_Reliable_Zones_data.xlsx')

print("----------------Regression Data --------------")
MNF_data.fillna(0, inplace = True)  # rip of the Nan values
print(MNF_data.head(10))  # Can put any integer to print the values

#### Get all data first
Ground_Truth     = np.array(MNF_data.iloc[:, Zones_MNF_col])
NonResr_Ratio    = np.array(MNF_data.iloc[:, NonRes_rate_col])
Data_Ranking     = np.argsort(NonResr_Ratio)
Sieved_data      = Data_Ranking[:30]
Ranking_higher   = Data_Ranking[-30:]

#### This are the variables
LenMains_data    = np.array(MNF_data.iloc[:, LengthMains_col])
NumConnec_data   = np.array(MNF_data.iloc[:, NumConnect_col])
IndustryRatio    = np.array(MNF_data.iloc[:, Industry_rate_col])
CommerctRatio    = np.array(MNF_data.iloc[:, Commer_rate_col])

# print(Sieved_data)
# print(IndustryRatio)

##### Create Data and normalize them
Data1  = np.stack((CommerctRatio, IndustryRatio), axis=1)
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
Yhat_1      = lasso.fit(Data1, Ground_Truth).predict(Data1)
R2_D1       = r2_score(Yhat_1, Ground_Truth)
MAE_D1      = mean_absolute_error(Yhat_1, Ground_Truth)
MAE_D1_bot  = mean_absolute_error(Yhat_1[Sieved_data], Ground_Truth[Sieved_data])
MAE_D1_top  = mean_absolute_error(Yhat_1[Ranking_higher], Ground_Truth[Ranking_higher])
print("Rsq: {} \t MAE: {}".format(R2_D1, MAE_D1))

#### Regression on Data 2
Yhat_2      = lasso.fit(Data2, Ground_Truth).predict(Data2)
R2_D2       = r2_score(Yhat_2, Ground_Truth)
MAE_D2      = mean_absolute_error(Yhat_2, Ground_Truth)
MAE_D2_bot  = mean_absolute_error(Yhat_2[Sieved_data], Ground_Truth[Sieved_data])
MAE_D2_top  = mean_absolute_error(Yhat_2[Ranking_higher], Ground_Truth[Ranking_higher])
print("Rsq: {} \t MAE: {}".format(R2_D2, MAE_D2))


#### Regression on Data 3
Yhat_3      = lasso.fit(Data3, Ground_Truth).predict(Data3)
R2_D3       = r2_score(Yhat_3, Ground_Truth)
MAE_D3      = mean_absolute_error(Yhat_3, Ground_Truth)
MAE_D3_bot  = mean_absolute_error(Yhat_3[Sieved_data], Ground_Truth[Sieved_data])
MAE_D3_top  = mean_absolute_error(Yhat_3[Ranking_higher], Ground_Truth[Ranking_higher])
print("Rsq: {} \t MAE: {}".format(R2_D3, MAE_D3))


#### Regression on Data 4
Yhat_4      = lasso.fit(Data4, Ground_Truth).predict(Data4)
R2_D4       = r2_score(Yhat_4, Ground_Truth)
MAE_D4      = mean_absolute_error(Yhat_4, Ground_Truth)
MAE_D4_bot  = mean_absolute_error(Yhat_4[Sieved_data], Ground_Truth[Sieved_data])
MAE_D4_top  = mean_absolute_error(Yhat_4[Ranking_higher], Ground_Truth[Ranking_higher])
print("Rsq: {} \t MAE: {}".format(R2_D4, MAE_D4))

GT_Mu_bot = np.sum(Ground_Truth[Sieved_data])
GT_Mu_top = np.sum(Ground_Truth[Ranking_higher])


plt.plot(Ground_Truth[Sieved_data])
plt.plot(Yhat_1[Sieved_data])
plt.plot(Yhat_2[Sieved_data])
plt.plot(Yhat_3[Sieved_data])
plt.plot(Yhat_4[Sieved_data])
plt.xlabel("Smallest 30 Zones according to Non-Residential Ratio")
plt.ylabel("Actual and Predicted MNF")
plt.legend(['GT :' + str(round(GT_Mu_bot,2)), 'ComRat + IndRat: ' + str(round(MAE_D1_bot*30,2)), 'ComRat + IndRat + LenMains: ' + str(round(MAE_D2_bot*30,2)),
            'ComRat + IndRat + NumConn: ' + str(round(MAE_D3_bot*30,2)), 'ComRat + IndRat + LenMains + NumConn: ' + str(round(MAE_D4_bot*30,2))])
plt.savefig(plots_dir + 'Lasso_{}_Bottom.pdf'.format(alpha))
plt.show()
plt.close()



plt.plot(Ground_Truth[Ranking_higher])
plt.plot(Yhat_1[Ranking_higher])
plt.plot(Yhat_2[Ranking_higher])
plt.plot(Yhat_3[Ranking_higher])
plt.plot(Yhat_4[Ranking_higher])
plt.legend(['GT :' + str(round(GT_Mu_top,2)), 'ComRat + IndRat :' + str(round(MAE_D1_top*30,2)), 'ComRat + IndRat + LenMains :' + str(round(MAE_D2_top*30,2)),
            'ComRat + IndRat + NumConn :' + str(round(MAE_D3_top*30,2)), 'ComRat + IndRat + LenMains + NumConn :' + str(round(MAE_D4_top*30,2))])
plt.savefig(plots_dir + 'Lasso_{}_Top.pdf'.format(alpha))
plt.show()
plt.close()