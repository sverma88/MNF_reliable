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
alpha       = 0.0
fit_inter   = False
Drop_MARS   = True
fit_lasso   = True

Zones_Names_col        = 1
NonRes_rate_col        = 11
Commercial_col         = 16
Industrial_col         = 19
Agric_rate_col         = 27
BabyH_rate_col         = 28
Church_rate_col        = 29
Club_rate_col          = 30
Commer_rate_col        = 31
Govt_rate_col          = 32
Hotel_rate_col         = 33
Industry_rate_col      = 34
PvtSch_rate_col        = 35
PvtHosp_rate_col       = 36
PubChrt_rate_col       = 37
PubHosp_rate_col       = 38
PubResv_rate_col       = 39
School_rate_col        = 40
UniPvt_rate_col        = 41
Zones_MNF_col          = 42
NumConnect_col         = 43
LengthMains_col        = 44





if fit_lasso:
    RegModel = Lasso(alpha = alpha, fit_intercept = fit_inter)
else:
    RegModel = LinearRegression(fit_intercept = fit_inter)

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

#### These are the variables
LenMains_data    = np.array(MNF_data.iloc[:, LengthMains_col])
NumConnec_data   = np.array(MNF_data.iloc[:, NumConnect_col])
IndustryRatio    = np.array(MNF_data.iloc[:, Industry_rate_col])
CommerctRatio    = np.array(MNF_data.iloc[:, Commer_rate_col])
AgricultRatio    = np.array(MNF_data.iloc[:, Agric_rate_col])
BabyHltRatio     = np.array(MNF_data.iloc[:, BabyH_rate_col])
ChurchPptyRatio  = np.array(MNF_data.iloc[:, Church_rate_col])
ClubRatio        = np.array(MNF_data.iloc[:, Club_rate_col])
GovermtRatio     = np.array(MNF_data.iloc[:, Govt_rate_col])
HotelRatio       = np.array(MNF_data.iloc[:, Hotel_rate_col])
PvtSchlRatio     = np.array(MNF_data.iloc[:, PvtSch_rate_col])
PvtHospRatio     = np.array(MNF_data.iloc[:, PvtHosp_rate_col])
PubChrtyRatio    = np.array(MNF_data.iloc[:, PubChrt_rate_col])
PubHospRatio     = np.array(MNF_data.iloc[:, PubHosp_rate_col])
PubResvRatio     = np.array(MNF_data.iloc[:, PubResv_rate_col])
Schl_StRatio     = np.array(MNF_data.iloc[:, School_rate_col])
PvtUniRatio      = np.array(MNF_data.iloc[:, UniPvt_rate_col])

CommercialVal    = np.array(MNF_data.iloc[:, Commercial_col])
IndustrialVal    = np.array(MNF_data.iloc[:, Industrial_col])

NonZero_IndustryRatio  = np.nonzero(IndustryRatio > 0)
Zero_IndustryRatio     = np.nonzero(IndustryRatio == 0)

Rest_Data = np.sum((AgricultRatio, BabyHltRatio, ChurchPptyRatio, ClubRatio, GovermtRatio,
                    HotelRatio, PvtSchlRatio, PvtHospRatio, PubChrtyRatio, PubHospRatio,
                    PubResvRatio, Schl_StRatio, PvtUniRatio), axis=0)

##### Create Data and normalize them
Data       = np.stack((CommerctRatio, IndustryRatio, Rest_Data), axis=1)
Std_Scaler = StandardScaler()
MM_Scaler  = MinMaxScaler()
Std_Scaler.fit(Data)
MM_Scaler.fit(Data)


Std_Data    = np.stack((CommerctRatio/(max(CommerctRatio) - min(CommerctRatio)),
                        IndustryRatio/ (max(IndustryRatio) - min(IndustryRatio)),
                        Rest_Data/ (max(Rest_Data) - min(Rest_Data))), axis=1)
MM_Data     = MM_Scaler.transform(Data)
UnScl_Data  = np.copy(Data)


#### Regression on Scaling by x/(max(x) - min(x)) Data
Std_Yhat      = RegModel.fit(Std_Data, Ground_Truth).predict(Std_Data)
R2_Std        = r2_score(Std_Yhat, Ground_Truth)
MAE_Std       = mean_absolute_error(Std_Yhat, Ground_Truth)
MAE_Std_bot   = mean_absolute_error(Std_Yhat[Ranking_lower], Ground_Truth[Ranking_lower])
MAE_Std_top   = mean_absolute_error(Std_Yhat[Ranking_higher], Ground_Truth[Ranking_higher])
Std_Coef      = RegModel.coef_
print("-----------Regression Results with Standard Scaler ---------\n")
print("Coefficient ---->",RegModel.coef_)
print("Rsq: {} \t MAE: {}".format(R2_Std, MAE_Std))
E_Std, Std_Pos, Std_Neg = compute_error(Ground_Truth,Std_Yhat)


#### Regression on MinMax Scaler Data
MM_Yhat      = RegModel.fit(MM_Data, Ground_Truth).predict(MM_Data)
R2_MM        = r2_score(MM_Yhat, Ground_Truth)
MAE_MM       = mean_absolute_error(MM_Yhat, Ground_Truth)
MAE_MM_bot   = mean_absolute_error(MM_Yhat[Ranking_lower], Ground_Truth[Ranking_lower])
MAE_MM_top   = mean_absolute_error(MM_Yhat[Ranking_higher], Ground_Truth[Ranking_higher])
MM_Coef      = RegModel.coef_
print("-----------Regression Results with MinMax Scaler ---------\n")
print("Coefficient ---->",RegModel.coef_)
print("Rsq: {} \t MAE: {}".format(R2_MM, MAE_MM))
E_MM, MM_Pos, MM_Neg = compute_error(Ground_Truth,MM_Yhat)


#### Regression on UnScaled Data
UnScl_Yhat      = RegModel.fit(UnScl_Data, Ground_Truth).predict(UnScl_Data)
R2_UnScl        = r2_score(UnScl_Yhat, Ground_Truth)
MAE_UnScl       = mean_absolute_error(UnScl_Yhat, Ground_Truth)
MAE_UnScl_bot   = mean_absolute_error(UnScl_Yhat[Ranking_lower], Ground_Truth[Ranking_lower])
MAE_UnScl_top   = mean_absolute_error(UnScl_Yhat[Ranking_higher], Ground_Truth[Ranking_higher])
UnScl_Coef      = RegModel.coef_
print("-----------Regression Results with Raw Data ---------\n")
print("Coefficient ---->",RegModel.coef_)
print("Rsq: {} \t MAE: {}".format(R2_UnScl, MAE_UnScl))
E_UnScl, UnScl_Pos, UnScl_Neg = compute_error(Ground_Truth,UnScl_Yhat)


GT_Mu_bot = np.sum(Ground_Truth[Ranking_lower])
GT_Mu_top = np.sum(Ground_Truth[Ranking_higher])


###### Create Plot for the Standard Scaler
fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(20, 10))
fig.suptitle('MNF Correction with Standard Scaling :  x/(max(x) - min(x))')
ax1.plot(Ground_Truth[Ranking_lower])
ax1.plot(E_Std[Ranking_lower])
ax1.text(0.71,0.78,'C1: {}  \nC2: {} \nC3: {}'.format(round(Std_Coef[0],2),round(Std_Coef[1],2),round(Std_Coef[2],2)),
         horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
ax1.legend(labels=("GT-MNF {}".format(round(np.mean(Ground_Truth[Ranking_lower]),2)),
                   "Corr-MNF: {}".format(round(np.mean(E_Std[Ranking_lower]),2))), loc='upper left')
ax1.title.set_text('Bottom 1/3')

ax2.plot(Ground_Truth[Ranking_middle])
ax2.plot(E_Std[Ranking_middle])
ax2.legend(labels=("GT-MNF: {}".format(round(np.mean(Ground_Truth[Ranking_middle]),2)),
                            "Corr-MNF: {}".format(round(np.mean(E_Std[Ranking_middle]),2))), loc='upper left')
ax2.title.set_text('Middle 1/3')

ax3.plot(Ground_Truth[Ranking_higher])
ax3.plot(E_Std[Ranking_higher])
ax3.legend(labels=("GT-MNF: {}".format(round(np.mean(Ground_Truth[Ranking_higher]),2)),
                   "Corr-MNF: {}".format(round(np.mean(E_Std[Ranking_higher]),2))), loc='upper left')
ax3.title.set_text('Top 1/3')

ax4.hist(E_Std[Ranking_lower], bins=30)
ax4.set(ylabel='Frequency')
ax4.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_Std[Ranking_lower]),count_n(E_Std[Ranking_lower])),
         horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
ax5.hist(E_Std[Ranking_middle], bins=30)
ax5.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_Std[Ranking_middle]),count_n(E_Std[Ranking_middle])),
         horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
ax6.hist(E_Std[Ranking_higher], bins=30)
ax6.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_Std[Ranking_higher]),count_n(E_Std[Ranking_higher])),
         horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)

plt.savefig('/scratch1/ver100/MNF_Regression/Regression_plots/CIR_Lasso_0.0_Standardize.png')

###### Create Plot for the MinMax Scaler
fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(20, 10))
fig.suptitle('MNF Correction with MinMaxScaling :  x - min(x)/(max(x) - min(x))')
ax1.plot(Ground_Truth[Ranking_lower])
ax1.plot(E_MM[Ranking_lower])
ax1.text(0.71,0.78,'C1: {}  \nC2: {} \nC3: {}'.format(round(MM_Coef[0],2),round(MM_Coef[1],2),round(MM_Coef[2],2)),
         horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
ax1.legend(labels=("GT-MNF: {}".format(round(np.mean(Ground_Truth[Ranking_lower]),2)),
                   "Corr-MNF: {}".format(round(np.mean(E_MM[Ranking_lower]),2))), loc='upper left')
ax1.title.set_text('Bottom 1/3')

ax2.plot(Ground_Truth[Ranking_middle])
ax2.plot(E_MM[Ranking_middle])
ax2.legend(labels=("GT-MNF: {}".format(round(np.mean(Ground_Truth[Ranking_middle]),2)),
                   "Corr-MNF: {}".format(round(np.mean(E_MM[Ranking_middle]),2))), loc='upper left')
ax2.title.set_text('Middle 1/3')

ax3.plot(Ground_Truth[Ranking_higher])
ax3.plot(E_MM[Ranking_higher])
ax3.legend(labels=("GT-MNF: {}".format(round(np.mean(Ground_Truth[Ranking_higher]),2)),
                   "Corr-MNF: {}".format(round(np.mean(E_MM[Ranking_higher]),2))), loc='upper left')
ax3.title.set_text('Top 1/3')

ax4.hist(E_MM[Ranking_lower], bins=30)
ax4.set(ylabel='Frequency')
ax4.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_MM[Ranking_lower]),count_n(E_MM[Ranking_lower])),
         horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
ax5.hist(E_MM[Ranking_middle], bins=30)
ax5.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_MM[Ranking_middle]),count_n(E_MM[Ranking_middle])),
         horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
ax6.hist(E_MM[Ranking_higher], bins=30)
ax6.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_MM[Ranking_higher]),count_n(E_MM[Ranking_higher])),
         horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)

plt.savefig('/scratch1/ver100/MNF_Regression/Regression_plots/CIR_Lasso_0.0_MinMax.png')



###### Create Plot for the Unscaled Data
fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(20, 10))
fig.suptitle('MNF Correction with No Scaling :  x ')
ax1.plot(Ground_Truth[Ranking_lower])
ax1.plot(E_UnScl[Ranking_lower])
ax1.text(0.71,0.78,'C1: {}  \nC2: {} \nC3: {}'.format(round(UnScl_Coef[0],2),round(UnScl_Coef[1],2),round(UnScl_Coef[2],2)),
         horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
ax1.legend(labels=("GT-MNF: {}".format(round(np.mean(Ground_Truth[Ranking_lower]),2)),
                   "Corr-MNF: {}".format(round(np.mean(E_UnScl[Ranking_lower]),2))), loc='upper left')
ax1.title.set_text('Bottom 1/3')

ax2.plot(Ground_Truth[Ranking_middle])
ax2.plot(E_UnScl[Ranking_middle])
ax2.legend(labels=("GT-MNF: {}".format(round(np.mean(Ground_Truth[Ranking_middle]),2)),
                   "Corr-MNF: {}".format(round(np.mean(E_UnScl[Ranking_middle]),2))), loc='upper left')
ax2.title.set_text('Middle 1/3')

ax3.plot(Ground_Truth[Ranking_higher])
ax3.plot(E_UnScl[Ranking_higher])
ax3.legend(labels=("GT-MNF: {}".format(round(np.mean(Ground_Truth[Ranking_higher]),2)),
                   "Corr-MNF: {}".format(round(np.mean(E_UnScl[Ranking_higher]),2))), loc='upper left')
ax3.title.set_text('Top 1/3')

ax4.hist(E_UnScl[Ranking_lower], bins=30)
ax4.set(ylabel='Frequency')
ax4.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_UnScl[Ranking_lower]),count_n(E_UnScl[Ranking_lower])),
         horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
ax5.hist(E_UnScl[Ranking_middle], bins=30)
ax5.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_UnScl[Ranking_middle]),count_n(E_UnScl[Ranking_middle])),
         horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
ax6.hist(E_UnScl[Ranking_higher], bins=30)
ax6.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_UnScl[Ranking_higher]),count_n(E_UnScl[Ranking_higher])),
         horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)


plt.savefig('/scratch1/ver100/MNF_Regression/Regression_plots/CIR_Lasso_0.0_UnScaled.png')

plt.show()