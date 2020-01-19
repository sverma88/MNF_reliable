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
fit_lasso   = True

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

NonZero_IndustryRatio  = np.nonzero(IndustryRatio > 0)
Zero_IndustryRatio     = np.nonzero(IndustryRatio == 0)


##### Create Data and normalize them
Data       = np.stack((CommerctRatio, IndustryRatio), axis=1)
Std_Scaler = StandardScaler()
MM_Scaler  = MinMaxScaler()
Std_Scaler.fit(Data)
MM_Scaler.fit(Data)


Std_Data    = np.stack((CommerctRatio/(max(CommerctRatio) - min(CommerctRatio)),
                        IndustryRatio/ (max(IndustryRatio) - min(IndustryRatio))), axis=1)
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

'''
fig, (ax1,ax2, ax3) = plt.subplots(1,3,figsize=(20, 10))
fig.suptitle('Zones Ranked according to Non-Residential Ratio')
ax1.plot(Ground_Truth[Ranking_lower])
ax1.plot(Std_Yhat[Ranking_lower])
ax1.plot(MM_Yhat[Ranking_lower])
ax1.plot(UnScl_Yhat[Ranking_lower])
ax1.set(title='Bottom-80')
ax1.legend(labels=(['GT', 'Standardize', 'MinMax','UnScaled']), loc='upper left')

ax2.plot(Ground_Truth[Ranking_middle])
ax2.plot(Std_Yhat[Ranking_middle])
ax2.plot(MM_Yhat[Ranking_middle])
ax2.plot(UnScl_Yhat[Ranking_middle])
ax2.set(title='Middle')
ax2.legend(labels=(['GT', 'Standardize', 'MinMax','UnScaled']), loc='upper left')

ax3.plot(Ground_Truth[Ranking_higher])
ax3.plot(Std_Yhat[Ranking_higher])
ax3.plot(MM_Yhat[Ranking_higher])
ax3.plot(UnScl_Yhat[Ranking_higher])
ax3.set(title='Top-80')
ax3.legend(labels=(['GT', 'Standardize', 'MinMax','UnScaled']), loc='upper left')

'''


###### Create Plot for the Standard Scaler
fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(20, 10))
fig.suptitle('MNF Correction with Standard Scaling :  x/(max(x) - min(x))')
ax1.plot(Ground_Truth[Ranking_lower])
ax1.plot(E_Std[Ranking_lower])
ax1.text(0.71,0.78,'C1: {}  C2: {}'.format(round(Std_Coef[0],2),round(Std_Coef[1],2)),
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

ax4.hist(E_Std[Ranking_lower])
ax4.set(ylabel='Frequency')
ax4.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_Std[Ranking_lower]),count_n(E_Std[Ranking_lower])),
         horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
ax5.hist(E_Std[Ranking_middle])
ax5.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_Std[Ranking_middle]),count_n(E_Std[Ranking_middle])),
         horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
ax6.hist(E_Std[Ranking_higher])
ax6.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_Std[Ranking_higher]),count_n(E_Std[Ranking_higher])),
         horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)

plt.savefig('/scratch1/ver100/MNF_Regression/Regression_plots/Lasso_0.5_Standardize.png')

###### Create Plot for the MinMax Scaler
fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(20, 10))
fig.suptitle('MNF Correction with MinMaxScaling :  x - min(x)/(max(x) - min(x))')
ax1.plot(Ground_Truth[Ranking_lower])
ax1.plot(E_MM[Ranking_lower])
ax1.text(0.71,0.78,'C1: {}  C2: {}'.format(round(MM_Coef[0],2),round(MM_Coef[1],2)),
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

ax4.hist(E_MM[Ranking_lower])
ax4.set(ylabel='Frequency')
ax4.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_MM[Ranking_lower]),count_n(E_MM[Ranking_lower])),
         horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
ax5.hist(E_MM[Ranking_middle])
ax5.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_MM[Ranking_middle]),count_n(E_MM[Ranking_middle])),
         horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
ax6.hist(E_MM[Ranking_higher])
ax6.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_MM[Ranking_higher]),count_n(E_MM[Ranking_higher])),
         horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)

plt.savefig('/scratch1/ver100/MNF_Regression/Regression_plots/Lasso_0.5_MinMax.png')




###### Create Plot for the Unscaled Data
fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(20, 10))
fig.suptitle('MNF Correction with No Scaling :  x ')
ax1.plot(Ground_Truth[Ranking_lower])
ax1.plot(E_UnScl[Ranking_lower])
ax1.text(0.71,0.78,'C1: {}  C2: {}'.format(round(UnScl_Coef[0],2),round(UnScl_Coef[1],2)),
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

ax4.hist(E_UnScl[Ranking_lower])
ax4.set(ylabel='Frequency')
ax4.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_UnScl[Ranking_lower]),count_n(E_UnScl[Ranking_lower])),
         horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
ax5.hist(E_UnScl[Ranking_middle])
ax5.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_UnScl[Ranking_middle]),count_n(E_UnScl[Ranking_middle])),
         horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
ax6.hist(E_UnScl[Ranking_higher])
ax6.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(count_p(E_UnScl[Ranking_higher]),count_n(E_UnScl[Ranking_higher])),
         horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)


plt.savefig('/scratch1/ver100/MNF_Regression/Regression_plots/Lasso_0.5_UnScaled.png')


'''
fig, ((ax1, ax2, ax3), (ax4,ax5,ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4,3,figsize=(20, 10))
fig.suptitle('MNF Correction with different Scaling')
ax1.plot(Ground_Truth[Data_Ranking])
ax1.plot(E_Std[Data_Ranking])
ax1.legend(labels=("GT-MNF", "Corr-MNF: {}".format(round(np.mean(E_Std),2))), loc='upper left')
ax1.title.set_text('Standardize')

ax2.plot(Ground_Truth[Data_Ranking])
ax2.plot(E_MM[Data_Ranking])
ax2.legend(labels=("GT-MNF", "Corr-MNF: {}".format(round(np.mean(E_MM),2))), loc='upper left')
ax2.title.set_text('MinMax')

ax3.plot(Ground_Truth[Data_Ranking])
ax3.plot(E_UnScl[Data_Ranking])
ax3.legend(labels=("GT-MNF", "Corr-MNF: {}".format(round(np.mean(E_UnScl),2))), loc='upper left')
ax3.title.set_text('UnScaled')

ax4.hist(E_Std[Ranking_lower], normed=True, bins=30)
ax4.set(ylabel='Bottom 80')
ax4.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(Std_Pos,Std_Neg), horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
ax5.hist(E_MM[Ranking_lower], normed=True, bins=30)
ax5.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(MM_Pos,MM_Neg), horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
ax6.hist(E_UnScl[Ranking_lower], normed=True, bins=30)
ax6.text(0.7,0.7,'Pos: {} \nNeg: {}'.format(UnScl_Pos,UnScl_Neg), horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)


ax7.hist(E_Std[Ranking_middle], normed=True, bins=30)
ax7.text(0.7,0.7,'C1: {} \nC2: {}'.format(round(Std_Coef[0],2),round(Std_Coef[1],2)), horizontalalignment='center', verticalalignment='center', transform=ax7.transAxes)
ax7.set(ylabel='Middle')
ax8.hist(E_MM[Ranking_middle], normed=True, bins=30)
ax8.text(0.7,0.7,'C1: {} \nC2: {}'.format(round(MM_Coef[0],2),round(MM_Coef[1],2)), horizontalalignment='center', verticalalignment='center', transform=ax8.transAxes)
ax9.hist(E_UnScl[Ranking_middle], normed=True, bins=30)
ax9.text(0.7,0.7,'C1: {} \nC2: {}'.format(round(UnScl_Coef[0],2),round(UnScl_Coef[1],2)), horizontalalignment='center', verticalalignment='center', transform=ax9.transAxes)

ax10.hist(E_Std[Ranking_higher], normed=True, bins=30)
ax10.text(0.7,0.7,'C1: {} \nC2: {}'.format(round(Std_Coef[0],2),round(Std_Coef[1],2)), horizontalalignment='center', verticalalignment='center', transform=ax7.transAxes)
ax10.set(ylabel='Top 80')
ax11.hist(E_MM[Ranking_higher], normed=True, bins=30)
ax11.text(0.7,0.7,'C1: {} \nC2: {}'.format(round(MM_Coef[0],2),round(MM_Coef[1],2)), horizontalalignment='center', verticalalignment='center', transform=ax8.transAxes)
ax12.hist(E_UnScl[Ranking_higher], normed=True, bins=30)
ax12.text(0.7,0.7,'C1: {} \nC2: {}'.format(round(UnScl_Coef[0],2),round(UnScl_Coef[1],2)), horizontalalignment='center', verticalalignment='center', transform=ax9.transAxes)


###### PLot Using Industrial Ratio Values
fig, ((ax1, ax2), (ax4,ax5)) = plt.subplots(2,2,figsize=(20, 10))
fig.suptitle('MNF Correction with different Scaling')
ax1.plot(Ground_Truth[NonZero_IndustryRatio])
ax1.plot(E_MM[NonZero_IndustryRatio])
ax1.legend(labels=("GT-MNF", "Corr-MNF: {}".format(round(np.mean(E_Std),2))), loc='upper left')
ax1.title.set_text('NonZero-Industrial Ratio')

ax2.plot(Ground_Truth[Zero_IndustryRatio])
ax2.plot(E_MM[Zero_IndustryRatio])
ax2.legend(labels=("GT-MNF", "Corr-MNF: {}".format(round(np.mean(E_MM),2))), loc='upper left')
ax2.title.set_text('Zero Industrial Ratio')


ax4.hist(E_MM[NonZero_IndustryRatio], normed=True, bins=30)
ax5.hist(E_MM[Zero_IndustryRatio], normed=True, bins=30)
'''

##### Print stats of Marfield if not dropped
# if not(Drop_MARS):
#     bar_data = [np.array(MNF_data.iloc[Marsfield_index,-5]),E_Std[Marsfield_index],E_MM[Marsfield_index],E_Std[Marsfield_index]]
#     y_pos = np.arange(len(bar_data))
#     bars = ('GT', 'Standard', 'MinMax', 'UnScaled')
#     plt.bar(y_pos,bar_data)
#     plt.title('Regression Analysis for Marfield')
#     plt.xticks(y_pos,bars,rotation=15)
#     # plt.show()
#
plt.show()