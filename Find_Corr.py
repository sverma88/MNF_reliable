"""
This scriot read the aligned MNF file and the industry usage files and find the correlation statistics
"""

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from utils import *

'''                                                                                                                     
path settings for the data retrival and results accumulation                                                            
Remeber all indexing starts with 0 and not 1, i.e. including headers and columns date                                            
'''

data_dir    = '/scratch1/ver100/Water_Project/data/'
plots_dir   = '/scratch1/ver100/MNF_reliable/plots/'
save_dir    = '/scratch1/ver100/MNF_reliable/data_generated/'

start_rows  = 8
NC_rows     = 6
LM_rows     = 7
PZ_rows     = 0
bill_perc   = 0.2
lim_A       = 6  # Lower Limit to calculate count of MNF's within bounds
lim_B       = 12 # Upper Limit to calculate count of MNF's within bounds

ML_Day_data       = pd.read_excel(data_dir + 'ML_day_updated.xlsx')  # add the file name
Billing_Data      = pd.read_excel(data_dir + 'Industry_water_consumption_pzone.xlsx')  # add the file name
Industrial_Data   = pd.read_excel(data_dir + 'MNF_IndustrialUsage_Data.xlsx')
LnC_data          = pd.read_excel(data_dir + 'Summary LnC.xlsx')


data_cols         = ML_Day_data.shape[1]
NC_data           = ML_Day_data.iloc[NC_rows,1:]
Pressure_Zones    = ML_Day_data.iloc[PZ_rows,1:]


print("----------------ML Data --------------")
print(ML_Day_data.head(10))  # Can put any integer to print the values
ML_Day_data[ML_Day_data != ML_Day_data] = 0  # rip of the Nan values

print("----------------MNF Industrial Data --------------")
print(Industrial_Data.head(10))  # Can put any integer to print the values
Industrial_Data.fillna(0, inplace = True) # rip of the Nan values

print("----------------Summary of Length and Connections Data --------------")
LnC_data = LnC_data.iloc[:,:4]  # Need to do this as their is legend in the sheet
LnC_data[LnC_data != LnC_data] = 0  # rip of the Nan values
Reliable_Zones          = LnC_data[LnC_data['Unnamed: 3'] != 0]
Reliable_Zones_Names    = list(Reliable_Zones.iloc[1:,0])
print(LnC_data.head(10))  # Can put any integer to print the values

print("---------------- Billing Data --------------")
print(Billing_Data.head(10))  # Can put any integer to print the values
Billing_Data.fillna(0, inplace = True)  # rip of the Nan values
Billing_Data = Billing_Data[Billing_Data['bill_14_NONRES_rate'] != 0]

# Billing_Data = Billing_Data.iloc[:,:2]
Billing_rates = np.array(Billing_Data.iloc[:,1])
# print(Billing_rates)


Billing_Qants        = {}   # dictionary to keep billing zones and their rates
Billing_Qants_Zones  = {}   # dictionary to keep billing zone names
Billing_Qants_rates  = {}   # dictionary to keep billing zone rates

# print(np.sum(Billing_rates))   For checking whether there exists a nan value

for iter in range(5):
    Boolean_matching = (Billing_rates > (iter*bill_perc)) & (Billing_rates <= ((iter+1)*bill_perc))
    Billing_Qants_rates[str(iter)]  = Billing_rates[Boolean_matching]
    Billing_Qants_Zones[str(iter)]  = Billing_Data.iloc[Boolean_matching,0]
    Billing_Qants[str(iter)] = dict(zip( Billing_Qants_Zones[str(iter)],Billing_Qants_rates[str(iter)]))

# print(" --------- Billing Dictionary ---------- \n")
# print((Billing_Qants['4']))  # Checking the output by printing the file

#### Now Start the matching


# MNF_Day_Conn_L, MNF_Count = Calculate_MNF_Day(ML_Day_data.iloc[start_rows:], NC_data, Pressure_Zones, lim_A, lim_B)
#
# Find_Zones_Corr(Billing_Qants, MNF_Day_Conn_L, plots_dir)

Update_reliable(Reliable_Zones_Names, Industrial_Data, save_dir)