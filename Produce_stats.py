"""
This scriot read the aligned MNF files and produce the statistics
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

data_dir = '/scratch1/ver100/Water_Project/data/'
plots_dir = '/scratch1/ver100/MNF_reliable/plots/'
start_rows = 8

ML_Day_data = pd.read_excel(data_dir + 'ML_day_updated.xlsx')  # add the file name
data_cols = ML_Day_data.shape[1]

LnC_data = pd.read_excel(data_dir + 'Summary LnC.xlsx')

print("----------------ML Data --------------")
print(ML_Day_data.head(10))  # Can put any integer to print the values
ML_Day_data[ML_Day_data != ML_Day_data] = 0  # rip of the Nan values

print("----------------Summary of Length and Connections Data --------------")
LnC_data = LnC_data.iloc[:,:4]  # Need to do this as their is legend in the sheet
LnC_data[LnC_data != LnC_data] = 0  # rip of the Nan values
print(LnC_data.head(10))  # Can put any integer to print the values


Pressure_Zones_ML_day    = list(ML_Day_data.iloc[0, :])
Pressure_Zones_SW        = LnC_data.iloc[:,[0,3]]
Reduced_LnC              = LnC_data[LnC_data['Unnamed: 3'] != 0]
Reduced_LnC_Steady       = LnC_data[LnC_data['Unnamed: 3'] == 'STEADY']
Reduced_LnC_Step_down    = LnC_data[LnC_data['Unnamed: 3'] == 'STEP DOWN']
Reduced_LnC_Step_up      = LnC_data[LnC_data['Unnamed: 3'] == 'STEP UP']
Reduced_LnC_Rising       = LnC_data[LnC_data['Unnamed: 3'] == 'RISING']
Reduced_LnC_Failing      = LnC_data[LnC_data['Unnamed: 3'] == 'FALLING']
# print(LnC_data[LnC_data['Unnamed: 3'] != 0])

Reliable_Zones_Names         = list(Reduced_LnC.iloc[1:,0])
Reliable_Zones_Steady        = list(Reduced_LnC_Steady.iloc[:,0])
Reliable_Zones_Step_down     = list(Reduced_LnC_Step_down.iloc[:,0])
Reliable_Zones_Step_up       = list(Reduced_LnC_Step_up.iloc[:,0])
Reliable_Zones_Rising        = list(Reduced_LnC_Rising.iloc[:,0])
Reliable_Zones_Falling       = list(Reduced_LnC_Failing.iloc[:,0])

print("----------- Reliable Zones --------------")
# print(Reliable_Zones_Step_up)
# print(Reduced_LnC_Step_up)
# print(Reduced_LnC)
print(len(Reliable_Zones_Names))


matched_zones = [zones for zones in Pressure_Zones_ML_day if zones in Reliable_Zones_Names]
Unmatched_zones = list(set(Reliable_Zones_Names)-set(matched_zones))
print("Unmatched Zones",Unmatched_zones)
matched_zones_index_ML_day      =  [Pressure_Zones_ML_day.index(zones) for zones in Reliable_Zones_Names]
matched_zones_index_Steady      =  [Pressure_Zones_ML_day.index(zones) for zones in Reliable_Zones_Steady]
matched_zones_index_Step_down   =  [Pressure_Zones_ML_day.index(zones) for zones in Reliable_Zones_Step_down]
matched_zones_index_Step_up     =  [Pressure_Zones_ML_day.index(zones) for zones in Reliable_Zones_Step_up]
matched_zones_index_Rising      =  [Pressure_Zones_ML_day.index(zones) for zones in Reliable_Zones_Rising]
matched_zones_index_Falling     =  [Pressure_Zones_ML_day.index(zones) for zones in Reliable_Zones_Falling]

Categorical_Zones_Index = [matched_zones_index_Steady, matched_zones_index_Falling, matched_zones_index_Step_down, matched_zones_index_Rising,
                           matched_zones_index_Step_up]  #### List of Lists
Zones_Categories = ['STEADY','FALLING','STEP DOWN','RISING','STEP UP','Others']

#### Find index of Unreliable/other Zones
all_indices = list(np.arange(1, data_cols))
all_reliable_zones = [item for sublist in Categorical_Zones_Index for item in sublist]
other_zones_index = list(set(all_indices) - set(all_reliable_zones))
Categorical_Zones_Index.append(other_zones_index)  #### append to the end of the list

# print(Categorical_Zones_Index)
# print(matched_zones_index_ML_day)
# print(ML_Day_data.iloc[:,matched_zones_index_ML_day[0]])
# print(len(matched_zones))

Categorical_zones_mnf_day, Reliable_zones_mnf_day, All_zones_mnf_day = \
    obtain_stats(ML_Day_data.iloc[start_rows:],matched_zones_index_ML_day,Categorical_Zones_Index,Zones_Categories)


# matched_zones = [zones for zones in Pressure_Zones_ML_day if zones in Pressure_Zones_SW]
# # print("Matched Zones",matched_zones)
# # matched_zones_index_ML_day =  [Pressure_Zones_ML_day.index(zones) for zones in Pressure_Zones_ML_day if zones in Pressure_Zo
# matched_zones_index_ML_day = [Pressure_Zones_ML_day.index(zones) for zones in matched_zones]
# matched_zones_index_SW = [Pressure_Zones_SW.index(zones) for zones in matched_zones]
# print(len(matched_zones))



#### Plot MNF/Day Distribution over all Zones #####
fig, ax = plt.subplots()

LnC_flag = list(Reduced_LnC.iloc[1:,3])
counts = Counter(LnC_flag)
common = counts.most_common()
common.append(('Others',len(other_zones_index)))
labels = [item[0] for item in common]
number = [item[1] for item in common]
nbars = len(common)
bar_plot = plt.bar(np.arange(nbars), number, tick_label=labels, orientation='vertical')
plt.ylabel('Frequency')
plt.title('MNF-ML/Day Categorical Distribution Over Zones')

bar_label = Categorical_zones_mnf_day
# print(LnC_flag)

# Put Lables on the bar plot MNF/Day
for idx,rect in enumerate(bar_plot):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
            np.round(bar_label[idx],3),
            ha='center', va='bottom', rotation=0, fontweight='bold')

plt.ylim(0,max(number)+30)
plt.text(0.7, 0.88, 'MNF-ML/Day Reliable {}\nMNF-ML/Day Others {}'.format(np.round(Reliable_zones_mnf_day,4), np.round(bar_label[-1],4))
         , horizontalalignment='center', verticalalignment='center',
         transform=ax.transAxes, fontweight='bold')
plt.savefig(plots_dir + 'MNF_Distribution.pdf')
# plt.show()



#### Plot Distribution of Zones #####
fig, ax = plt.subplots()

LnC_flag = list(Reduced_LnC.iloc[1:,3])
counts = Counter(LnC_flag)
common = counts.most_common()
common.append(('Others',len(other_zones_index)))
labels = [item[0] for item in common]
number = [item[1] for item in common]
nbars = len(common)
bar_plot = plt.bar(np.arange(nbars), number, tick_label=labels, orientation='vertical')
plt.ylabel('Frequency')
plt.title('Categorical Distribution of Zones')

bar_label = number

# Put Lables on the bar plot MNF/Day
for idx,rect in enumerate(bar_plot):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
            np.round(bar_label[idx],3),
            ha='center', va='bottom', rotation=0, fontweight='bold')

plt.ylim(0,max(number)+30)
plt.text(0.7, 0.88, 'MNF-ML/Day all Zones {}'.format(np.round(All_zones_mnf_day,4))
         , horizontalalignment='center', verticalalignment='center',
         transform=ax.transAxes, fontweight='bold')
plt.savefig(plots_dir + 'Zones_Distribution.pdf')
# plt.show()