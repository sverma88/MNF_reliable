"""
This scriot read the MNF files 1. ML-Day, 2. Summary LnC and aligns the number of connections and Length of Mains
"""


import numpy as np
import pandas as pd

from utils import *

'''
path settings for the data retrival and results accumulation
Remeber all indexing starts with 0 and not 1, i.e. including Pressure Zones
'''

data_dir = '/scratch1/ver100/Water_Project/data/'
start_rows = 8

ML_Day_data = pd.read_excel(data_dir + 'ML-Day.xlsx') # add the file name
LnC_data = pd.read_excel(data_dir + 'Summary LnC.xlsx')

print("----------------ML Data --------------")
print(ML_Day_data.head(10)) # Can put any integer to print the values
ML_Day_data[ML_Day_data != ML_Day_data] = 0  # rip of the Nan values

print("----------------Summary of Length and Connections Data --------------")
print(LnC_data.head(10)) # Can put any integer to print the values
LnC_data[LnC_data != LnC_data] = 0  # rip of the Nan values


Pressure_Zones_ML_day = list(ML_Day_data.iloc[0,:])
# print(Pressure_Zones_ML_day)

Pressure_Zones_SW = list(LnC_data.iloc[:-2,0])
Number_Connections_SW = list(LnC_data.iloc[:-2,1])
Length_Mains_SW = list(LnC_data.iloc[:-2,2])
# print(Pressure_Zones_SW)
# print(Number_Connections_SW)

matched_zones = [zones for zones in Pressure_Zones_ML_day if zones in Pressure_Zones_SW]
# print("Matched Zones",matched_zones)
# matched_zones_index_ML_day =  [Pressure_Zones_ML_day.index(zones) for zones in Pressure_Zones_ML_day if zones in Pressure_Zones_SW]
matched_zones_index_ML_day =  [Pressure_Zones_ML_day.index(zones) for zones in matched_zones]
matched_zones_index_SW =  [Pressure_Zones_SW.index(zones) for zones in matched_zones]
print(len(matched_zones))
print("------- Matched Pressure Zones---------")
print(matched_zones)
print("------- Index of Matched Pressure Zones ML-Day Johnny---------")
print(matched_zones_index_ML_day)
print("------- Index of Matched Pressure Zones Sydeny Waters, Krish---------")
print(matched_zones_index_SW)
# print(Pressure_Zones_SW)

# for iter,zones in enumerate(matched_zones):
#     ML_Day_data.iloc[]
#

# index = 2
# print("previous NC",ML_Day_data.iloc[6,matched_zones_index_ML_day[index]])
# print("previous LM",ML_Day_data.iloc[7,matched_zones_index_ML_day[index]])
# ML_Day_data.iloc[6,matched_zones_index_ML_day[index]]  = Number_Connections_SW[matched_zones_index_SW[index]]
# ML_Day_data.iloc[7,matched_zones_index_ML_day[index]]  = Length_Mains_SW[matched_zones_index_SW[index]]
# print("revised NC",ML_Day_data.iloc[6,matched_zones_index_ML_day[index]])
# print("revised LM",ML_Day_data.iloc[7,matched_zones_index_ML_day[index]])
#


for iter, zones in enumerate(matched_zones):
    print("Zone --->",zones)
    print("previous NC",ML_Day_data.iloc[6,matched_zones_index_ML_day[iter]])
    print("previous LC",ML_Day_data.iloc[7,matched_zones_index_ML_day[iter]])
    ML_Day_data.iloc[6,matched_zones_index_ML_day[iter]]  = Number_Connections_SW[matched_zones_index_SW[iter]]
    ML_Day_data.iloc[7,matched_zones_index_ML_day[iter]]  = Length_Mains_SW[matched_zones_index_SW[iter]]
    print("Updated LC",ML_Day_data.iloc[7,matched_zones_index_ML_day[iter]])
    print("Updated NC",ML_Day_data.iloc[6,matched_zones_index_ML_day[iter]])

with pd.ExcelWriter(data_dir + 'ML_day_updated.xlsx') as writer:
        ML_Day_data.to_excel(writer, "data", header=True, index=False)










