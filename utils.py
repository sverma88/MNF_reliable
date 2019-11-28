import numpy as np
import datetime
import matplotlib.dates as mdates
from matplotlib.transforms import Transform
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import csv



def count_sparsity(data):
    '''

    :param data: A real matrix of m rows and n columns
    :return: returns the level of sparsity in each column
    '''

    print("data shape",data.shape)
    data_rows = data.shape[0]
    data_cols = data.shape[1]

    data_sparsity = np.count_nonzero(data==0, axis=0) # axis tell us zeros in each column

    sparsity_ratio =  data_sparsity / data_rows

    #### index of colums with no missing value
    non_sparse_index = np.reshape(np.array(np.nonzero(sparsity_ratio==0)), -1)

    return non_sparse_index[1:]

def data_statistics(data, index, k):
    '''

    :param
            data: A real matrix of m rows and n columns
            index: the months index, its a dictionary
            k: threshold, the number of days a month must have

    :return: returns the level of sparsity in each column
    '''

    print("data shape",data.shape)
    data  = np.array(data)

    data_rows = data.shape[0]
    data_cols = data.shape[1]

    data_sparsity = np.count_nonzero(data==0, axis=0) # axis tell us zeros in each column
    sparsity_ratio =  data_sparsity / data_rows

    #### index of colums with no missing value
    non_sparse_index = np.reshape(np.array(np.nonzero(sparsity_ratio==0)), -1)

    zones_non_sparse = len(non_sparse_index[1:])
    zones_sparse = data_cols - zones_non_sparse -1
    total_zones = data_cols - 1

    print("Total Zones ---->",total_zones)
    print("Sparse Zones ---->",zones_sparse)
    print("Non-Sparse Zones ---->",zones_non_sparse)

    data_june =  data[index['June']:index['July'],1:]
    data_july =  data[index['July']:index['Aug'],1:]
    data_aug =  data[index['Aug']:index['Sep'],1:]

    print("data june",data_june.shape)

    data_neg_june = np.count_nonzero(data_june<0, axis=0) # axis tell us zeros in each column
    data_neg_july = np.count_nonzero(data_july<0, axis=0) # axis tell us zeros in each column
    data_neg_aug  = np.count_nonzero(data_aug<0, axis=0) # axis tell us zeros in each column

    neg_ratio =  (data_neg_june >= k)*1 + (data_neg_july >= k)*1 + (data_neg_aug >= k)*1

    #### index of colums with negative values
    neg_sparse_index = np.reshape(np.array(np.nonzero(neg_ratio==3)), -1)
    zones_neg_sparse = len(neg_sparse_index)

    print("Zones with %d Negatives MNF values ---->" %k, zones_neg_sparse)

    neg_non_sparse = list(set(non_sparse_index[1:]) & set(neg_sparse_index))
    zones_neg_non = len(neg_non_sparse)

    print("Neg and Non-Sparse Zones with %d MNF Values" %k,zones_neg_non)


    data_sparsity_june = 30 - np.count_nonzero(data_june==0, axis=0) # axis tell us zeros in each column
    data_sparsity_july = 31 - np.count_nonzero(data_july==0, axis=0) # axis tell us zeros in each column
    data_sparsity_aug = 31 - np.count_nonzero(data_aug==0, axis=0) # axis tell us zeros in each column

    sparsity_ratio =  (data_sparsity_june >= k)*1 + (data_sparsity_july >= k)*1 + (data_sparsity_aug >= k)*1

    #### index of colums with no missing value
    non_sparse_winter_index = np.reshape(np.array(np.nonzero(sparsity_ratio==3)), -1)


    return non_sparse_winter_index[1:]

def count_sparsity_k(data, index, k):
    '''

    :param
            data: A real matrix of m rows and n columns
            index: the months index, its a dictionary
            k: threshold, the number of days a month must have

    :return: returns the level of sparsity in each column
    '''

    print("data shape",data.shape)

    data  = np.array(data)

    data_june =  data[index['June']:index['July'],:]
    data_july =  data[index['July']:index['Aug'],:]
    data_aug =  data[index['Aug']:index['Sep'],:]


    data_sparsity_june = 30 - np.count_nonzero(data_june==0, axis=0) # axis tell us zeros in each column
    data_sparsity_july = 31 - np.count_nonzero(data_july==0, axis=0) # axis tell us zeros in each column
    data_sparsity_aug = 31 - np.count_nonzero(data_aug==0, axis=0) # axis tell us zeros in each column

    sparsity_ratio =  (data_sparsity_june >= k)*1 + (data_sparsity_july >= k)*1 + (data_sparsity_aug >= k)*1

    #### index of colums with no missing value
    non_sparse_index = np.reshape(np.array(np.nonzero(sparsity_ratio==3)), -1)

    return non_sparse_index[1:]



def date2yday(x):
    """
    x is in matplotlib datenums, so they are floats.
    """
    y = x - mdates.date2num(datetime.datetime(2018, 1, 1))
    return y


def yday2date(x):
    """
    return a matplotlib datenum (x is days since start of year)
    """
    y = x + mdates.date2num(datetime.datetime(2018, 1, 1))
    return y

def obtain_stats(x, zone_index,zones_categorical_index, zones_categories):
# def obtain_stats(x):

    '''
    data starts from index 1 but is supplied from index 0 i.e. it includes dates in the first column
    :param x: the data
    :param zone_index: the list of zones index which are reliable
    :param zones_categorical_index: list-of-lists, where each list contains index of reliable zones
    :param zones_categories:  list of zones categories (steady, falling etc.)
    :return: statistics
    '''


    data = np.array(x)
    data_values = data[:,1:]

    total_days = data.shape[0]
    total_Zones = data_values.shape[1] -1

    absolute_sum = np.sum(np.absolute(data_values))  #take absolute of each value
    total_sum = np.sum(data_values)

    positive_values = (data_values >0)*data_values
    positive_sum = np.sum(positive_values)

    negative_value = (data_values <0)*data_values
    negative_sum = np.sum(negative_value)

    print("----- All Zones Statistics --------")
    print(" Total Zones ---->", total_Zones)
    print("All Zones MNF Absolute Sum/Day  ---->", absolute_sum / total_days)
    print("All Zones MNF Positive Sum/Day  ---->", positive_sum / total_days)
    print("All Zones MNF Negative Sum/Day  ---->", negative_sum / total_days)
    print("All Zones MNF Total Sum ---->", total_sum)
    print("All Zones MNF/Day  ---->",total_sum/total_days)
    print("\n ")
    All_zones_mnf_day = total_sum / total_days

    data_index = data[:,zone_index]
    # print(data_index.shape)
    # print(data_index[:,0])


    absolute_sum_indexed = np.sum(np.absolute(data_index))  # take absolute of each value
    total_sum_indexed = np.sum(data_index)

    positive_values_indexed = (data_index > 0) * data_index
    positive_sum_indexed = np.sum(positive_values_indexed)

    negative_value_indexed = (data_index < 0) * data_index
    negative_sum_indexed = np.sum(negative_value_indexed)

    print("----- Reliable Zones Statistics --------")
    print(" Total Zones ---->", len(zone_index))
    print("Reliable Zones MNF Absolute Sum/Day  ---->", absolute_sum_indexed / total_days)
    print("Reliable Zones MNF Positive Sum/Day ---->", positive_sum_indexed / total_days)
    print("Reliable Zones MNF Negative Sum/Day ---->", negative_sum_indexed / total_days)
    print("Reliable Zones MNF Total Sum ---->", total_sum_indexed)
    print("Reliable Zones MNF/Day ---->", total_sum_indexed / total_days)
    print("\n ")
    Reliable_zones_mnf_day = total_sum_indexed / total_days


    #### Reliable Zones Individual Statistics
    Categorical_zones_mnf_day = []



    for iter,categories in enumerate(zones_categories):
        print("\n \n")
        print("------ {} Zone -------".format(zones_categories[iter]))

       #### Calculate the statistics

        data_index = data[:, zones_categorical_index[iter]]

        absolute_sum_indexed = np.sum(np.absolute(data_index))  # take absolute of each value
        total_sum_indexed = np.sum(data_index)

        positive_values_indexed = (data_index > 0) * data_index
        positive_sum_indexed = np.sum(positive_values_indexed)

        negative_value_indexed = (data_index < 0) * data_index
        negative_sum_indexed = np.sum(negative_value_indexed)

        print("----- Zones Statistics --------")
        print(" Total Zones ---->", len(zones_categorical_index[iter]))
        print("MNF Absolute Sum/Day  ---->", absolute_sum_indexed / total_days)
        print("MNF Positive Sum/Day ---->", positive_sum_indexed / total_days)
        print("MNF Negative Sum/Day ---->", negative_sum_indexed / total_days)
        print("MNF Total Sum ---->", total_sum_indexed )
        print("MNF/Day ---->", total_sum_indexed / total_days)

        Categorical_zones_mnf_day.append(total_sum_indexed / total_days)


    return Categorical_zones_mnf_day, Reliable_zones_mnf_day, All_zones_mnf_day


def Calculate_MNF_Day(data,connections,zone_names, lim_A, lim_b):
    '''

    :param data         : Raw MNF data for each zone from Johnny's sheet, remember column '0' is date
    :param connections  : The Number of connections for each zone
    :param zone_names   : The name of each pressure zone
    :param lim_A        : Lower Limit to calculate count of MNF's within bounds
    :param lim_B        : Upper Limit to calculate count of MNF's within bounds
    :return             : MNF/NC for each zone
    '''

    data = np.array(data)
    data_values = data[:, 1:]
    total_days = data.shape[0]
    # print(data_values[0])

    connections = np.array(connections)
    connections[connections == 0] = 1.0

    zone_sum = np.sum(data_values, axis=0)
    MNF_Day_Conn_L = (zone_sum * 1000000) / (connections * total_days *24)

    # print(connections)
    # print(zone_sum)

    Boolean_matching = (MNF_Day_Conn_L > lim_A) & (MNF_Day_Conn_L <= lim_b)
    MNF_Count = sum(Boolean_matching*1)
    # print(Boolean_matching)

    MNF_Day_Conn_L = dict(zip(zone_names,MNF_Day_Conn_L))  # Dictionary of MNF Pressure Zones and their respective MNF's

    return MNF_Day_Conn_L, MNF_Count


def Find_Zones_Corr(Billing_Qants, MNF_Day_Conn_L, plots_dir):
    '''

    :param Billing_Qants        : Dictionary of dictionaries with Billing Qantas
    :param MNF_Day_Conn_L       : Dictionary of MNF Pressure Zones and their respective MNF
    :param plots_dir            : Directory to save the plots
    :return                     : Dictionary of MNF Pressure Zones and industrial usage each quant
    '''

    Pressure_Zones = MNF_Day_Conn_L.keys()
    Pressure_Zones_revised = [PZ[:-6] if ('_NO_PZ' in PZ) else PZ for PZ in Pressure_Zones]
    MNF_Day_Conn_L_values  = [MNF_Day_Conn_L[k] for k in Pressure_Zones]
    MNF_Day_Conn_L_revised = dict(zip(Pressure_Zones_revised, MNF_Day_Conn_L_values))
    # print(Pressure_Zones_revised)
    perc = 0.2

    for iter,key in enumerate(Billing_Qants.keys()):

        # matched_stats = csv.writer(open("matched_stats_{}.csv".format(iter), "w"))  # csv file to write the quats
        keys_quant = Billing_Qants[key].keys()
        Quant      = Billing_Qants[key]
        matched_keys = [k for k in keys_quant if k in Pressure_Zones_revised]
        Industrial_usage = [Quant[k] for k in matched_keys]
        Respective_MNF   = [MNF_Day_Conn_L_revised[k] for k in matched_keys]

        Matched_data = list(zip(matched_keys, Respective_MNF, Industrial_usage))
        Matched_data_length = len(Industrial_usage)

        with open(plots_dir + "matched_stats_{}.csv".format(iter), 'w') as csv_file:
            file_writer = csv.writer(csv_file)
            file_writer.writerow(['Pressure Zone', 'MNF', 'Industrial Usage'])
            for row in range(Matched_data_length):
                file_writer.writerow(Matched_data[row])


        corr = pearsonr(Industrial_usage,Respective_MNF)
        print('----- Billing Quant {} ---- \n'.format(key))
        print("Peasons's Correlation : ---->",corr[0])
        print("p-value : ---->",corr[1])

        # Plot the graphs
        fig, ax = plt.subplots()
        plt.plot(Respective_MNF, Industrial_usage, linestyle= 'none', marker='o')
        plt.ylabel('Industrial Usage: {} - {}'.format(np.round(perc*iter,1), np.round(perc*(iter+1),1)))
        plt.xlabel('MNF in L/D/Connection')
        plt.title('Correlation plot for MNF-Industrial Usage')
        plt.grid(True)
        plt.text(0.4, 0.88, 'Peasons\'s Correlation : {}'.format(np.round(corr[0],4))
                 , horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes, fontweight='bold', fontsize= 14)
        plt.savefig(plots_dir + 'Plt_Corr_Usage_{}_{}.png'.format(perc*iter, perc*(iter+1)), bbox_inches='tight')
        plt.savefig(plots_dir + 'Plt_Corr_Usage_{}_{}.pdf'.format(perc*iter, perc*(iter+1)))
        # plt.show()

        # print("-------- Matched Zones ---------")
        # print(matched_keys)
        #
        # print("-------- Industrial Usage ---------")
        # print(Industrial_usage)
        #
        print("-------- MNF Usage ---------")
        print(Respective_MNF)


def Update_reliable(Reliable_Zones, All_Zones, save_dir):

    '''

    :param Reliable_Zones   : Reliable Zones as per krish
    :param All_Zones        : Data from any existing Sheet
    :param save_dir         : Directory to save the sheet
    :return                 : none
    '''

    Zones_name       = All_Zones.iloc[:,0]
    MNF              = All_Zones.iloc[:,1]
    Industry_Usage   = All_Zones.iloc[:,2]
    Reliable_Zones   = [PZ[:-6] if ('_NO_PZ' in PZ) else PZ for PZ in Reliable_Zones] # Remove NO PZ as Zones Names does not have it
    Zone_Reliability = [PZ in Reliable_Zones for PZ in Zones_name]
    Zone_Reliability = list(np.array(Zone_Reliability)*1)

    Matched_data = list(zip(Zones_name, MNF, Industry_Usage, Zone_Reliability))
    Matched_data_length = len(Zones_name)

    with open(save_dir + "Zones_MNF_Industry_ReliabilityPtr.csv", 'w') as csv_file:
        file_writer = csv.writer(csv_file)
        file_writer.writerow(['Pressure Zone', 'MNF', 'Industrial Usage', 'Reliability'])
        for row in range(Matched_data_length):
            file_writer.writerow(Matched_data[row])

















