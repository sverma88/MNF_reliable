import numpy as np
import datetime
import matplotlib.dates as mdates
from matplotlib.transforms import Transform
import matplotlib.pyplot as plt



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

    data_rows = data.shape[0]
    data_cols = data_values.shape[1]

    absolute_sum = np.sum(np.absolute(data_values))  #take absolute of each value
    total_sum = np.sum(data_values)

    positive_values = (data_values >0)*data_values
    positive_sum = np.sum(positive_values)

    negative_value = (data_values <0)*data_values
    negative_sum = np.sum(negative_value)

    print("----- All Zones Statistics --------")
    print(" Total Zones ---->", data_cols -1)
    print("All Zones MNF Absolute Sum ---->", absolute_sum)
    print("All Zones MNF Positive Sum ---->", positive_sum)
    print("All Zones MNF Negative Sum ---->", negative_sum)
    print("All Zones MNF Total Sum ---->", total_sum)
    print("All Zones MNF/Day  ---->",total_sum/data_rows)
    print("\n ")
    All_zones_mnf_day = total_sum / data_rows

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
    print("Reliable Zones MNF Absolute Sum  ---->", absolute_sum_indexed)
    print("Reliable Zones MNF Positive Sum ---->", positive_sum_indexed)
    print("Reliable Zones MNF Negative Sum ---->", negative_sum_indexed)
    print("Reliable Zones MNF Total Sum ---->", total_sum_indexed)
    print("Reliable Zones MNF/Day ---->", total_sum_indexed / data_rows)
    print("\n ")
    Reliable_zones_mnf_day = total_sum_indexed / data_rows


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
        # print("MNF Absolute Sum  ---->", absolute_sum_indexed)
        # print("MNF Positive Sum ---->", positive_sum_indexed)
        # print("MNF Negative Sum ---->", negative_sum_indexed)
        # print("MNF Total Sum ---->", total_sum_indexed)
        print("MNF/Day ---->", total_sum_indexed / data_rows)

        Categorical_zones_mnf_day.append(total_sum_indexed / data_rows)


    return Categorical_zones_mnf_day, Reliable_zones_mnf_day, All_zones_mnf_day








