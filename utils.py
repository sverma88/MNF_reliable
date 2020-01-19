import numpy as np

def compute_error(Y_true, Y_Pred):

    '''

    :param Y_true:  Ground Truth
    :param Y_Pred:  Predicted Values
    :return:        Mean of the error
    '''


    Diff        =  Y_true - Y_Pred
    Mean_Diff   = np.mean(Diff)
    Sum_Diff    = np.sum(Diff)

    NumPositives = np.sum(Diff >= 0)
    NumNegatives = np.sum(Diff <= 0)
    NumZeros     = np.sum(Diff == 0)

    print("The Mean of the Error is: {}".format(Mean_Diff))
    print("Number of Positive Vales in the corrected MNF are: {}".format(NumPositives))
    print("Number of Negative Vales in the corrected MNF are: {}".format(NumNegatives))
    print("Number of Zero Vales in the corrected MNF are: {}".format(NumZeros))
    # print("The sum of the Error is: {}".format(Sum_Diff))

    return Diff, NumPositives, NumNegatives

def count_p(data):

    '''

    :param data:  series of numbers
    :return:      number of positives in the series
    '''

    NumPositives = np.sum(data >= 0)

    return NumPositives

def count_n(data):

    '''

    :param data:  series of numbers
    :return:      number of negatives in the series
    '''

    NumNegatives= np.sum(data < 0)

    return NumNegatives


def compute_mean_stats(Y_true, Data_Ranking):

    '''

    :param Y_true:  Ground Truth
    :param Y_Pred:  Non-Residental Usage Ratio Ranking
    :return:
    '''


    Ordered_Y_10       =  Y_true[Data_Ranking[:10]]
    Ordered_Y_20       =  Y_true[Data_Ranking[:20]]
    Ordered_Y_30       =  Y_true[Data_Ranking]

    Mean_Y10           = np.mean(Ordered_Y_10)
    Mean_Y20           = np.mean(Ordered_Y_20)
    Mean_Y30           = np.mean(Ordered_Y_30)


    print("The Mean of the top 10 zones is: {}".format(Mean_Y10))
    print("The Mean of the top 20 zones is: {}".format(Mean_Y20))
    print("The Mean of the top 30 zones is: {}".format(Mean_Y30))

