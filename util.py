import numpy as np
from scipy import stats

def entropy(class_y):
    """Compute entropy for information gain.
    
    For example, given a list of [0,0,0,1,1,1,1,1,1], it returns 0.92.

    Args:
        class_y (list): list of class labels (0's and 1's)

    Returns:
        entropy (float): entropy.
    """
    entropy = 0
    classes, counts = np.unique(class_y, return_counts=True)
    probs = counts/len(class_y)
    for p in probs:
        if p != 0:
            entropy -= p * np.log2(p)
    return entropy


def partition_classes(X, y, split_attribute, split_val):
    """Partition the data and labels based on the split value given.
    
    For example, given the following X and y:
    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]
    Columns 0 and 2 in X represent numeric attributes, while column 1 is a
    categorical attribute.

    Consider the case in which we call the function with split_attribute = 0
    and split_val = 3 (mean of column 0) and then we divide X into two lists -
    X_left, where column 0 is <= 3, and X_right, where column 0 is > 3.
    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]
    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]
    
    Consider another case in which we call the function with split_attribute
    = 1 and split_val = 'bb' and then we divide X into two lists, one where
    column 1 is 'bb', and the other where it is not 'bb'.
    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]
    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]

    Args:
        X (list): data containing all the explanatory variables / attributes /
          predictors.
        y (list): labels.
        split_attribute (int): column index of the attribute to split on.
        split_val (float or str): either a numerical or categorical value to
          divide the split_attribute.

    Returns:
        X_left (list): X partitions based on split_attribute and split_value.
        X_right (list): X partitions based on split_attribute and split_value.
        y_left (list): y partitions corresponding to X_left.
        y_right (list): y partitions corresponding to X_right.
    """
    X_left = []
    X_right = []
    y_left = []
    y_right = []


    if isinstance(X[0][split_attribute], str):
    # DON'T: if type(X[0][split_attribute]) == str, as per PEP 8
        for i in range(len(X)):
            if X[i][split_attribute] == split_val:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])
    else:
        for i in range(len(X)):
            if X[i][split_attribute] <= split_val:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

    return (X_left, X_right, y_left, y_right)


def information_gain(previous_y, current_y):
    """Compute the information gain from the partition on the previous_y labels
    that resulted in the current_y labels.

    For example, given the following previous_y and current_y:
    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]
    it returns 0.45915.
    Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf

    Args:
        previous_y (list): the distribution of original labels (0's and 1's)
        current_y (): the distribution of labels after splitting based on a
          particular split attribute and split value
    """
    info_gain = entropy(previous_y)
    previous_len = len(previous_y)
    for subset in current_y:
        info_gain -= (entropy(subset)*len(subset)) / previous_len

    return info_gain
