import ast
import collections

import numpy as np

from util import entropy, information_gain, partition_classes

class DecisionTree(object):
    def __init__(self):
        # Initialize the tree as an empty dictionary.
        self.tree = {}


    def build_node(self, X, y):
        """Build nodes for decision trees.

        Args:
            X (list): data containing all the explanatory variables / 
              attributes / predictors.
            y (list): labels.

        Returns:
            dict: node storing the partition and corresponding split attribute
              and splite value.
        """
        # Makes it easy to manipulate.
        X_array = np.array(X)
        # Find the classes of y and their counts.
        classes, counts = np.unique(y, return_counts=True)

        # If all data points in X has the same class value y, then return a
        # leaf node that predicts y as an output.
        if len(classes) == 1:
            return classes[0]

        # If all data points in X have the same attributes, then return a leaf
        # node that predicts the majority of the class values in y as an output.
        if all(x == X[0] for x in X):
            return classes[np.where(counts == max(counts))[0][0]]

        info_gain = []

        for i in range(len(X[0])):
            # If the attribute is categorical, then use the mode, the most
            # frequent level, as the split value. If there are more than one
            # mode, then select the first one alphabetically.
            if isinstance(X[0][i], str):
            # DON'T: type(X[0][i]) == str, as per PEP 8
                counter = collections.Counter(X_array[:,i])
                mode = counter.most_common(1)[0][0]
                potential_partitions = partition_classes(X, y, i, mode)
            # If the attribute is numeric, then use the mean as the split value.
            else:
                mean = np.mean(np.array(X_array[:,i]).astype(np.float))
                potential_partitions = partition_classes(X, y, i, mean)
            info_gain.append(information_gain(y, [potential_partitions[2],
                                                  potential_partitions[3]]))

        # If there is no more information gain from the split, i.e. the max of
        # info_gain is still zero, then return a leaf node that predicts the
        # majority of the class values in y as an output.
        if max(info_gain) == 0:
            return classes[np.where(counts == max(counts))[0][0]]

        # Determine the attribute with the maximum information gain
        best_attr = info_gain.index(max(info_gain))

        # If the best attribute is categorical, then use its mode as the split
        # value.
        if isinstance(X[0][best_attr], str):
        # DON'T: type(X[0][best_attr]) == str, as per PEP 8
            counter = collections.Counter(X_array[:,best_attr])
            split_val = counter.most_common(1)[0][0]
            partitions = partition_classes(X, y, best_attr, split_val)

        # If the best attribute is numeric, then use its mean as the split
        # value.
        else:
            split_val = np.mean(X_array[:,best_attr])
            partitions = partition_classes(X, y, best_attr, split_val)

        # Run the build_node() function recursively to continue building node
        # if it is not a leaf one.
        return {"split_attribute": best_attr,
                "split_value": split_val,
                "left": self.build_node(partitions[0], partitions[2]),
                "right": self.build_node(partitions[1], partitions[3])}


    def learn(self, X, y):
        """Train the decision tree using X and y
        
        Args:
            X (list): data containing all the explanatory variables / 
              attributes / predictors.
            y (list): labels.

        Returns:
            self.tree (dict): the decision tree built.
        """
        self.tree = self.build_node(X, y)


    def classify(self, record):
        """Classify the record using self.tree and return the predictions

        Args:
            record (dict): the decision tree built.

        Returns:
            results (int): predicted labels.
        """
        results = self.tree
        while type(results) == dict:
            if isinstance(record[results["split_attribute"]], str):
            # DON'T: type(record[results["split_attribute"]]) == str, as per
            # PEP 8
                if record[results["split_attribute"]] == results["split_value"]:
                    results = results["left"]
                else:
                    results = results["right"]
            else:
                if record[results["split_attribute"]] <= results["split_value"]:
                    results = results["left"]
                else:
                    results = results["right"]
        return results
