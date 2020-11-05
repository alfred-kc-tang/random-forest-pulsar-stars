import ast
import csv

import numpy as np

from decision_tree import DecisionTree

"""
Assumptions of this module:
1. X is assumed to be a matrix with n rows and d columns where n is the number
   of total records and d is the number of features of each record.
2. y is assumed to be a vector of labels of length n.
3. XX is similar to X, except that XX also contains the data label for each
record.
"""

class RandomForest(object):
    num_trees = 0
    decision_trees = []
    # The bootstrapping datasets for trees. It is a list of lists, where each
    # list in bootstraps_datasets is a bootstrapped dataset.
    bootstraps_datasets = []
    # The true class labels, corresponding to records in the bootstrapping
    # datasets. It is a list of lists, where the i-th list contains the labels
    # corresponding to records in the i-th bootstrapped dataset.
    bootstraps_labels = []

    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree() for i in range(num_trees)]

    def _bootstrapping(self, XX, n):
        """Create a sample data of size n by sampling with replacement from XX.
        
        The corresponding class labels for the sampled records (for training
        purposes) is also returned.
        Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)

        Args:
            XX (list): the original dataset.
            n (int): the size of the bootstrapped sample.

        Returns:
            samples (list): the sampled dataset.
            labels (list): class labels for the sampled records.
        """
        samples = []
        labels = []
        randint = np.random.randint(len(XX), size=n)
        for i in randint:
            samples.append(XX[i][:-1])
            labels.append(XX[i][-1])
        return (samples, labels)

    def bootstrapping(self, XX):
        """Initialize the bootstap datasets for each tree.
        
        Args:
            XX (list): the original dataset.
        """
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def fitting(self):
        """Train `num_trees` decision trees using the bootstraps datasets."""
        for i in range(self.num_trees):
            self.decision_trees[i].learn(self.bootstraps_datasets[i], 
                                         self.bootstraps_labels[i])

    def voting(self, X):
        """Classify from the trees that consider the record as out-of-bag.

        Args:
            X (list): data containing all the explanatory variables / 
              attributes / predictors.
        
        Returns:
            y (list): predicted labels.
        """
        y = []

        for record in X:
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                if record not in dataset:
                    # Find the set of trees that consider the record as an out-
                    # of-bag sample.
                    OOB_tree = self.decision_trees[i]
                    # Classify using each of the above trees.
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)

            counts = np.bincount(votes)

            # Handle the case where the record is not an out-of-bag sample for
            # any tree: to randomly pick a tree to classify.
            if len(counts) == 0:
                y = np.append(y, self.decision_trees[np.random.randint(
                        len(self.decision_trees))].classify(record))
            # Otherwise, use majority vote to classify.
            else:
                y = np.append(y, np.argmax(counts))

        return y


def main():
    X = list()
    y = list()
    XX = list()  # Contains both attributes and labels
    numerical_cols = set([i for i in range(0, 9)])  # indices of numeric columns

    print("reading pulsar_stars")
    with open("pulsar_stars.csv") as f:
        next(f, None)
        for line in csv.reader(f, delimiter=","):
            xline = []
            for i in range(len(line)):
                if i in numerical_cols:
                    xline.append(ast.literal_eval(line[i]))
                else:
                    xline.append(line[i])

            X.append(xline[:-1])
            y.append(xline[-1])
            XX.append(xline[:])

    # Minimum forest_size should be 10.
    forest_size = 10

    # Initialize a random forest.
    randomForest = RandomForest(forest_size)

    # Create the bootstrapping datasets.
    print("creating the bootstrap datasets")
    randomForest.bootstrapping(XX)

    # Build trees in the forest.
    print("fitting the forest")
    randomForest.fitting()

    # Estimate an unbiased error of the random forest based on out-of-bag (OOB)
    # error estimate.
    y_predicted = randomForest.voting(X)

    results = [prediction == truth for prediction, truth in zip(y_predicted, y)]
    accuracy = float(results.count(True)) / float(len(results))

    print("accuracy: %.4f" % accuracy)
    print("OOB estimate: %.4f" % (1 - accuracy))


if __name__ == "__main__":
    main()
