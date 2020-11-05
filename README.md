# [Random Forest on Pulsar Stars](https://alfred-kctang.github.io/random-forest-pulsar-stars/)

## Table of Contents

* [Introduction](#introduction)
* [Data Source](#data-source)
* [Methodology](#methodology)
* [Conclusion](#conclusion)
* [Reference](#reference)
* [Keywords](#keywords)
* [Coding Style](#coding-style)
* [License](#license)

## Introduction

The random forest algorithm in this project is written using only NumPy and SciPy, as well as the Python Standard Library.

## Data Source

The data set is from a [Kaggle competition](https://www.kaggle.com/colearninglounge/predicting-pulsar-starintermediate). The dataset has been cleaned that data with missing attributes were removed. Each line in the csv file describes an instance using 9 columns: the first 8 columns represent the attributes of the pulsar candidate, namely mean of the integrated profile, standard deviation of the integrated profile, excess kurtosis of the integrated profile, skewness of the integrated profile, mean of the DM-SNR curve, standard deviation of the DM-SNR curve, excess kurtosis of the DM-SNR curve, skewness of the DM-SNR curve; whereas the last column is the response variable that tells us if the observation is a pulsar or not (1 means it is a pulsar, 0 means it is not).

## Methodology

In random forests, one of the main questions is how to choose the attribute and its value for splitting nodes. Choice of a given attribute rather than others is determined by a measure called entropy, which quantifies the average amount of information or uncertainty inherent in the possible outcomes for a given response variable or label. Statistically speaking, it captures the degree of "impurity" of the distribution: fewer classes are more probable than others results in lower entropy, whereas all classes are equally likely leads to higher entropy. Thus, whether a split on a given attribute is the best is determined by whether the entropy is lowest given that split. Concerning the split value, the mode is chosen when the attribute is categorical or mean is selected when the attribute is numeric, for the sake of simplicity. For more technical details, please refer to the CMU lecture slides for reference.

On the other hand, explicit cross-validation or use of a separate test set for performance evaluation is not necessarily required for random forests. Out-of-bag (OOB) error estimate is an alternative that has been shown to be reasonably accurate and unbiased, and thus being adopted in this project. What the "bag" means in the term is that data samples that are used for training are sub-sampled using bagging, i.e. bootstrap aggregating. To put it simply, OOB error estimate measures the mean prediction error on each training sample (or observation) that are predicted using only the trees whose training bootstrap samples did not consist of this very sample.

## Conclusion

The algorithm took 108 seconds to run with 97.54% accuracy on the data. In general, a random forest performs better than a single decision tree due to its stability, for the former suffers less overfitting to a particular data set than the latter, by taking the average predictions from trees learned from random samples and split by a random set of features.

## References

[CMU Lecture Slides](www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf).

Breiman L, Cutler A. [Random Forests](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm).

Cutler A, Zhao G. [PERT - Perfect Random Tree Ensembles](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.2940&rep=rep1&type=pdf).

## Keywords

Entropy; Information Gain; Out-of-bag (OOB) error estimate; Random Forest.

## Coding Style

This project adopts the recommended practice of [PEP 8](https://www.python.org/dev/peps/pep-0008/) as well as [PEP 257](https://www.python.org/dev/peps/pep-0257/), and also by reference to [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html).

## License

This repository is covered under the [MIT License](https://github.com/alfred-kctang/random-forest-pulsar-stars/blob/master/LICENSE).
