# By D we denote data and by L we denote labels.
# Splitting data (or generating data for training) is very important in ML,
# thus we need to see standard methods / functions to do so.
# Each new method will start with the comment in which I write:
# (1) Name of the method / function,
# (2) the way how this method works,
# (3) link to google page where you can find more information.

# The sklearn library is the standard library for Machine Learning, and
# it is efficient tool for predictive data analysis. It is built on
# NumPy, SciPy, and matplotlib. It has a lot of useful things and for sure
# we will study most of them. The sklearn has:
# (1) Classification - identifying which category an object belongs to.
# (2) Regression - predicting a continuous-valued attribute associated with an object.
# (3) Clustering - automatic grouping of similar objects into sets.
# (4) Dimensionality reduction - reducing the number of random variables to consider.
# (5) Model selection - comparing, validating and choosing parameters and models.
# (6) Preprocessing - feature extraction and normalization.
# For more information visit: https://scikit-learn.org/stable/

# Here we use only sklearn.model_selection
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

print("------- train_test_split ----------")

# Define data set and its labels.
D1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])
L1 = np.array([0, 0, 0, 0, 0, 1, 1])

# This command splits X, Y with proportion 80 / 20
# it will also split labels Y in the way that
# X_train and y_train will match together.
D1_train, D1_test, T1_train, T1_test = train_test_split(D1, L1, test_size = 0.2)

print(f"Standard split command: \n")
print(f"D1_train: \n {D1_train}")
print(f"T1_train: \n {T1_train}")
print(f"D1_test: \n {D1_test}")
print(f"T1_test: \n {T1_test}")
print("\n")

print("-------- train_test_split with stratify option ----------")

# Let us add new data in the next way:
# Create array t = [0,...,17] by typing np.arange(17),
# reshape this array to get [(0,1), (2,3), ..., (16,17)] by
# typing t.reshape(9,2).
# To avoid 0 as a value add 1 to whole t.
D2 = np.arange(18).reshape((9, 2)) + 1
T2 = [0, 0, 0, 0, 0, 0, 1, 1, 1]

# As you see T2 has a lot of 0 at the beginning and 1 at the end. While splitting
# You may get something like that:
# split1 = [almost all is zero], split2 = [almost all is 1],
# and thus the proportion of different items is broken. This may cause an invalid training in the future.
# To avoid that we use stratify = T2 to ensure that in train and test arrays we get
# a similar number of 1 and 0 in each of them.
D2_train, D2_test, T2_train, T2_test = train_test_split(D2, T2, test_size = 0.33, stratify = T2)

print(f"Split with stratify: \n")
print(f"D2_train: \n {D2_train}")
print(f"T2_train: \n {T2_train}")
print(f"D2_test: \n {D2_test}")
print(f"T2_test: \n {T2_test}")
print("\n")

print("-------- ShuffleSplit -----------")

# Again we use the same trick as before.
D3 = np.arange(16).reshape((8, 2)) + 1
T3 = np.array([0, 1, 0, 1, 0, 1, 1, 1])

# ShuffleSplit will split our D3 five times on train_set and test_set.
# Random_state = 0 means that we turn on randomness of the splits.
# In random_state you can put arbitrary number (or None).
ShuffleSplit_object = ShuffleSplit(n_splits = 5, test_size = .25, random_state = 0)

# Returns the number of splitting iterations in the cross-validator.
# In our case it is 5.
print(f"The number of splits in Shuffle split: {ShuffleSplit_object.get_n_splits(D3)} ")

# This command will generate our split.
# Note that our splits are just indexes, thus we don't need
# to split T3.
ShuffleSplit_object.split(D3)

# We print result in the loop.
# enumerate command takes in general some iterable object (for example list)
# and returns generator of {index, value}. For example
# enumerate([4,5,6]) = {[0,4], [1,5], [2,6]}.
# Generator is a special Python object which allow us to declare
# a function that behaves like an iterator, i.e. can be used in a for loop.
# To see more about generators visit: https://wiki.python.org/moin/Generators
for i, (train_index, test_index) in enumerate(ShuffleSplit_object.split(D3)):
    print(f"Fold {i + 1}:")
    print(f"Train: index={train_index}")
    print(f"Test:  index={test_index}")

print("\n")

print("---------- Kfold --------------")

# The KFold method is very similar to the ShuffleSplit.
# The difference is that KFold is splitting data better.
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])
y = np.array([0, 0, 0, 0, 1, 1, 1])

# Create an KFold object.
KFold_object = KFold(n_splits = 3, shuffle = True)

# Printing all information about KFold_object.
print(KFold_object)

# Generate splits.
KFold_object.split(X)

# As in the Shuffle split we use for loop to see splits.
for i, (train_index, test_index) in enumerate(KFold_object.split(X)):
    print(f"Fold {i + 1}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

print("\n")


print("---------- Stratified KFold ---------")

# Add examples

print("---------- RepeatedStratifiedKFold -------------")

# Add exapmles


