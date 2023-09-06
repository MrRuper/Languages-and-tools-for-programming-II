# This code is a modified code of dr Andrzej Mizera laboratory materials for the
# 'Introduction to the Machine learning' course. Here I added my own comments and changed
# code a bit for better understanding.

import numpy as np

# Read data from ionosphere.data.csv file.
file_name = 'ionosphere.data.csv'

# The np.genfromtxt command will convert data to the numpy array.
# delimiter - the string used to separate values
# dtype - type of readed data (so data_array will be a numpy array of strings)
# skip_header - the number of lines to skip at the beginning of the file.
data_array = np.genfromtxt(file_name, delimiter = ',', dtype = str, skip_header = 0)

print(data_array)

# Copy to the X all data_array matrix except of the last column.
# Last column has 'g' or 'b' (see ionosphere.data.csv)
X = data_array[:, :-1]

# Change type of X to the float to make further computations.
# It is better to convert to the special numpy.float type.
X = X.astype(np.float32)

# Define two variables describing the size of X.
number_of_rows = X.shape[0]
number_of_columns = X.shape[1]

# Define two empty lists for holding: average of each row
# and deviation of each row.
average_of_rows = []
standard_deviation_of_rows = []

# Adding in a loop. The append method adds value to the end of
# the list.

for i in range(number_of_rows):
    average_of_rows.append(np.average(X[i, :]))
    standard_deviation_of_rows.append(np.std(X[i, :]))

# Convert lists to the numpy.array for optimizing further computations.
average_of_rows = np.array(average_of_rows)
standard_deviation_of_rows = np.array(standard_deviation_of_rows)

print('Average values:', average_of_rows)
print('Standard deviation values:', standard_deviation_of_rows)

# For each row find all elements which are less then
# the average value of that feature.
remember_values_lower_than_average = []

for i in range(number_of_rows):
    remember_row = []

    for j in range(number_of_columns):
        if X[i][j] < average_of_rows[i]:
            remember_row.append(X[i][j])

        # Each row has to have the same length to retype (see (*) bellow),
        # so I added None value if nothing else is added.
        else:
            remember_row.append(None)

    remember_values_lower_than_average.append(remember_row)

# (*): Here if you want to retype the list of list to the numpy array
# you need to ensure that every list is of the same length.
remember_values_lower_than_average = np.array(remember_values_lower_than_average)
print(remember_values_lower_than_average)
