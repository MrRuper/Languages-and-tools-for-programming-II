# This code is a modified code of dr Andrzej Mizera laboratory materials for the
# 'Introduction to the Machine learning' course. Here I added my own comments and changed
# code a bit for better understanding.

import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid function in two different ways.

# Classical way to define functions.
def sigmoid_function(z):
    answer = z / (1 + np.exp(-z))

    return answer

# Using lambda functions (to see more go here: https://realpython.com/python-lambda/)
S = lambda z : z / (1 + np.exp(-z))

interval_size = 30

# This command will split interval [-6,6] onto interval_size equal pieces
# and store all endpoints to the z. The array z is the numpy array.
z = np.linspace(-6, 6, interval_size)

# subplots() command create the fig and ax objects.
# ax is axes, fig is figure.
fig, ax = plt.subplots()

# set_facecolor sets color of the figure. The array inside of this command
# is the color (i.e. weights of 4 basics color: red, black, green, yellow).
# You can just write for example fig.set_facecolor("green").
fig.set_facecolor([0.9, 0.4, 0.5, 0.6])

# Axes setting.
ax.set(title='Activation Function', ylabel='Sigmoid function', xlabel='z')
ax.set_xlim([-6, 6])
ax.set_ylim([-1, 1])

# Plotting the sigmoid function. The most important part is
# ax.plot(z, sigmoid_function(z)) and it will take the array z
# and to each element of z it will apply S function. Next it will join
# that points with lines. Another part of that command is the optional part.
# Note: if you use the lambda function you have to type:
# ax.plot(z, list(map(S,z)).
# Note: the 'go-' command will mark on the curve the values of S on each point
# of the z.
ax.plot(z, sigmoid_function(z), 'go-', linewidth = 2, label = 'Sigmoid Curve')

# This will add grid to the figure object (it does not modify the sigmoid curve).
ax.grid(True, color='black')

# This command displays the legend on your plot. In this case when only one curve
# is displayed, it is not very important.
ax.legend()

# This command is displaying whole plot window.
plt.show()





