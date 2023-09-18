import time

import numpy as np
import matplotlib.pyplot as plt

# To plot a graph of one dimensional function f we simply took some array A,
# we found f(A) and after that we plotted set of pairs (A, f(A)) joining them
# with line. Unfortunately the 3D graph is not as easy as 2D.

# To plot some 3D graph we firstly have to define grid of points,
# they will serve as an argument points.

# The sense of the meshgrid is next:
# we take some points A on x-axe and some points B on y-axe.
# For example A = [-10, -9.9, -9.8, .... , 9.9, 10],
# B = [-10, -9.9, -9.8, .... , 9.9, 10] then
# np.mgrid(A, B) will create ALL possible ordered pairs (a,b)
# for a in A and b in B.
# Note that mgrid will code that pairs in a tricky way:
# values x will keep in the arguments[0] and values y will
# keep in arguments[1]. To see more visit:
# https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html
arguments = np.mgrid[-10:10:0.1, -10:10:0.1]

# Define the SE function (squared error functon)
# Weights of the model
x = np.array([0.5, 0.3])


# Squared error function. Here x is an array of "weights".
# Here x, y are arguments,
# weights - array of length at least 2.
def SE_function(z, y, weights):
    return (1 - 1 / (1 + np.exp(-(z * weights[0] + y * weights[1])))) ** 2


# Use mgrid as arguments. The SE keeps now values SE_function(arguments).
SE = SE_function(arguments[0], arguments[1], [0.5, 0.3])
SE = np.array(SE)

# Now we can plot that points using plt library.
# See more: https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
fig = plt.figure()

# To show different ways to plot we reserve already
# place for three surfaces.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw = {'projection': '3d'})

# First surface.
ax1.set(title = 'Squared Error',
        zlabel ='SE',
        xlabel = 'argument_0',
        ylabel = 'arguments_1')
ax1.plot_surface(arguments[0], arguments[1], SE)

# Second surface.
ax2.set(title = 'Squared Error',
        zlabel = 'SE',
        xlabel = 'argument_0',
        ylabel = 'arguments_1')
ax2.plot_surface(arguments[0], arguments[1], SE,
                 rstride = 20,
                 cstride = 20,
                 cmap = 'viridis',
                 edgecolor = 'white')

# Third surface.
ax3.set(title='Squared Error',
        zlabel='SE',
        xlabel='argument_0',
        ylabel='argument_1')
ax3.plot_wireframe(arguments[0], arguments[1], SE,
                   color = 'black',
                   rstride = 20,
                   cstride = 20)

# Saves figure to the surface.png file.
plt.savefig('surface.png')
plt.show()
