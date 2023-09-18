import matplotlib.pyplot as plt

# OS module in Python provides functions for interacting with the operating system.
# You can use os.path() to find some files and os.open() to open it.
import os

# Folder which contains images.
path = "images"

# os.path.join is joining paths into one path.
# For more information about os.path.join visit:
# https://www.geeksforgeeks.org/python-os-path-join-method/
# plt.imread command reads image from a file into an array
# (i.e. picture in general is the "matrix" of pixels, where
# pixel is the triple (a1,a2,a3), 0 <= ai <= 255 meaning
# the percent of red, green, blue in that "area").
# The rgb_picture and nir_picture below are numpy arrays.
# To see more about plt.imread visit:
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html
rgb_picture = plt.imread(os.path.join(path, 'rgb_image.png'))
nir_picture = plt.imread(os.path.join(path, 'nir_image.png'))

print('RGB shape:', rgb_picture.shape)
print('NIR shape:', nir_picture.shape)

# The next part is for showing our pictures.
# The way we plot picture is simply write plt.imshow(picture_name),
# but taking into account that we have two of them we want to plot
# them in a row. In that case we create the place for plotting using subplot command
# (see more https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html).
fig, axes = plt.subplots(1, 2) # creates 1 row, 2 columns.

# Name each column.
axes[0].set(title = 'RGB')
axes[1].set(title = 'NIR')

# Plotting images using imshow command.
axes[0].imshow(rgb_picture)
axes[1].imshow(nir_picture, cmap = 'gray') # Displays a grayscale image.

# Final command to show pictures.
plt.show()
