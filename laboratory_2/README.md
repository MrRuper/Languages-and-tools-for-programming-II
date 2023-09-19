# What is this?

This are modified notes from the second laboratory of "Introduction to the Machine Learning" course provided by dr Andrzej Mizera.

# What is the goal of that laboratory?

The goal is to show how to split data (see [data_split.py](https://github.com/MrRuper/Languages-and-tools-for-programming-II/blob/main/laboratory_2/data_split.py)), 
how to plot a 3D surface basing on some function (see [Sigmoid_3D_plotting.py](https://github.com/MrRuper/Languages-and-tools-for-programming-II/blob/main/laboratory_2/Sigmoid_3D_plotting.py))
and how to read image from the folder and use it in Python code (see [image_plotting.py](https://github.com/MrRuper/Languages-and-tools-for-programming-II/blob/main/laboratory_2/image_plotting.py)).

# What does this folder contain?

This folder contains:

- **data_split.py** - here standard data split methods are shown. Data split is a major and well known problem in Machine learning. To train some model we simple
split data on two sets: train and test data (train dataset we further split on train and validation datasets). In this file **train_test_split**, **ShuffleSplit**, **KFold** and other similar methods are presented,
- **images** - this folder contains two images: **nir_image.png** and **rgb_image.png**. We use them in [image_plotting.py](https://github.com/MrRuper/Languages-and-tools-for-programming-II/blob/main/laboratory_2/image_plotting.py) file,
- **image_plotting.py** - here we show how to plot picture in python code **images** folder,
- **Sigmoid_3D_plotting.py** - here we plot the 3D surface using a **meshgrid** command and this plot we save to the **surface.png** file.

# How to use it?

You can simple download this whole folder (the **surface.png** is optional) and run every file independently to see results or you can use it as a materials for styding.
