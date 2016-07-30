#! /usr/bin/python

"""
LinearRegression.py

This class is designed to perform linear regression (gradient descent) 
on a dataset and then be able to predict given new x labels.
"""

__author__ = 'Manan Mehta'
__email__ = 'mehtamanan@icloud.com'
__version__ = '1.0.0'

import numpy as np
import plotly as py
import matplotlib.pyplot as plt

from Exceptions import *

class LinearRegression(object):
    """ 
    Contains methods to load data, normalize data, plot data and predict outputs
    """

    def load_data(self, *files):
        """
        Returns X and y labels as vectors/matrices.

        If two files are given, the first one is assumed
        to be X and the second one assumed Y.
        If one file is given, the last column is assumed
        to be Y and the columns before are assumed X.
        """
        if len(files) == 1:
            dataXY = np.genfromtxt(files[0], delimiter = ',')
            pos = dataXY.shape[1]
            y = dataXY[0:, pos - 1: pos]
            X = dataXY[0:, 0:pos - 1]
        else:
            X = np.genfromtxt(files[0], delimiter = ',')
            y = np.genfromtxt(files[1], delimiter = ',')
        self.X = X
        self.y = y
        return (X, y)

    def plot(self):
        """
        Plots the loaded data
        Throws DataHandlingException if more than one x-label
        Throws NoDataException if data is not loaded
        """
        if not hasattr(self, 'X'):
            raise NoDataException()
        elif self.X.shape[1] > 1:
            raise DataHandlingException()
        else:
            plt.plot(self.X[0:, 0], self.y[0:,0], 'rx')
            plt.show()
