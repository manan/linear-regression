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
            dataXY = np.matrix(np.genfromtxt(files[0], delimiter = ','))
            pos = dataXY.shape[1]
            y = dataXY[:, pos -1]
            X = dataXY[:, :pos - 1]
        else:
            X = np.matrix(np.genfromtxt(files[0], delimiter = ','))
            y = np.matrix(np.genfromtxt(files[1], delimiter = ','))
        ones = np.matrix(np.ones(shape = (X.shape[0], 1)))
        self.X = np.hstack((ones, X))
        self.y = y
        self.theta = np.matrix(np.zeros(shape = (self.X.shape[1], 1)))
        return (X, y)

    def plot(self):
        """
        Plots the loaded data
        Throws DataHandlingException if more than one x-label
        Throws NoDataException if data is not loaded
        """
        if not hasattr(self, 'X'):
            raise NoDataException()
        elif self.X.shape[1] > 2:
            raise DataHandlingException()
        else:
            plt.plot(self.X[:,1], self.y, 'rx')
            plt.show()

    def normalize(self):
        """
        Normalizes the data such that the mean is 0
        and the data fits -0.5<x<0.5 roughly and
        returns the new X matrice
        Stores the mean and range of all x features 
        except the 1s added for vectorization

        Throws NoDataException is data is not loaded
        """
        if not hasattr(self, 'X'):
            raise NoDataException()
        else:
            self.X_mean = np.matrix(np.zeros(shape=(1,self.X.shape[1] - 1)))
            self.X_range = np.matrix(np.zeros(shape=(1,self.X.shape[1] - 1)))
            for i in range(self.X.shape[1]):
                if not i == 0:
                    tempX = self.X[:, i]
                    meanX = np.mean(tempX)
                    self.X_mean[0, i - 1] = meanX
                    rangeX = max(tempX) - min(tempX)
                    self.X_range[0, i - 1] = rangeX
                    tempX = (tempX - meanX)/rangeX
                    self.X[:, i] = tempX
            return self.X
        pass

    def compute_cost(self):
        """
        Cost Function of the Linear Regression algorithm.
        Returns (1/2m) * [summation] (H(x)-y)^2
        Halved average of summed squared error
        """
        m = self.X.shape[0]
        preds = self.X * self.theta
        errors = preds - self.y
        sq_errors = np.square(errors)
        summation = np.sum(sq_errors)
        J = ((1.0/(2*m))*summation)
        return J
