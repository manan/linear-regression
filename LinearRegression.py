#! /usr/bin/python

"""
LinearRegression.py

This class is designed to perform linear regression (gradient descent) 
on a dataset and then be able to predict given new x labels.
"""

__author__ = 'Manan Mehta'
__email__ = 'mehtamanan@icloud.com'
__version__ = '1.0.0'

import sys
import numpy as np
import plotly as py
import matplotlib.pyplot as plt

from Exceptions import *

class LinearRegression(object):
    """ 
    Contains methods to load data, normalize data, plot data and predict outputs
    """

    def __init__(self):
        self.norm = False #Data has not been normalized
        self.gd = False #Gradient descent has not been carried out

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

    def plot(self):
        """
        Plots the loaded data and the prediction line
        Throws DataHandlingException if more than one x-label
        Throws NoDataException if data is not loaded
        """
        if not hasattr(self, 'X'):
            raise NoDataException()
        elif self.X.shape[1] > 2:
            raise DataHandlingException()
        else:
            if not self.gd:
                self.gradient_descent()
            X = self.X[:,1]
            y = self.X * self.theta
            plt.plot(self.X[:,1], self.y, 'rx', X, y, 'g-')
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
        if not self.norm:
            self.norm = True
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
            pass
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

    def gradient_descent(self):
        """
        Carries out gradient descent to compute the
        values of theta such that cost function is 
        reduced (= error reduced)
        """
        if not self.gd:
            self.gd = True
            num_iters = 5000
            alpha = 0.01
            m = self.X.shape[0]
            n = self.X.shape[1]
            history = np.matrix(np.zeros(shape = (1, num_iters)))
            for i in range(num_iters):
                delta = np.matrix(np.zeros(shape = (n, 1)))
                for j in range(m):
                    Xi = self.X[j,:].T
                    pred = self.theta.T * Xi
                    error = pred - self.y[j,0]
                    delta += (error[0,0] * Xi)
                self.theta = self.theta - ((alpha/m)*delta)
                history[0,i] = self.compute_cost()
            self.history = history

    def predict(self, x):
        if not self.norm:
            one = np.matrix('1')
            x = np.hstack((one, x))
            pred = self.theta.T * x.T
            return pred[0,0]
        else:
            return 0 # stub (Normalization is not working as fast)
        pass

    def can_plot(self):
        if not hasattr(self, 'X'):
            return False
        elif self.X.shape[1] > 2:
            return False
        else:
            return True
