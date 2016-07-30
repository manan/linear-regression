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

class LinearRegression(object):
    """ 
    Contains methods to load data, normalize data, plot data and predict outputs
    """
