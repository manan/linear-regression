#! /usr/bin/python

import LinearRegression as lr
import numpy as np
import sys

def main(filename):
    classifier = lr.LinearRegression()
    classifier.load_data(filename)
    print "Computed cost: ", classifier.compute_cost()
    print "Gradient descent being carried out ..."
    classifier.gradient_descent()
    print "Theta values deduced: ", classifier.theta
    print "New computed cost: ", classifier.compute_cost()
    print "Prediction for 35, 000 people: ", classifier.predict(np.matrix('3.5')) * 10000
    print "Prediction for 70, 000 people: ", classifier.predict(np.matrix('7.0')) * 10000
    if classifier.can_plot():
        classifier.plot()

if __name__ == '__main__':
    f = sys.argv[1]
    main(f)
