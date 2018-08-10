#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : Ryan Fan 
@E-Mail  : ryanfan0528@gmail.com
@Version : v1.0
"""

from csv import DictReader
import csv
from math import exp, log, sqrt

alpha = .05  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 1.1     # L1 regularization, larger value means more regularized
L2 = 1.1     # L2 regularization, larger value means more regularized

D = 2 ** 24           # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

class ftrl_proximal(object):
    """FTRL online learner with the hasing trick using liblinear format data.
    Parameters:
    ----------
    alpha (float): alpha in the per-coordinate rate
    beta (float): beta in the per-coordinate rate
    l1 (float): L1 regularization parameter
    l2 (float): L2 regularization parameter
    n (list of float): buffer to compute feature weights
    w (list of float): feature weights
    z (list of float): lazy weights
    interaction (boolean): whether to use 2nd order interaction or not
    D (long): maximum value for hash
    Reference:
    ----------
    http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    """

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        """Initialize the FTRL class object.
        Parameters:
        ----------
        alpha (float): alpha in the per-coordinate rate
        beta (float): beta in the per-coordinate rate
        l1 (float): L1 regularization parameter
        l2 (float): L2 regularization parameter
        D (long): maximum value for hash
        interaction (boolean): whether to use 2nd order interaction or not
        """
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # Feature related parameters
        self.D = D
        self.interaction = interaction
        
        # initialize weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield int(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        """Predict for features.
        Parameters:
        ----------
        x (list of int): a list of index of non-zero features
        Outputs:
        ----------
        p (float): prediction for input features
        """

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        """Update the model.
        Parameters:
        ----------
        x (list of int): a list of index of non-zero features
        p (float): prediction for input features
        y (int): value of the target
        Outputs:
        ----------
        updates model weights and counts
        """

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g
